# Conectar el modelo de IA — qué hay que hacer en DECA-Back

Hola. El modelo de ECG ya está listo y corre como un servicio HTTP aparte. Esto es lo que
falta del lado del backend para conectarlo.

**El resumen**: hoy `POST /analisis` recibe `{ pacienteId, porcentaje }` y el médico tipea
el porcentaje a mano. Tiene que pasar a recibir el **archivo del ECG**, mandárselo al
servicio de inferencia, y guardar lo que vuelve. El chequeo de `estaAsignado`, la
notificación y los dos `GET` no cambian.

La referencia completa de la API del modelo está en [`API.md`](API.md) — acá va sólo lo que
hay que tocar.

---

## Los 6 cambios

| # | Archivo | Qué |
|---|---|---|
| 1 | `.env.example`, `src/config/env.js` | Dos variables nuevas |
| 2 | `src/config/schema.sql` | Tres columnas en `analisis` |
| 3 | `src/services/inferenciaService.js` | **Archivo nuevo**: el cliente HTTP |
| 4 | `src/services/analisisService.js` | Guardar los campos nuevos |
| 5 | `src/controllers/analisisController.js` | Recibir el archivo, llamar al modelo |
| 6 | `src/routes/analisisRoutes.js` | `multer` para el multipart |

Más los tests, que están listados abajo con nombre y apellido.

---

## 1. Variables de entorno

```bash
# .env.example
DECA_INFERENCIA_URL=http://localhost:8000
DECA_API_TOKEN=
```

```js
// src/config/env.js — dentro del objeto que ya exporta
  inferencia: {
    url: process.env.DECA_INFERENCIA_URL,
    token: process.env.DECA_API_TOKEN,
  },
```

`DECA_API_TOKEN` es un secreto compartido entre el backend y el servicio de inferencia. El
servicio no tiene usuarios ni sesiones: confía en que quien le habla es el backend, y nada
más. **No lo expongas al frontend.**

---

## 2. Migración

Agregar al final del bloque de `analisis` en `schema.sql` (antes del `DROP TABLE` final):

```sql
ALTER TABLE analisis ADD COLUMN IF NOT EXISTS banda VARCHAR(6)
  CHECK (banda IN ('alta', 'media', 'baja'));
ALTER TABLE analisis ADD COLUMN IF NOT EXISTS score NUMERIC(8, 6);
ALTER TABLE analisis ADD COLUMN IF NOT EXISTS modelo_sha VARCHAR(16);
```

Después `npm run migrate`. Es idempotente como el resto del archivo.

**Qué va en cada columna:**

- **`porcentaje`** (la que ya existe) → el **percentil de riesgo**. Entra tal cual en
  `NUMERIC(5,2)` y el `CHECK (0..100)` sigue valiendo.
- **`banda`** → `alta` / `media` / `baja`. **Es lo más importante que devuelve el modelo**;
  si sólo guardás el percentil, perdés lo único que tiene un valor predictivo medido.
- **`score`** → la salida cruda de la red. Sólo para trazabilidad. **No se muestra nunca**
  (ver "Cosas que te van a morder", abajo).
- **`modelo_sha`** → qué versión del modelo produjo ese análisis. Si algún día se cambia el
  modelo, esto es lo que te permite saber qué análisis viejos son comparables con los
  nuevos.

---

## 3. `src/services/inferenciaService.js` — archivo nuevo

```js
import env from '../config/env.js';

const TIMEOUT_MS = 30_000;

class ECGRechazado extends Error {
  constructor(status, error) {
    super(error?.mensaje || 'El ECG no se pudo analizar.');
    this.status = status;
    this.codigo = error?.codigo || 'desconocido';
    this.detalle = error?.detalle;
  }
}

async function analizar({ buffer, nombreArchivo, frecuencia, derivaciones }) {
  const form = new FormData();
  form.append('archivo', new Blob([buffer]), nombreArchivo || 'ecg');
  if (frecuencia) form.append('frecuencia', String(frecuencia));
  if (derivaciones) form.append('derivaciones', derivaciones);

  let respuesta;
  try {
    respuesta = await fetch(`${env.inferencia.url}/analizar`, {
      method: 'POST',
      headers: { 'X-DECA-Token': env.inferencia.token },
      body: form,
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
  } catch (err) {
    // El servicio no contestó: cayó, está arrancando, o no hay red. Es 503, no 500:
    // el pedido del médico estaba bien y reintentar puede funcionar.
    throw new ECGRechazado(503, {
      codigo: 'inferencia_no_disponible',
      mensaje: 'El servicio de análisis no está disponible. Intentá de nuevo en un momento.',
      detalle: err.message,
    });
  }

  const cuerpo = await respuesta.json().catch(() => null);
  if (!respuesta.ok || !cuerpo?.ok) {
    throw new ECGRechazado(respuesta.status, cuerpo?.error);
  }
  return cuerpo.analisis;
}

export { ECGRechazado };
export default { analizar, ECGRechazado };
```

`FormData`, `Blob`, `fetch` y `AbortSignal.timeout` son nativos en Node 18+. No hace falta
`axios` ni `node-fetch`.

---

## 4. `analisisService.js`

```js
const SELECT_FORMATEADO =
  `id, paciente_id, porcentaje, banda, score, modelo_sha, ${enHorarioArgentino('fecha_hora_entrega')}`;

async function crear({ pacienteId, porcentaje, banda, score, modeloSha }) {
  const result = await pool.query(
    `INSERT INTO analisis (paciente_id, porcentaje, banda, score, modelo_sha)
     VALUES ($1, $2, $3, $4, $5)
     RETURNING ${SELECT_FORMATEADO}`,
    [pacienteId, porcentaje, banda, score, modeloSha]
  );
  return result.rows[0];
}
```

`listarPorPaciente` no se toca: ya usa `SELECT_FORMATEADO`, así que hereda las columnas
nuevas sola.

---

## 5. `analisisController.js`

Reemplazar `realizar` entera. Lo que se conserva: `estaAsignado`, `notificarNuevoAnalisis`,
el `23503` → 404. Lo que se va: `porcentajeValido`, que queda sin uso — **borrala**, el
porcentaje ya no lo manda nadie.

```js
import inferenciaService, { ECGRechazado } from '../services/inferenciaService.js';

async function realizar(req, res) {
  try {
    const { pacienteId, frecuencia, derivaciones } = req.body;

    if (!pacienteId) {
      return res.status(400).json({ ok: false, error: 'Falta el paciente.' });
    }
    if (!req.file) {
      return res.status(400).json({ ok: false, error: 'Falta el archivo del ECG.' });
    }
    if (!(await estaAsignado(pacienteId, req.usuario.id))) {
      return res.status(403).json({ ok: false, error: 'Ese paciente no está asignado a tu cuenta.' });
    }

    const resultado = await inferenciaService.analizar({
      buffer: req.file.buffer,
      nombreArchivo: req.file.originalname,
      frecuencia,
      derivaciones,
    });

    const analisis = await analisisService.crear({
      pacienteId,
      porcentaje: resultado.percentil,
      banda: resultado.banda,
      score: resultado.score,
      modeloSha: resultado.modelo.sha256,
    });

    await notificarNuevoAnalisis(pacienteId);

    // `interpretacion` y `calidad` no se guardan: son para mostrar en el momento.
    // En los GET posteriores sólo vuelve lo que está en la base (ver nota abajo).
    res.status(201).json({
      ok: true,
      analisis,
      interpretacion: resultado.interpretacion,
      calidad: resultado.calidad,
    });
  } catch (err) {
    if (err instanceof ECGRechazado) {
      return res.status(err.status).json({ ok: false, error: err.message, codigo: err.codigo });
    }
    if (err.code === '23503') {
      return res.status(404).json({ ok: false, error: 'El paciente no existe.' });
    }
    res.status(500).json({ ok: false, error: err.message });
  }
}
```

### El texto de la banda en los GET

`interpretacion.texto` sale del servicio y no se guarda, así que en `GET /analisis` no
está. Lo más simple es un mapa constante en el backend:

```js
const TEXTO_BANDA = {
  alta: 'Prioridad alta. De cada 100 personas priorizadas así, cerca de 30 resultaron '
      + 'positivas en la validación del modelo.',
  media: 'Prioridad intermedia. Apenas por encima de la prevalencia general.',
  baja: 'Prioridad baja. Casi todos los que caen acá son negativos, pero no descarta Chagas.',
};
```

Si el modelo se recalibra, estos textos hay que actualizarlos: la fuente de verdad es
`GET /contrato` del servicio de inferencia, que los devuelve junto con las métricas.

---

## 6. `analisisRoutes.js`

```bash
npm install multer
```

```js
import multer from 'multer';

// memoryStorage porque el archivo se reenvía y se descarta: nunca toca el disco.
const subida = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 8 * 1024 * 1024 },
});

router.post('/', permitirRoles('medico'), subida.single('archivo'), analisisController.realizar);
```

Los dos `GET` quedan igual.

**Ojo con los errores de multer**: si el archivo pasa el límite, multer tira un
`MulterError` que sin manejar se va como 500. Conviene un middleware de error en
`server.js`:

```js
app.use((err, req, res, next) => {
  if (err?.name === 'MulterError') {
    return res.status(413).json({ ok: false, error: 'El archivo es demasiado grande.' });
  }
  next(err);
});
```

---

## 7. Tests que se rompen

Los tres archivos de `analisis`. Van con nombre para que no haya que cazarlos.

### `src/controllers/analisisController.test.js`

| Test | Qué hacer |
|---|---|
| `devuelve 400 si falta el paciente o el porcentaje` | Renombrar a "si falta el paciente o el archivo"; el `req` pasa a `{ body: { pacienteId: 1 }, file: undefined }` |
| `devuelve 400 si el porcentaje no es un número` | **Borrar** |
| `devuelve 400 si el porcentaje está fuera de rango` | **Borrar** |
| `devuelve 403 si el paciente no está asignado` | Agregar `file` al `req` |
| `crea el análisis, notifica al paciente y devuelve 201` | Mockear `inferenciaService.analizar`; el `expect` de `crear` pasa a los 5 campos |
| `no falla la creación aunque la notificación no se pueda crear` | Ídem |
| `devuelve 404 si el paciente no existe` | Ídem |
| `devuelve 500 ante un error inesperado` | Ídem |
| — | **Nuevo**: 422 cuando el servicio rechaza el ECG |
| — | **Nuevo**: 503 cuando el servicio no responde |

El `req` de los tests de `realizar` pasa a tener esta forma:

```js
const req = {
  usuario: { id: 2, rol: 'medico' },
  body: { pacienteId: 1, frecuencia: '500' },
  file: { buffer: Buffer.from('I,II,III\n0,0,0\n'), originalname: 'ecg.csv' },
};
```

Y el mock del servicio de inferencia:

```js
vi.spyOn(inferenciaService, 'analizar').mockResolvedValue({
  percentil: 98.7,
  banda: 'alta',
  score: 0.961234,
  interpretacion: { texto: '…', ppv: 0.2953 },
  calidad: { duracion_recibida_s: 10 },
  modelo: { sha256: 'b0e2ecfc838e169c' },
});
```

Para el caso 422:

```js
vi.spyOn(inferenciaService, 'analizar').mockRejectedValue(
  new ECGRechazado(422, { codigo: 'senal_corta', mensaje: 'El ECG tiene menos de 7.0 s…' })
);
```

Los bloques `listarPropios` y `listarDePaciente` **no cambian**, salvo agregarle
`banda`, `score` y `modelo_sha` al objeto `analisis` de arriba del archivo.

### `src/services/analisisService.test.js`

- `inserta el análisis y devuelve la fila creada` → `expect(params).toEqual([1, 42.5])`
  pasa a los 5 parámetros nuevos.
- El fixture `analisis` de arriba del archivo, con las columnas nuevas.

### `src/routes/analisisRoutes.test.js`

Todos los `POST /analisis` usan `.send({ pacienteId, porcentaje })` y pasan a multipart.
Supertest lo soporta nativo:

```js
const res = await request(app)
  .post('/analisis')
  .set('Authorization', `Bearer ${token('medico', 2)}`)
  .field('pacienteId', '1')
  .field('frecuencia', '500')
  .attach('archivo', Buffer.from('I,II,III\n0,0,0\n'), 'ecg.csv');
```

El test `devuelve 400 si faltan datos` ahora significa "falta el archivo". Los `GET` no se
tocan.

---

## 8. Qué tiene que mandar el frontend

`multipart/form-data` a `POST /analisis`:

| campo | obligatorio | |
|---|---|---|
| `pacienteId` | sí | |
| `archivo` | sí | `.csv`, `.json` o `.zip` (WFDB) |
| `frecuencia` | **sí para CSV** | Hz. El JSON y el WFDB la traen adentro |
| `derivaciones` | sólo si el CSV no tiene encabezado | nombres separados por coma |

**La frecuencia de muestreo para un CSV es obligatoria y no se puede adivinar.** Si el
formulario no la pide, todos los CSV van a ser rechazados con
`frecuencia_no_declarada`. Lo más práctico es un campo con 500 Hz por defecto, que es lo
más común.

El CSV tiene que tener **encabezado con los nombres de las derivaciones** (`I`, `II`,
`aVR`, `V1`…). Alcanza con 8 (I, II y las 6 precordiales): el servicio reconstruye las
otras cuatro solo. Las unidades no importan (mV, µV, cuentas de ADC: da igual).

**Lo que no se puede procesar: una foto o un PDF del ECG impreso.** El modelo necesita
señal digital. Si el equipo del hospital sólo imprime en papel, eso es un problema a
resolver antes y no se arregla del lado del software.

---

## 9. Cosas que te van a morder

**El límite de body de Vercel son ~4,5 MB**, no los 32 MB que acepta el servicio de
inferencia. Un ECG de 10 s a 500 Hz en CSV son ~600 KB, así que entra bien, pero un export
verboso o una tira larga puede no entrar y Vercel lo corta antes de que tu código vea nada.
Si aparece, la salida es subir el archivo a storage y mandarle la URL al servicio, o sacar
ese endpoint de Vercel.

**El timeout de función de Vercel** son 10 s en Hobby. El análisis tarda ~500 ms, pero si el
servicio de inferencia está en un tier gratuito que se duerme, el primer request después de
un rato puede tardar 30 s y te va a comer el timeout. Si eso pasa, hay que pagar un tier que
no duerma o pingear el `/salud` periódicamente.

**Un 422 no es una caída.** Significa "el pedido estaba bien, el ECG no sirve": archivo
corto, corrupto, sin nombres de derivación. Mostrale `error.mensaje` al médico para que
suba otro archivo. Si lo tratás como error del servidor, la UX va a ser pésima justo en el
caso más común.

**No muestres `score` como porcentaje, nunca.** La salida cruda de la red parece una
probabilidad y no lo es: el umbral de la banda alta está en `0.93`, y ahí el valor
predictivo real medido es **29,5 %**. Mostrar "93 %" le erra por un factor de tres, y le
erra hacia arriba, que es el lado peligroso. El número que se muestra es `porcentaje` (el
percentil) y siempre acompañado de la banda.

**El resultado no es un diagnóstico.** La interfaz tiene que decirlo. El servicio devuelve
la advertencia ya escrita en `interpretacion.advertencia`, para no tener que inventarla:

> Resultado de tamizaje: indica prioridad para la prueba serológica de Chagas, no
> diagnostica cardiopatía chagásica ni reemplaza la evaluación clínica.

---

## 10. Cómo probarlo

**Sin el servicio andando** — mockeás `inferenciaService.analizar` en los tests, como en la
sección 7. No hace falta Python para que `npm test` pase.

**Con el servicio de verdad**, si tenés el repo de IA:

```bash
pip install -r requirements.txt -r requirements-api.txt
DECA_API_SIN_AUTH=1 python src/servidor.py --puerto 8000
```

Con `DECA_API_SIN_AUTH=1` no pide token, sólo para desarrollo local. Después, desde el
backend, `DECA_INFERENCIA_URL=http://localhost:8000`.

Para ver si está vivo y qué modelo tiene cargado:

```bash
curl http://localhost:8000/salud
```

Y para chequear el contrato completo (formatos, códigos de error, métricas de cada banda):

```bash
curl -H "X-DECA-Token: $DECA_API_TOKEN" http://localhost:8000/contrato
```

---

## Lo que falta definir, y no depende de vos

- **Dónde se hostea el servicio de inferencia.** Vercel no puede alcanzar una máquina detrás
  de un NAT, así que necesita una URL pública. Con CPU alcanza: ~500 ms por análisis, no
  hace falta GPU.
- **En qué formato exporta el equipo del hospital.** Por ahora el servicio lee CSV, JSON y
  WFDB. Agregar otro formato (SCP-ECG, DICOM, XML de GE o Philips) es media hora del lado
  de IA, pero hay que saber cuál es.

Cualquier duda sobre lo que devuelve el modelo está en [`API.md`](API.md), y si algo no está
ahí, preguntá.
