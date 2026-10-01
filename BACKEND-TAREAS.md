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

## Revisión del 01/10 — cómo quedó `d8225c5`

Revisé los 5 commits posteriores a la revisión anterior. **Los dos pendientes del 28/09 siguen
igual** (detalle abajo, en esa revisión), y apareció un problema nuevo con el simulador.

### ❌ Sigue igual: la columna `banda` rechaza `no_alta`

`schema.sql` todavía tiene `VARCHAR(6) CHECK (banda IN ('alta', 'media', 'baja'))`. Ahora
importa más que antes: **el servicio de inferencia ya tiene imagen Docker** (`Dockerfile` en
este repo) y se va a publicar, y en cuanto `DECA_INFERENCIA_URL` apunte a algo, ~98,5 % de los
`POST /analisis` van a dar 500. El arreglo está en la [sección 2](#2-migración).

### ⚠️ Sigue faltando: el texto de la banda en los `GET`

Sin cambios respecto del 28/09: no está `TEXTO_BANDA` ([sección 5](#el-texto-de-la-banda-en-los-get)).

### ❌ Nuevo: sin `DECA_INFERENCIA_URL`, el backend inventa resultados y se pueden mandar al paciente

`inferenciaService.analizar` ahora devuelve un resultado **al azar** (`simular()`) cuando no hay
URL configurada, en vez de un 503. Para probar la pantalla sirve. El problema es lo que pasa
después:

- `analisisController.realizar` lo guarda en `analisis` igual que uno real: porcentaje y banda
  al azar.
- El texto "Resultado simulado" viaja **sólo en la respuesta del `POST`**. No se guarda en la
  base, así que los `GET` muestran un porcentaje y una banda sin ninguna marca de que son
  inventados. Lo único que los delata es `modelo_sha = 'simulado'`, y nada lo muestra.
- El médico puede **aprobarlo y enviarlo**, y el paciente recibe un resultado de Chagas
  sorteado.

Hoy `DECA_INFERENCIA_URL` no está configurada en Vercel, así que **esto le pasa a todo
análisis que se cree en el despliegue actual**.

**Arreglo propuesto**: que el simulador sólo corra si se lo pide explícitamente (por ejemplo
`DECA_INFERENCIA_SIMULADA=1` en `.env` local, nunca en Vercel) y que, sin URL y sin esa
variable, vuelva el 503 `inferencia_no_disponible` de antes. Si se quiere conservar en algún
entorno compartido, `aprobar` y `enviar` deberían rechazar los análisis con
`modelo_sha = 'simulado'`. Y conviene borrar de la base los análisis simulados que ya se hayan
creado (`DELETE FROM analisis WHERE modelo_sha = 'simulado'`), sobre todo los que tengan
`enviado = TRUE`.

Detalle menor: el simulador reparte tres bandas (`alta` desde el percentil 90) y el modelo
real devuelve dos, con `alta` desde el percentil ~98,5. La pantalla probada con el simulador
muestra `alta` en el 10 % de los casos, siete veces más que con el modelo real, y nunca
muestra `no_alta`, que es lo que va a ver casi todo el mundo. Si el simulador se queda,
`bandaDe` debería devolver `percentil >= 98.5 ? 'alta' : 'no_alta'`, y el test de
`inferenciaService` debería cubrir `no_alta`.

### ✅ Está bien

El flujo de aprobación (`aprobado`/`enviado`, `aprobar` → `enviar`, `rechazar` sólo si no está
aprobado) encaja con lo que pide la validación clínica: el médico ve el resultado antes que el
paciente. El guardado del ECG en Vercel Blob con acceso privado por defecto también está bien.

---

## Revisión del 28/09 — cómo quedó `b847950`

Revisé DECA-Back en `cda0f2f` y corrí `npm test` (235/235 pasan). Casi todo está bien, pero
hay **un bug que rompe el caso más común**.

### ❌ Hay que arreglarlo: la columna `banda` rechaza `no_alta`

Implementaron una versión anterior de este documento, que tenía tres bandas
(`alta`/`media`/`baja`). El 23/09 el modelo pasó a **dos**: `alta` y `no_alta`. En
`schema.sql` quedó:

```sql
ALTER TABLE analisis ADD COLUMN IF NOT EXISTS banda VARCHAR(6)
  CHECK (banda IN ('alta', 'media', 'baja'));
```

`'no_alta'` tiene 7 caracteres y no está en el `CHECK`, así que Postgres rechaza el INSERT y
el controller devuelve 500. **Le pasa a ~98,6 % de los ECG**, porque casi todos caen en
`no_alta`. El ECG se analiza bien, pero el resultado no se guarda y el médico ve un error.
Los tests no lo detectan porque todos los mocks usan `banda: 'alta'`.

**Arreglo**: el SQL corregido está en la [sección 2](#2-migración), y el test que falta en la
[sección 7](#7-tests-que-se-rompen). Cambiar sólo el `ADD COLUMN` no alcanza: si ya corrieron
`npm run migrate`, el `IF NOT EXISTS` deja la columna vieja como está.

### ⚠️ Falta: el texto de la banda en los `GET`

No está el mapa `TEXTO_BANDA` ([sección 5](#el-texto-de-la-banda-en-los-get)). Sin él,
`GET /analisis` devuelve `banda: 'no_alta'` sin la aclaración de que **eso no descarta
Chagas**, que es lo más importante que tiene que ver el médico. Ojo: la versión vieja de
este documento tenía el mapa con tres entradas; usen el de ahora, que tiene dos.

### ✅ Está bien, no tocar

`.env.example` y `env.js`, `inferenciaService.js`, `analisisService.crear` con los 5 campos,
el controller (400 / 403 / 422 / 503 / 404 bien separados, `estaAsignado` conservado),
`multer` con `memoryStorage` y el handler de 413 en `server.js`, los tests de 422 y 503.

### ⏳ Pendiente, no depende de ustedes

El servicio de inferencia todavía no tiene URL pública. Hasta que la tenga,
`DECA_INFERENCIA_URL` en Vercel no apunta a nada y todo `POST /analisis` devuelve 503. Para
probar localmente, ver la [sección 10](#10-cómo-probarlo).

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
ALTER TABLE analisis ADD COLUMN IF NOT EXISTS banda VARCHAR(7)
  CHECK (banda IN ('alta', 'no_alta'));
ALTER TABLE analisis ADD COLUMN IF NOT EXISTS score NUMERIC(8, 6);
ALTER TABLE analisis ADD COLUMN IF NOT EXISTS modelo_sha VARCHAR(16);
```

Después `npm run migrate`. Es idempotente como el resto del archivo.

**Si ya migraron con la versión de tres bandas** (`VARCHAR(6)`, `'alta','media','baja'`), el
`ADD COLUMN IF NOT EXISTS` de arriba no hace nada porque la columna ya existe. Agreguen esto
justo debajo, también en `schema.sql`, para que corrija la base ya creada. Es idempotente:

```sql
ALTER TABLE analisis DROP CONSTRAINT IF EXISTS analisis_banda_check;
ALTER TABLE analisis ALTER COLUMN banda TYPE VARCHAR(7);
ALTER TABLE analisis ADD CONSTRAINT analisis_banda_check
  CHECK (banda IN ('alta', 'no_alta'));
```

`analisis_banda_check` es el nombre que Postgres le pone solo a ese `CHECK`. Si el
`ADD CONSTRAINT` falla porque ya existe con otro nombre, búsquenlo con
`\d analisis` en `psql`. Si hay filas de prueba con `media` o `baja`, el `ADD CONSTRAINT`
falla: bórrenlas antes (son de prueba, el modelo nunca devolvió esos valores).

**Qué va en cada columna:**

- **`porcentaje`** (la que ya existe) → el **percentil de riesgo**. Entra tal cual en
  `NUMERIC(5,2)` y el `CHECK (0..100)` sigue valiendo.
- **`banda`** → `alta` / `no_alta`. **Es lo más importante que devuelve el modelo**: `alta`
  es la que manda a serología. Si sólo guardás el percentil, perdés lo único que tiene un
  valor predictivo medido. Y en pantalla la banda va antes que el percentil: un `no_alta`
  puede tener percentil 97, y leído solo eso suena a urgencia.
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
  no_alta: 'Sin prioridad por ECG. Esto NO descarta Chagas: cerca de 4 de cada 5 casos '
      + 'reales caen acá. Con antecedentes epidemiológicos (zona endémica, madre con '
      + 'Chagas, transfusiones) corresponde ofrecer la serología igual.',
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
| — | **Nuevo**: 201 con `banda: 'no_alta'` (el caso más común; ver abajo) |

**Que al menos un test use `no_alta`.** Si todos los mocks devuelven `alta`, nunca se
prueba el caso que le pasa a casi todos los pacientes. El test del controller no alcanza
para detectar el problema de la columna, porque mockea la base. Lo que sí lo detecta es
probar contra una base real: `POST /analisis` con el servicio de inferencia corriendo
localmente (sección 10) y un ECG cualquiera, que casi seguro cae en `no_alta`.

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
  hace falta GPU. Se va a correr en un servidor de la institución (`DESPLIEGUE.md` en DECA-AI);
  falta definir con sistemas cómo se publica la URL.
- **En qué formato exporta el equipo del hospital.** Por ahora el servicio lee CSV, JSON y
  WFDB. Agregar otro formato (SCP-ECG, DICOM, XML de GE o Philips) es media hora del lado
  de IA, pero hay que saber cuál es.

Cualquier duda sobre lo que devuelve el modelo está en [`API.md`](API.md), y si algo no está
ahí, preguntá.
