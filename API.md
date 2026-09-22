# Servicio de inferencia de DECA — contrato de integración

Para el equipo de [DECA-Back](https://github.com/bf0880-boop/DECA-Back). Describe cómo
pedirle al modelo que analice un ECG y qué hacer con lo que devuelve. **Todavía no hay
ningún cambio hecho en el backend**: este documento dice qué habría que agregar.

---

## 1. Qué hace el modelo, y qué no

El modelo lee un ECG de 12 derivaciones y estima la **prioridad** de esa persona para
hacerse la prueba serológica de Chagas, que es gratuita en Argentina. **No diagnostica
cardiopatía chagásica y no reemplaza al médico.**

Lo que está medido, sobre 34.772 pacientes que el modelo nunca vio (test, 14/09/2026):

| Banda | % de la población | Qué significa |
|---|---|---|
| **alta** | 1,4 % | De cada 100 personas priorizadas así, **~30 resultaron positivas** |
| **media** | 62,2 % | VPP 2,9 %: apenas por encima de la prevalencia general (1,9 %) |
| **baja** | 37,8 % | VPN 99,76 %, pero **el 4,7 % de los casos reales cae acá** |

### El número que se muestra es un percentil, no una probabilidad

La red devuelve internamente un score entre 0 y 1, y **ese score no es la probabilidad de
tener Chagas.** El umbral de la banda alta está en 0,930 y ahí el valor predictivo medido
es 29,5 %. Mostrar "93 %" sería errarle por un factor de tres, en el sentido que más daño
hace.

Por eso el campo principal de la respuesta es `percentil`: dónde cae este ECG dentro de la
distribución de scores de una población de referencia (34.780 pacientes de validación).
`"percentil": 98.7` se lee **"este ECG puntúa más alto que el 98,7 % de los ECG de
referencia"**. Eso es literalmente cierto y es lo que sirve para decidir a quién se manda
primero a serología.

**Texto sugerido para la interfaz** — el servicio ya devuelve uno en
`analisis.interpretacion.texto`:

> Prioridad alta (percentil 98,7). De cada 100 personas priorizadas así, cerca de 30
> resultaron positivas en la validación del modelo. Se sugiere solicitar serología.
> *Este resultado no diagnostica Chagas.*

---

## 2. Por qué es un servicio aparte y no una función del backend

DECA-Back corre en Vercel serverless: no puede ejecutar PyTorch ni lanzar un proceso
Python. Las opciones eran:

| Opción | Por qué no |
|---|---|
| Exportar a ONNX y correrlo en Node | El modelo pesa 78 MB y el runtime nativo más, contra los 250 MB de bundle de Vercel |
| Reescribir el preprocesamiento en JS | **El motivo real del descarte.** El resampleo y la normalización tienen que ser idénticos a los del entrenamiento; si no, las métricas de arriba dejan de aplicar al número que devuelve la API, y no hay forma de notarlo mirando la salida |
| `spawn('python')` desde Node | Imposible en serverless |

Entonces: **el backend habla HTTP con un servicio Python** que usa exactamente el mismo
código de preprocesamiento que procesó el corpus de entrenamiento.

```
navegador → DECA-Back (Vercel, Node)  →  servicio de inferencia (Python)
             ├ JWT, rol médico             ├ sin usuarios, sin sesiones
             ├ chequeo paciente asignado   ├ token compartido
             └ guarda en Postgres          └ sin estado: no guarda la señal
```

**El archivo sube por el backend, no directo al servicio.** El backend ya tiene el JWT y
el chequeo `estaAsignado`; el servicio de inferencia nunca se expone a internet. El
servicio **no guarda ni loguea la señal**: la procesa en RAM y la descarta. Guardar el ECG,
si se quiere guardar, es decisión del backend, que es quien tiene el consentimiento.

---

## 3. Endpoints

Base URL configurable; token en el header `X-DECA-Token` en todos salvo `/salud`.

### `POST /analizar` — el camino principal

`multipart/form-data`:

| campo | tipo | obligatorio | nota |
|---|---|---|---|
| `archivo` | file | sí | `.csv`, `.json` o `.zip` (WFDB) |
| `frecuencia` | number | sólo para CSV | Hz. El JSON y el WFDB la traen adentro |
| `derivaciones` | string | sólo si el archivo no las nombra | separadas por coma |
| `formato` | string | no | `csv` \| `json` \| `wfdb`; por defecto se detecta |

```bash
curl -X POST https://inferencia.deca/analizar \
  -H "X-DECA-Token: $DECA_API_TOKEN" \
  -F "archivo=@ecg_paciente.csv" \
  -F "frecuencia=500"
```

Desde Node (18+, sin dependencias):

```js
const form = new FormData();
form.append('archivo', new Blob([buffer]), nombreArchivo);
form.append('frecuencia', String(frecuencia));

const r = await fetch(`${process.env.DECA_INFERENCIA_URL}/analizar`, {
  method: 'POST',
  headers: { 'X-DECA-Token': process.env.DECA_API_TOKEN },
  body: form,
});
const cuerpo = await r.json();
if (!cuerpo.ok) {
  // 422 = el ECG no sirve. Es un resultado esperable, no una caída del servicio:
  // mostrale cuerpo.error.mensaje al médico para que suba otro archivo.
  return res.status(r.status).json(cuerpo);
}
const { percentil, banda, score, interpretacion } = cuerpo.analisis;
```

### `POST /analizar/json` — si el frontend ya tiene la matriz

Cuerpo `application/json`, sin multipart:

```json
{ "frecuencia": 500,
  "derivaciones": ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"],
  "senal": [[0.01, -0.02, ...], ...] }
```

`senal` puede venir como `(muestras, derivaciones)` o `(derivaciones, muestras)`: se
orienta sola comparando contra la cantidad de nombres.

### `GET /contrato` — la fuente de verdad, servida por el servicio

Devuelve formatos aceptados, nombres canónicos de derivación, el catálogo completo de
códigos de error y las métricas de cada banda. **Conviene leerlo desde el backend en vez de
copiar estas tablas**: si el modelo se recalibra, cambia solo.

### `GET /salud` — sin token

Para el health check del orquestador. Devuelve el device, el hash del checkpoint cargado y
la fecha de calibración.

---

## 4. Respuesta

```jsonc
{
  "ok": true,
  "analisis": {
    "percentil": 98.7,          // ← esto es lo que se muestra y se guarda
    "banda": "alta",            // "alta" | "media" | "baja"
    "score": 0.961234,          // crudo, para trazabilidad. NO mostrarlo como porcentaje
    "interpretacion": {
      "texto": "Prioridad alta. …",
      "ppv": 0.2953,            // medido en test
      "sensibilidad": 0.2177,
      "poblacion": 0.0141,
      "advertencia": "Resultado de tamizaje: …no diagnostica…"
    },
    "referencia": { "poblacion": "validacion / arena A (CODE-15%)…", "n": 34780 },
    "patrones": {               // hallazgos ECG, como explicación del score
      "brd":   { "score": 0.94, "auprc_test": 0.8024, "usable": true },
      "hbai":  { "score": 0.11, "auprc_test": 0.4029, "usable": true },
      "extra": { "score": 0.03, "auprc_test": 0.6411, "usable": true },
      "zona":  { "score": 0.20, "auprc_test": 0.0458, "usable": false }
    },
    "calidad": {                // qué llegó y qué se analizó
      "duracion_recibida_s": 10.0,
      "ventana_analizada_s": 7.0,
      "derivaciones_reconstruidas": [],
      "derivaciones_planas": [],
      "avisos": []
    },
    "modelo": { "checkpoint": "patrones-lr8/mejor.pt", "sha256": "b0e2ecfc838e169c" }
  }
}
```

**`patrones` viene con su calidad al lado a propósito.** `zona` mide AUPRC 0,046, o sea
ruido, y por eso trae `"usable": false` — **no mostrarlo**. `brd` (bloqueo de rama derecha)
mide 0,80 y es el hallazgo sólido. Mostrar un score de patrón sin su calidad invita a
leerlo como un hallazgo clínico, que es justo lo que no es.

**`calidad.ventana_analizada_s` es siempre 7,0.** Si llega una tira de 30 s, se analizan
los 7 s centrados. Vale mostrárselo al médico.

---

## 5. Qué mandar: formatos y derivaciones

### La regla que no se puede romper: las derivaciones van por nombre

El servicio **nunca** carga derivaciones por posición y rechaza cualquier archivo donde no
pueda saber qué columna es cuál. No es exceso de celo: hay datasets reales cuyo orden es
`DI, DII, DIII, aVL, aVF, aVR, V1…V6` — con aVR y aVL cambiadas de lugar. Leer eso por
posición no falla: produce un ECG con dos derivaciones intercambiadas, que el modelo puntúa
mal y nadie detecta nunca.

Orden canónico: `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`.

Se reconocen los alias habituales sin que haya que normalizar nada del lado del backend:
`DI`/`D1`/`L1` → `I`, `AVR`/`VR` → `aVR`, `C1..C6` → `V1..V6`, `Lead II`, `MDC_ECG_LEAD_I`,
mayúsculas o minúsculas, con o sin espacios.

### Con 8 derivaciones alcanza

De las 12 sólo 8 son independientes. Si llegan **I, II y las 6 precordiales**, el servicio
reconstruye `III`, `aVR`, `aVL` y `aVF` con las identidades de Einthoven y Goldberger —
exactas, no aproximadas — y lo informa en `calidad.derivaciones_reconstruidas`.

### Las unidades no importan

mV, µV o cuentas de ADC dan el mismo resultado: cada derivación se normaliza contra sí
misma. **No hace falta declarar unidad ni ganancia.** Una columna extra de tiempo o de
índice (`tiempo`, `sample`, `ms`…) se descarta sola.

### Formatos

| formato | cómo | frecuencia |
|---|---|---|
| **CSV** | encabezado con nombres de derivación, una fila por muestra. Acepta `,` `;` tab, y coma decimal | **hay que declararla** |
| **JSON** | ver arriba | la trae adentro |
| **WFDB** | `.hea` + `.dat` **juntos en un `.zip`** | la trae el `.hea` |

Agregar un formato (SCP-ECG, DICOM, XML de GE/Philips) es escribir una función y sumarla al
registro `LECTORES` en `src/lectura_ecg.py`; nada más cambia. Cuando se sepa qué exporta el
equipo del hospital, se agrega.

**Lo que no se puede leer: una foto o un PDF del ECG impreso.** El modelo necesita señal
digital. Digitalizar papel es un problema de investigación aparte y no se resuelve acá.
Si el hospital sólo tiene papel, el proyecto tiene un problema de origen que hay que
resolver antes, no un problema de formato.

---

## 6. Errores

Todos vienen con el mismo sobre `{ ok, error }` que ya usa el backend:

```json
{ "ok": false, "error": { "codigo": "senal_corta",
                          "mensaje": "El ECG tiene menos de 7.0 s de señal util…",
                          "detalle": "duracion recibida 2.00s a 400 Hz" } }
```

| HTTP | Significado | Qué hacer |
|---|---|---|
| **422** | El pedido está bien, **el ECG no sirve** | Mostrarle `error.mensaje` al médico y pedirle otro archivo. **No es una caída del servicio** |
| 401 | Token ausente o incorrecto | Revisar `DECA_API_TOKEN` |
| 413 | Archivo > 32 MB | — |
| 503 | El modelo todavía no cargó | Reintentar |

`error.codigo` es estable y está pensado para que el backend arme su propio mensaje sin
parsear texto. Los 16 códigos salen de `GET /contrato`. Los que más van a aparecer:

- `senal_corta` — menos de 7,0 s de señal útil. No se rellena con ceros: eso cambiaría la
  señal que ve la red.
- `frecuencia_no_declarada` — CSV sin `frecuencia`. No se puede deducir de la señal.
- `derivaciones_no_declaradas` / `derivacion_desconocida` — no se sabe qué columna es cuál.
- `derivaciones_faltantes` — faltan precordiales, que no se pueden reconstruir.
- `senal_corrupta` — hay NaN o infinitos en el archivo.
- `senal_plana` — 7 o más derivaciones sin señal (electrodos desconectados).

---

## 7. Lo que habría que agregar en DECA-Back

Hoy `POST /analisis` recibe `{ pacienteId, porcentaje }` y **el médico tipea el porcentaje a
mano**: no hay upload de ECG en ningún lado, ni llamada a ningún modelo. Para conectar el
modelo hacen falta tres cosas:

**a) Un endpoint que reciba el archivo.** `POST /analisis` pasa a aceptar
`multipart/form-data` con el ECG, valida `estaAsignado` como ya lo hace, reenvía al servicio
de inferencia y guarda lo que vuelve. El `porcentaje` deja de venir del cuerpo del pedido.

**b) Columnas nuevas en `analisis`.** `porcentaje` guarda el **percentil** — entra tal cual
en `NUMERIC(5,2)` y el `CHECK (0..100)` sigue valiendo. Pero si sólo se guarda eso se pierde
lo más importante, que es la banda:

```sql
ALTER TABLE analisis ADD COLUMN IF NOT EXISTS banda VARCHAR(6)
  CHECK (banda IN ('alta','media','baja'));
ALTER TABLE analisis ADD COLUMN IF NOT EXISTS score NUMERIC(8,6);
ALTER TABLE analisis ADD COLUMN IF NOT EXISTS modelo_sha VARCHAR(16);
```

`modelo_sha` importa para trazabilidad: si algún día se cambia el modelo, hay que poder
saber qué versión produjo cada análisis viejo.

**c) Dos variables de entorno:** `DECA_INFERENCIA_URL` y `DECA_API_TOKEN`.

---

## 8. Limitaciones que conviene tener a la vista

Están medidas, no son advertencias de forma.

- **La banda alta es, operativamente, un detector de bloqueo de rama derecha.** Encuentra al
  73,3 % de los positivos que tienen BRD anotado y al 10,4 % de los que no. Un ECG sin BRD
  rara vez va a entrar en banda alta, aunque la persona esté infectada.
- **Un resultado bajo no descarta Chagas.** El 4,7 % de los casos reales cae en banda baja.
- **A las mujeres las encuentra menos** (18,5 % contra 27,3 % en hombres), porque tienen
  menor prevalencia de BRD, no porque el modelo las trate distinto a igual ECG.
- **El modelo no sabe reconocer si lo que recibió es un ECG.** Se le puede dar ruido blanco
  y devuelve un score alto con total confianza (medido: 0,615, percentil 92). Los controles
  de calidad atajan señal corta, corrupta o plana, pero **no** "esto no es un
  electrocardiograma". El archivo que llega tiene que venir de un equipo real.
- **La población de referencia del percentil es brasileña** (CODE-15%, Minas Gerais). Es la
  más parecida que existe con etiqueta, pero no es argentina.

---

## 9. Levantar el servicio

```bash
pip install -r requirements.txt -r requirements-api.txt
export DECA_API_TOKEN=<secreto compartido con el backend>
python src/servidor.py --host 0.0.0.0 --puerto 8000
```

No necesita el SSD ni los datasets: el checkpoint (78 MB) y la calibración viajan en el
repo, en `models/patrones-lr8/`. Anda en CPU — **471 ms por análisis**, medido — así que no
hace falta GPU para servir. Variables: `DECA_DEVICE` (`cpu`/`cuda`), `DECA_CHECKPOINT`,
`DECA_CALIBRACION`, `DECA_API_SIN_AUTH=1` (sólo desarrollo local).

**Dónde corre es un tema abierto.** Vercel no puede alcanzar una laptop detrás de un NAT,
así que el servicio necesita una URL pública: un host con CPU (Render, Railway, Fly) alcanza
de sobra. Para desarrollo, un túnel contra la máquina local.

Si se cambia el checkpoint hay que recalibrar — el servicio se niega a arrancar con una
calibración que no corresponde al modelo, porque los percentiles no significarían nada:

```bash
python src/calibrar_servicio.py --checkpoint <ruta>   # necesita el SSD
```
