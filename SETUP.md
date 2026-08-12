# Puesta en marcha desde cero

Cómo dejar una PC nueva corriendo el módulo de IA de DECA. Cubre **Windows** y **Linux** (la máquina de entrenamiento con la RTX 4090 tiene Ubuntu).

Para *qué* hace cada cosa y *por qué* está diseñada así, ver [ROADMAP.md](ROADMAP.md) (contexto del problema) y [FASES.md](FASES.md) (bitácora de decisiones). Este archivo es solo el instructivo.

---

## Primero: ¿cuál de los dos escenarios es?

Casi siempre es el **A**.

| | **A — Solo entrenar** | **B — Rehacer el dataset desde cero** |
|---|---|---|
| Cuándo | El SSD ya tiene `fase2_preprocessed.hdf5` | Se perdió el SSD, o se cambió el preprocesamiento |
| Qué corrés | `train.py` | Fases 0-2 completas, después `train.py` |
| Cuánto tarda | minutos de setup | ~2 h de cómputo + horas de descarga |
| Espacio | ~46 GB (ya ocupados) | ~120 GB en el SSD |

El escenario B está al final. Si el SSD está sano, salteátelo.

---

## Requisitos

- **Python 3.12 o superior.** No es negociable: `src/config.py` usa `os.listdrives()`, que existe desde 3.12. (Acá se usa 3.13.5.)
- **Git.**
- **GPU NVIDIA con CUDA** para entrenar. Anda en CPU, pero muy lento — sirve para la corrida de humo, no para las 30 épocas.
- **El SSD externo** con la carpeta `DECA-datasets`. Formato exFAT (se lee y escribe desde los dos sistemas).
- Espacio en el SSD: **~117 GB** ocupados hoy (63,4 GB CODE-15% + 7,1 GB PTB-XL + 0,6 GB SaMi-Trop + 45,4 GB del HDF5 de Fase 2).

Nada de esto se guarda en el disco interno ni en git.

---

## Paso 1 — Clonar el repo

```bash
git clone <url-del-repo> DECA-AI
cd DECA-AI
```

## Paso 2 — Entorno virtual

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Si PowerShell se niega con un error de *execution policy*:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**Linux:**
```bash
sudo apt install python3-venv    # si falla el paso siguiente
python3 -m venv .venv
source .venv/bin/activate
```

De acá en adelante, todos los comandos asumen el venv activado.

## Paso 3 — Dependencias

```bash
pip install -r requirements.txt
```

**torch es la excepción y hay que instalarlo aparte.** La rueda que está en PyPI viene **sin CUDA**: si lo instalás con `pip install torch` a secas, `torch.cuda.is_available()` va a devolver `False` y vas a entrenar en CPU sin darte cuenta. Hay que usar el índice de PyTorch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

`cu126` es la versión de CUDA. Para saber cuál te corresponde, mirá el driver:

```bash
nvidia-smi
```

La versión de CUDA que reporta arriba a la derecha es el **máximo** que soporta el driver; podés usar esa o cualquiera menor de las que publica PyTorch (`cu121`, `cu124`, `cu126`, …).

Verificá que quedó bien **antes de seguir**:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Tiene que decir `True` y la versión tiene que terminar en `+cu126` (o la que hayas elegido). Si dice `2.13.0` a secas, instalaste la de PyPI: desinstalá (`pip uninstall torch`) y repetí con el `--index-url`.

> **La descarga son 2,6 GB.** En una red lenta puede tardar más de una hora. Si se corta, `pip install` de nuevo retoma desde el caché.

## Paso 4 — Conectar el SSD y verificar que se detecta

Enchufá el SSD y corré:

```bash
python src/config.py
```

Tiene que salir algo así:

```
Sistema  = Windows (nt)
DATA_DIR = D:\DECA-datasets  (via autodeteccion)

  OK    code15           D:\DECA-datasets\code15
  OK    samitrop         D:\DECA-datasets\samitrop
  OK    ptbxl            D:\DECA-datasets\ptbxl
  OK    metadata.parquet D:\DECA-datasets\metadata.parquet
  OK    fase2 hdf5       D:\DECA-datasets\fase2_preprocessed.hdf5
  OK    fase2 metadata   D:\DECA-datasets\fase2_metadata.parquet
```

**No hay ninguna ruta hardcodeada en el código**: el SSD cambia de letra según el puerto USB (ya montó en `D:` y en `E:`) y en Linux ni siquiera tiene letra. La autodetección recorre los puntos de montaje de cada sistema buscando una carpeta `DECA-datasets` que tenga adentro `code15/`, `samitrop/` y `ptbxl/`.

Si no la encuentra, fijala a mano:

```powershell
# Windows
$env:DECA_DATA_DIR = "D:/DECA-datasets"
```
```bash
# Linux — dejalo en el ~/.bashrc y olvidate
export DECA_DATA_DIR=/media/$USER/SSD/DECA-datasets
```

---

# Escenario A — Entrenar

## A.1 — Chequear las piezas por separado

Cada módulo se auto-reporta al correrlo solo. Son segundos y te ahorran descubrir el problema a mitad de un entrenamiento:

```bash
python src/dataset.py    # cobertura del label de RBBB, splits, pos_weight
python src/model.py      # cantidad de parámetros, forward de prueba
```

`dataset.py` tiene que mostrar:

```
Cobertura del label de RBBB (mascara=1) por dataset:
               n  con_rbbb  positivos_rbbb  % con label
code15    339019  339019.0          9457.0        100.0
ptbxl      21799       0.0             0.0          0.0
samitrop    1545       0.0             0.0          0.0

train   238027 registros  (code15=236944, samitrop=1083)
val      54846 registros  (code15=51371, ptbxl=3245, samitrop=230)
test     54204 registros  (code15=50704, ptbxl=3268, samitrop=232)

pos_weight chagas (train, sin ptbxl) = 29.7
```

**Mirá que `ptbxl` tenga 0 en `con_rbbb`.** Si tiene ~3.100, el merge del label de RBBB se está haciendo mal (los `record_id` no son únicos entre datasets — ver [FASES.md](FASES.md), Fase 4).

`model.py` tiene que decir `6,563,730 parametros`.

## A.2 — Corrida de humo (obligatoria, ~1-2 min)

**La primera corrida no busca un resultado, busca confirmar que no hay errores.** Es una decisión escrita en FASES.md antes de entrenar nada.

```bash
python src/train.py --limit-train 500 --limit-val 2000 --epocas 1 --workers 2 --nombre humo
```

Lo único que prueba es que el pipeline funciona de punta a punta: carga de datos, forma del tensor, pérdida enmascarada sin NaN, checkpoint escrito, loop de validación completo.

⚠️ **El AUC de esta corrida no significa nada.** 500 registros con ~2% de positivos son unos 10 positivos: cualquier número que salga es ruido. No lo interpretes, no lo anotes, no lo compares.

## A.3 — Corrida real

```bash
# Windows / GPU de notebook (12 GB)
python src/train.py --epocas 30 --batch 64 --workers 4

# Linux / RTX 4090 (24 GB)
python src/train.py --epocas 30 --batch 256 --workers 8
```

Cronometrá la primera época antes de dejarlo corriendo: el cuello de botella probablemente sea el I/O del SSD por USB, no la GPU (son ~32 GB leídos por época, en accesos aleatorios de 134 KB).

Los artefactos van a `<SSD>/modelos/<nombre>/`:

| archivo | qué es |
|---|---|
| `mejor.pt` | checkpoint con el mejor AUPRC de arena A en validación |
| `ultimo.pt` | checkpoint de la última época |
| `historia.json` | métricas de todas las arenas, época por época |
| `args.json` | los argumentos exactos de la corrida (para reproducirla) |

## A.4 — Cómo leer la salida

Cada época imprime algo así:

```
  A code15    AUC 0.8xxx  AUPRC 0.0xxx  prev 1.91%
    banda media (sens 95%): espec  xx.x%  PPV  x.x%  deriva xx.x%
    banda alta  (PPV>=30%): sens   xx.x%  PPV xx.x%  deriva  x.x%
  B samitrop  recall  xx.x% (bajo) / xx.x% (alto)   n=230
  C ptbxl     espec   xx.x% (bajo) / xx.x% (alto)   n=3245
  D serologia AUC 0.xxxx  AUPRC 0.xxxx  (230+ vs 50xxx-)
  atajo de fuente: AUC pooled 0.xxxx  delta vs A +0.xxxx
```

Las cuatro **arenas** están separadas a propósito y **nunca hay que mirar una métrica mezclada entre datasets** — SaMi-Trop es 100% positivo y PTB-XL 100% negativo, así que un AUC sobre la mezcla premia a un modelo que solo aprendió a reconocer de qué fuente viene la señal.

- **Arena A (CODE-15% sola) es la que decide el go/no-go.** Una sola fuente con las dos clases, así que ahí el atajo de fuente es imposible por construcción.
- **`delta vs A` es el diagnóstico de atajo, no performance.** Si es grande, el modelo está usando la fuente como proxy de la etiqueta.
- El criterio de go/no-go (AUC ≥0,93 / 0,85-0,93 / <0,85) está fijado **desde antes de entrenar** en FASES.md, Fase 3. No lo renegocies mirando el número que salió.

## A.5 — Ablaciones

```bash
# ¿el peso de la etiqueta serológica (default 3.0) fue buena elección?
python src/train.py --peso-strong 1.0 --nombre abl-peso1
python src/train.py --peso-strong 5.0 --nombre abl-peso5

# ¿sacar PTB-XL del entrenamiento costó algo?
python src/train.py --con-ptbxl --nombre abl-conptbxl

# oversampling en vez de pos_weight
python src/train.py --sampler balanceado --nombre abl-sampler
```

Comparalas **por arena A**, que es inmune al cambio de pesos y hace de juez honesto.

---

# Escenario B — Rehacer el dataset desde cero

Solo si se perdió el SSD o cambió el preprocesamiento. Son varias horas.

## B.1 — Bajar los datasets

No están en git (~71 GB). Se bajan de Zenodo/PhysioNet y se acomodan así:

```
DECA-datasets/
├── code15/      exams_part{0..17}.hdf5, exams.csv, code15_chagas_labels.csv
├── samitrop/    exams.hdf5, exams.csv
└── ptbxl/       ptbxl_database.csv, records500/  (WFDB, ~87.000 archivos chicos)
```

En `scripts/` hay ayudantes de bash para esto (no son parte del pipeline de Python):

```bash
scripts/dl_chunked.sh <url> <salida> <bytes_esperados> <n_chunks> [offset]
```

Baja en chunks paralelos con reintento por chunk y es **idempotente**: si se corta, lo volvés a correr y baja solo lo que falta. Los otros dos (`check_progress.sh`, `rm_progress.sh`) son para ver progreso y borrar carpetas grandes en exFAT, que si no no dan ninguna señal de vida.

## B.2 — Correr el pipeline, en orden

```bash
python src/convert_ptbxl.py       # WFDB -> ptbxl.hdf5 (los otros dos ya son HDF5)
python src/build_metadata.py      # consolida labels y demografía -> metadata.parquet
python src/preprocess.py          # Fase 2 -> fase2_preprocessed.hdf5   (~55 min)
python src/verify_preprocessed.py # barrido de valores no finitos       (~20 min)
```

Notas:

- `convert_ptbxl.py` **se niega a pisar** un `ptbxl.hdf5` existente. Borralo a mano si querés regenerarlo.
- `preprocess.py` acepta `--limit N` para probar rápido antes de tirarse las ~55 minutos.
- **`verify_preprocessed.py` no es opcional.** Hay corrupción en los archivos de origen (un registro de CODE-15% trae NaN y valores de ~1e38 de fábrica) y **un solo NaN que llegue al entrenamiento vuelve NaN la pérdida, después los gradientes y después todos los pesos de la red** — y de ahí no se vuelve. Correlo después de cada `preprocess.py`. Con `--fix` reescribe el parquet sin las filas rotas, con backup previo.

Después seguí con el **escenario A**.

---

# Problemas conocidos

### `No se encontro la carpeta de datos en ningun punto de montaje`
El SSD no está conectado, o está montado en un lugar que la búsqueda no cubre (busca hasta 2 niveles bajo `/media`, `/run/media`, `/mnt`, `/Volumes`, `/data`, `$HOME`). Solución: `export DECA_DATA_DIR=/ruta/exacta`.

### `torch.cuda.is_available()` da `False`
Instalaste la rueda de PyPI, que no trae CUDA. `pip uninstall torch` y reinstalá con `--index-url https://download.pytorch.org/whl/cu126`. Si igual da `False`, chequeá que `nvidia-smi` funcione — sin driver no hay CUDA.

### `OSError: Unable to open file (unable to lock file)` — solo en Linux
HDF5 pide un lock al abrir y sobre exFAT eso falla, aun en solo lectura. El `ECGDataset` ya abre con `locking=False`, así que si ves esto es en otro script. Solución general:
```bash
export HDF5_USE_FILE_LOCKING=FALSE
```

### El SSD se monta de solo lectura en Linux
Pasa cuando Windows lo dejó "sucio" (apagado con *fast startup*, o desconectado sin expulsar). Los checkpoints no se van a poder escribir. Solución: enchufarlo en Windows, expulsarlo bien, y volver a montarlo. Si Ubuntu no lo monta en absoluto, falta el soporte de exFAT:
```bash
sudo apt install exfatprogs
```

### La pérdida da NaN
`train.py` corta con un error explícito en vez de seguir. Corré `python src/verify_preprocessed.py` — es exactamente para esto.

### El entrenamiento va lentísimo y la GPU está al 10%
El cuello de botella es el I/O del SSD por USB, no la GPU. Subí `--workers`, y si no alcanza, copiá el HDF5 al disco interno (son 45 GB). Pasarlo a float16 lo dejaría en ~23 GB y es seguro porque ya está z-scoreado — está anotado como pendiente en FASES.md, todavía sin hacer.

### Los `num_workers` cuelgan en Windows
Windows usa `spawn`, así que todo tiene que ser picklable y el código de entrada tiene que estar bajo `if __name__ == "__main__"` (ya lo está). Si igual cuelga, probá `--workers 0` para descartar que el problema sea otro.

---

# Referencia rápida

```bash
python src/config.py                  # ¿qué ruta resuelve? ¿está todo?
python src/dataset.py                 # splits, cobertura de RBBB, pos_weight
python src/model.py                   # parámetros y forward de prueba
python src/split_patients.py          # reporte del split por paciente

python src/train.py --limit-train 500 --limit-val 2000 --epocas 1 --nombre humo
python src/train.py --epocas 30 --batch 256 --workers 8
```

Flags de `train.py` que importan:

| flag | default | para qué |
|---|---|---|
| `--epocas` | 30 | |
| `--batch` | 64 | 256 en la 4090 |
| `--lr` | 1e-3 | |
| `--peso-strong` | 3.0 | peso de la etiqueta serológica; primera ablación |
| `--peso-rbbb` | 0.5 | peso de la cabeza de RBBB en la pérdida |
| `--con-ptbxl` | off | ablación inversa: mete PTB-XL en el entrenamiento |
| `--sampler` | ninguno | `balanceado` = oversampling en vez de `pos_weight` |
| `--workers` | 4 | |
| `--sin-amp` | — | apaga precisión mixta (para depurar NaN) |
| `--nombre` | timestamp | nombre de la carpeta de la corrida |