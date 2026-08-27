# IA — Fases del proyecto y dificultades

Este documento complementa a [ROADMAP.md](ROADMAP.md): mientras el roadmap define el **qué** (contexto, objetivo, datasets), acá se define el **cómo y en qué orden**, fase por fase, junto con las dificultades esperadas en cada una. Estado actual: **Fase 0 completa y verificada** (los 3 datasets descargados, unificados en HDF5, consolidados en `metadata.parquet` y validados por round-trip). **Fase 1 completa** (las 4 tareas de EDA cubiertas por notebooks reproducibles en `notebooks/`); la **Fase 3 quedó cerrada** (4 decisiones tomadas el 2026-08-08, punto de operación en 95% de sensibilidad con tres bandas y go/no-go escrito antes de entrenar). **Fase 2 completa (2026-08-10)**: diseño corregido y código implementado (`src/split_patients.py`, `src/preprocess.py`), corrido sobre el corpus completo: 362.363 registros preprocesados en `fase2_preprocessed.hdf5`, con el split por paciente verificado sin fuga. **Fase 4 en progreso (arrancada 2026-08-12)**: pipeline de entrenamiento implementado y validado de punta a punta (`src/dataset.py`, `src/model.py`, `src/evaluar.py`, `src/train.py`), corre en Windows y Linux; primera corrida real de 1 época sobre el dataset completo dio AUC 0,80 en la arena de go/no-go, con el diagnóstico de atajo de fuente en la dirección correcta. **Corrida larga de 30 épocas completada (2026-08-25, `real30ep`): AUC arena A converge a ~0,843 y no mejora más — NO-GO contra el criterio de Fase 3 (piso 0,85), y por debajo del mejor modelo hasta ahora (`abl-peso1`, AUPRC 0,1755 vs. 0,1705).** **Ablación de oversampling completada (2026-08-26, `real30ep-sampler`): peor que `abl-peso1`, no rompe el techo.** `peso_strong=1,0` (`abl-peso1`) sigue siendo el mejor modelo global; ninguna de las dos rutas propuestas para el desbalance (subir peso, oversampling) supera el techo de AUC ~0,84. **Sesión de análisis del 2026-08-26 (ver sección propia al final de Fase 4): el AUC ~0,84 resulta estar en el estado del arte publicado (PLOS NTD 2023 da 0,80; el 5° puesto del Moody Challenge 2025 da 0,840), el go/no-go de 0,93 fijado en Fase 3 no lo alcanza nadie en el campo, buena parte del techo es ruido de etiqueta (serología vs. cardiopatía) y no capacidad del modelo, el scheduler de LR venía apagando el entrenamiento solo desde la época ~7, y PTB-XL —hoy excluido del train— contiene los 3 patrones objetivo del ROADMAP como etiqueta validada.**

Convención de estado: 🔲 no iniciada · 🟡 en progreso · ✅ completa.

---

## Fase 0 — Entorno y obtención de datos ✅

Antes de escribir cualquier línea de modelado hay que poder leer un ECG.

**Tareas:**
- [x] Crear entorno de Python para IA (venv/conda) con `h5py`, `numpy`, `pandas`, `scipy` como base, más `wfdb` (solo se usa para convertir PTB-XL de WFDB→HDF5). `.venv` creado en el repo (2026-08-07) con `requirements.txt`.
- [x] Descargar los datasets desde Zenodo/PhysioNet: CODE-15%, SaMi-Trop, PTB-XL (ver tabla en el ROADMAP). Se guardan en el **SSD externo**, no en git. CODE-15% y SaMi-Trop ya estaban; PTB-XL (zip + WFDB) se completó y confirmó el 2026-08-07.
- [x] Unificar a **HDF5** (formato canónico, ver ROADMAP): CODE-15% y SaMi-Trop solo se descomprimen; PTB-XL se convierte. Consolidar labels/demografía en `metadata.parquet`. `python src/convert_ptbxl.py` corrido el 2026-08-07: 21.799 registros, `ptbxl.hdf5` (4.87 GB), sin NaN. Los WFDB de `records500/` ya convertidos se borraron para liberar espacio (se conserva `records100/` y el zip original). `metadata.parquet` regenerado con los 3 datasets (`build_metadata.py`): 366.854 filas, sin NaN, y verificado por round-trip (leer la señal vía `source_file`/`row_index` y confirmar contra el `exam_id` guardado en el HDF5).
  - **Corregido el 2026-08-08:** las 21.799 filas de PTB-XL tenían `row_index=0` (un TODO que quedó de cuando el HDF5 todavía no existía) y `source_file` apuntando a rutas `records500/...` ya borradas. Con eso, leer una señal de PTB-XL desde el metadata devolvía siempre el registro 1 — sin fallar. Ahora `source_file='ptbxl.hdf5'` y `row_index` es la posición real, validada comparando el `exam_id` del HDF5 contra el `ecg_id` del CSV (los `ecg_id` tienen huecos, así que `row_index != ecg_id - 1`).
- [x] Definir cómo se referencian los datos desde el código (variable de entorno / config apuntando a la ruta del SSD, no paths hardcodeados). `src/config.py` resuelve la ruta en dos pasos: si está seteada `DECA_DATA_DIR` la usa, y si no **autodetecta** el SSD recorriendo las unidades montadas en busca de una carpeta `DECA-datasets` que tenga adentro `code15/`, `samitrop/` y `ptbxl/`. Ya no hay ninguna letra de unidad en el código: el SSD cambia de letra según el puerto USB (montó en `E:` y ahora en `D:`), así que fijar una garantizaba romperse tarde o temprano. Para ver qué ruta se está resolviendo: `python src/config.py`.

**Dificultades:**
- **Peso.** CODE-15% son ~46 GB comprimidos (cientos de miles de registros); no entra en git ni conviene en el disco interno. Por eso vive en el SSD externo.
- **exFAT y archivos chicos.** El SSD es exFAT: rinde bien con pocos archivos grandes (HDF5) y mal con cientos de miles de archivos chicos (motivo extra para no usar WFDB por-registro en CODE-15%).
- **Formato HDF5 al principio.** Primer contacto con `h5py` y lectura de señales multicanal por bloques (`x[start:end]`) — curva de aprendizaje corta pero real. Para PTB-XL además hay que leer WFDB una vez para convertirlo.
- **Datasets privados de Centro/Sudamérica** mencionados en el ROADMAP no son públicos — esta fase solo cubre CODE-15%, SaMi-Trop y PTB-XL, y queda pendiente gestionar el acceso a los privados más adelante (posible cuello de botella institucional, no técnico).

---

## Fase 1 — Exploración de datos (EDA) ✅

**Tareas:**
- [x] Cargar una muestra de cada dataset y visualizar señales crudas (las 12 derivaciones). → `notebooks/01_eda_senales.ipynb`.
- [x] Revisar distribución de edades, sexo, frecuencia de muestreo real vs. nominal, duración real de cada registro. → `notebooks/02_distribuciones.ipynb` (edad/sexo, sobre los 366.854 registros completos) + `notebooks/04_calidad_senal.ipynb` (duración real, sobre muestra).
- [x] Cuantificar el desbalance de clases por dataset (CODE-15% ~2% positivos, SaMi-Trop 100% positivos por serología — ver nota resuelta más abajo). → `notebooks/03_desbalance_clases.ipynb`.
- [x] Revisar calidad de señal: ruido, artefactos de movimiento, derivaciones faltantes o corruptas. → `notebooks/04_calidad_senal.ipynb`.

**Dificultades:**
- **Tres datasets, tres realidades distintas.** No se pueden explorar como si fueran uno solo: CODE-15% es population-based con etiqueta autorreportada, SaMi-Trop es una cohorte de enfermos con etiqueta serológica, PTB-XL es de una región no endémica y sirve como negativo "por presunción" más que por diagnóstico confirmado. Cualquier estadística agregada sin discriminar por origen va a ser engañosa.
- **Filas de padding en CODE-15%.** Cada uno de los 18 `exams_part{n}.hdf5` trae una fila extra al final con `exam_id=0` y la señal entera en ceros (por eso son 20.001 filas y no 20.000; 18 en total). Vienen así desde Zenodo. `metadata.parquet` no las incluye —el merge contra `exams.csv` las descarta solas— pero **cualquier recorrido directo de `tracings` las va a encontrar** y van a aparecer como "registros corruptos" en los chequeos de calidad de señal. Recorrer siempre vía metadata, o filtrar `exam_id != 0`.
- **Ruido de señal real.** Estos son ECG de campo (telesalud en zonas rurales para CODE-15%/SaMi-Trop), no señales de laboratorio — esperar artefactos, baseline wander, ruido de línea eléctrica.
- **Volumen para EDA manual.** Con cientos de miles de registros no se puede inspeccionar todo a ojo; hay que definir criterios automáticos de calidad de señal antes de poder confiar en agregados.

### Mediciones ya hechas (2026-08-08, confirmadas y formalizadas el 2026-08-10)

Adelantadas sobre muestras de 200 registros por dataset, porque condicionan el diseño de la Fase 2. Reproducidas el 2026-08-10 en `notebooks/04_calidad_senal.ipynb` sobre una muestra nueva de 300 registros/dataset (semilla fija) — números consistentes, diferencias solo por tamaño de muestra:

| | % de registros con padding | \|amplitud\| máx. mediana | duración real mediana |
|---|---|---|---|
| CODE-15% | 63,5% (64,7% en la muestra de confirmación) | 4,08 (4,29) | 7,33 s de 10,24 nominales |
| SaMi-Trop | 52,5% (48,3%) | 4,33 (4,50) | 9,82 s (10,24 s) de 10,24 |
| PTB-XL | **0,0%** | **1,82 (1,94)** | 10,00 s de 10,00 |

**Hallazgos nuevos del 2026-08-10** (`notebooks/04_calidad_senal.ipynb`, muestra de 300/dataset): sin NaN en ninguna de las 900 señales muestreadas; derivaciones planas (posible electrodo desconectado) en solo 2 registros de 900 (1 en CODE-15% con 1 derivación plana, 1 en SaMi-Trop con 2) — infrecuente, pero a filtrar explícitamente en Fase 2 en vez de asumir que no existe.

**Anormalidades ECG disponibles como etiqueta en CODE-15%.** `exams.csv` trae, por registro y para los 343.424 con label de Chagas: `1dAVb`, `RBBB`, `LBBB`, `SB`, `ST`, `AF`, `normal_ecg`. Cruzadas contra `chagas`:

| | en Chagas+ | en Chagas− | enriquecimiento |
|---|---|---|---|
| **RBBB** (bloqueo de rama derecha) | **19,97%** | 2,43% | **8,22×** |
| AF (fibrilación auricular) | 6,16% | 1,94% | 3,17× |
| 1dAVb (bloqueo AV 1er grado) | 4,07% | 1,60% | 2,54× |
| LBBB | 3,40% | 1,71% | 1,99× |
| SB | 3,17% | 1,59% | 1,99× |
| ST | 1,57% | 2,22% | 0,71× |
| normal_ecg | 16,25% | 39,58% | 0,41× |

Es el hallazgo más importante hasta acá: **RBBB es el BRD del ROADMAP**, uno de los 3 patrones objetivo, y está disponible como ground truth por registro en 343.424 ECG. El enriquecimiento de 8,22× confirma con datos la premisa clínica del proyecto. SaMi-Trop solo aporta `normal_ecg` (17,54%, o sea 82% anormales, coherente con una cohorte de enfermos).

---

## Fase 2 — Preprocesamiento y unificación ✅

**Tareas:**
- Resamplear todo a una frecuencia común (400 Hz vs 500 Hz según dataset).
- Definir una duración/ventana estándar de señal para alimentar al modelo (los registros varían entre 7.3s y 10.2s).
- Integrar las etiquetas externas (`code15_chagas_labels.csv`, `exams.csv`, `ptbxl_database.csv`) al pipeline de forma consistente.
- Etiquetar cada muestra con su tipo de "confianza" (weak/self-reported vs. strong/serológica) para poder ponderarla distinto en el entrenamiento.
- Diseñar el split train/val/test teniendo en cuenta que el desbalance y la fuente de la etiqueta difieren por dataset (evitar que todo el positivo confiable termine en un solo split).

**Dificultades:**
- **Mezclar frecuencias y duraciones sin introducir sesgos.** Resamplear no es gratis: hay que validar que no se pierdan las morfologías relevantes (ondas Q, QRS ancho) que son justamente los patrones objetivo.
- **Etiquetas de confianza distinta conviviendo en el mismo dataset de entrenamiento.** Si se tratan igual, el ruido de las etiquetas autorreportadas de CODE-15% puede contaminar el aprendizaje de patrones que en SaMi-Trop están bien validados.
- **Split no trivial.** Un split aleatorio simple puede terminar con casi todos los positivos confiables (SaMi-Trop) en un solo conjunto, dejando al modelo sin señal fuerte en otro. Hay que diseñar el split a propósito, estratificando por dataset de origen y por label.
- **Fuga de información (data leakage).** Si un mismo paciente aparece en más de un registro/dataset, hay que evitar que termine partido entre train y test. **Medido el 2026-08-08: no es hipotético, es masivo.** CODE-15% tiene 345.779 exámenes de solo **233.770 pacientes**; 66.929 pacientes tienen más de un examen y uno tiene **38**. Son **112.009 filas (32% del dataset)** que son un examen repetido de alguien ya visto. Un split aleatorio por examen pone al mismo paciente de los dos lados y el modelo memoriza pacientes. El split va **por `patient_id`** (columna presente tanto en `exams.csv` como en `code15_chagas_labels.csv`). A nivel paciente la prevalencia es 1,90% (4.444 de 233.513).

### Diseño acordado (2026-08-08)

Ataca de frente el atajo de la fuente medido en la Fase 1:

- **Ventana común de 7,0 s a 400 Hz = 2.800 muestras**, recortada sobre señal real y descartando el padding. 7,0 s porque el registro más corto de CODE-15% tiene 7,33 s útiles: entra en los tres datasets sin rellenar nada. **Elimina el atajo del padding.**
- **PTB-XL de 500 → 400 Hz** con `resample_poly` (factor exacto 4/5).
- **Z-score por registro y por derivación**, no global: normalizar cada registro contra sí mismo **elimina la diferencia de escala entre datasets**. Cualquier estadístico que sí sea global se calcula solo sobre train.
- **Split por `patient_id`**, estratificado por dataset de origen y por label.
- Salida: HDF5 ya normalizado `(N, 2800, 12)`, listo para copiar a la máquina de entrenamiento (no se mueven los 72 GB crudos).

### Corrección e implementación (2026-08-10)

**Corrección al diseño anterior.** La afirmación "7,0 s entra en los tres datasets sin rellenar nada" nunca se había chequeado contra SaMi-Trop, solo contra CODE-15%. Medido con censo completo/muestra grande antes de implementar:

| dataset | mínimo real | % de registros con <7,0s |
|---|---|---|
| SaMi-Trop (censo completo) | 3,92 s | **5,27%** (86 de 1.631) |
| CODE-15% (muestra de 5.000) | 0,00 s | 1,12% (incluye al menos 1 registro **totalmente vacío**, no solo corto) |
| PTB-XL (censo completo) | 10,00 s | 0% |

**Decisión: esos registros se descartan del dataset, no se rellenan con ceros.** Rellenar reintroduciría, aunque sea en una fracción chica de registros, exactamente el atajo de padding que este diseño existe para eliminar (ver Fase 4: el padding correlaciona con dataset/label). Se pierde 5,27% de los positivos fuertes de SaMi-Trop (quedan ~1.545), un costo aceptado a cambio de no filtrar la señal de padding.

**Ventana centrada, no desde el inicio.** Se recortan los 2.800 muestras del medio de la señal real, no las primeras: en `notebooks/01_eda_senales.ipynb` se observaron transientes/picos justo donde arranca la señal real después del padding, consistentes con el electrodo asentándose — un artefacto de contacto, no señal clínica útil.

**`patient_id` para el split.** Se agregó como columna a `metadata.parquet` (`build_metadata.py`). CODE-15% y PTB-XL lo traen en la fuente (PTB-XL también tiene pacientes repetidos: 2.111 de 18.869). **SaMi-Trop no tiene columna de paciente en `exams.csv`** — se usa `record_id` como proxy, asumiendo 1 examen por paciente (verificado: sin `exam_id` duplicados en los 1.631 registros).

**Split 70/15/15** (no 80/10/10): con solo ~1.631 positivos fuertes en SaMi-Trop (menos aún tras descartar los <7,0s), conviene priorizar un val/test más grande para calibrar el umbral de 95% de sensibilidad (Fase 3) y medir AUC con confianza — el costo en volumen de entrenamiento es marginal porque CODE-15% domina el total sin importar el ratio exacto.

**Split por paciente, sin tratar distinto a quién tiene 1 examen vs. varios.** Agrupar por `patient_id` y mandar todos los exámenes de un paciente al mismo split ya elimina la fuga de información — no hace falta (ni conviene) forzar a los pacientes con exámenes repetidos a ir todos a train, porque eso sesgaría val/test hacia "pacientes de un solo examen" sin ganar nada adicional en fuga. Verificado tras implementar: 0 pacientes con más de un split asignado.

**Implementado:**
- `src/split_patients.py`: agrupa por `(dataset, patient_id)`, estratifica por `(dataset, label a nivel paciente)`, split 70/15/15 con semilla fija. Verificado sin fuga y con prevalencia de positivos consistente entre splits (~2,2-2,25%).
- `src/preprocess.py`: recorte de padding → descarte si <7,0s → descarte si hay valores no finitos → resampleo (solo PTB-XL) → ventana centrada de 2.800 muestras → z-score por registro/derivación (con guarda contra división por 0 en derivaciones planas). Salida: `D:\DECA-datasets\fase2_preprocessed.hdf5` + `fase2_metadata.parquet` (rutas en `src/config.py`).
- `src/verify_preprocessed.py`: barre el HDF5 de salida buscando ventanas con valores no finitos y las cruza contra el metadata, distinguiendo las **referenciadas** (llegan al entrenamiento) de las **huérfanas** (están en el HDF5 pero ningún registro del metadata las nombra — inofensivas). Sin argumentos solo reporta y sale con código 1 si hay problemas; con `--fix` reescribe el parquet sin esas filas, con backup previo. Correrlo después de cada corrida de `preprocess.py` y antes de copiar el HDF5 a la máquina de entrenamiento. Barrido completo ≈20 min sobre el SSD externo.

### Corrida sobre el corpus completo (2026-08-10) — Fase 2 CERRADA

Corrida completa: **55 min 40 s** para los 366.854 registros (~130 registros/s). Salida **48,7 GB**.

**362.363 registros** quedaron en el dataset final. Descartes:

| dataset | <7,0 s de señal real | % del dataset | corruptos |
|---|---|---|---|
| code15 | 4.404 | 1,27% | 1 |
| samitrop | 86 | **5,27%** | 0 |
| ptbxl | 0 | 0% | 0 |

Los 86 de SaMi-Trop coinciden exactamente con el censo completo previo: el descarte se comportó como se había medido.

**Split final, verificado sin fuga y con positivos parejos:**

| split | code15 | ptbxl | samitrop | % positivos |
|---|---|---|---|---|
| train | 236.944 | 15.286 | 1.083 | 2,23% |
| val | 51.371 | 3.245 | 230 | 2,25% |
| test | 50.704 | 3.268 | 232 | 2,22% |

**Hallazgo nuevo: hay corrupción en los archivos de origen.** La primera corrida tiró `RuntimeWarning: overflow` de NumPy. Barrido completo de los 48,7 GB buscando valores no finitos: **1 registro de 362.364 (0,0003%)**, el `record_id` **2858700** (`code15/exams_part2.hdf5`, fila 14712, split train, label negativo). El archivo crudo trae **23 celdas ya en NaN** y **1.544 muestras de magnitud ~1e38** mezcladas con señal normal de ~0,03 mV. No lo generó el pipeline: 10 de las 12 derivaciones ya venían con NaN de fábrica; en las otras 2, los 1e38 desbordaron float32 al elevar al cuadrado para el desvío (el cuadrado desborda a partir de ~1e19), dando `inf`, y después `inf − inf` → `NaN`.

**Por qué importa aunque sea 1 en 362.364:** un solo NaN en un lote de entrenamiento vuelve NaN la pérdida, después los gradientes y después *todos* los pesos de la red — y ya no se recupera, la red devuelve NaN para cualquier entrada. El síntoma en Fase 4 sería una pérdida que explota sin causa aparente.

**Resuelto así:** `procesar_registro()` ahora descarta el registro si `np.isfinite()` falla sobre la señal real. El chequeo va **antes del resampleo** a propósito: `resample_poly` es una convolución, así que un NaN fuera de la ventana se esparciría hacia adentro. La función pasó a devolver `(ventana, motivo)` para poder contar los dos tipos de descarte por separado. No se volvió a correr el pipeline entero (56 min por 1 registro): se filtró esa fila de `fase2_metadata.parquet`, que es lo que lee el entrenamiento — la fila basura sigue en el HDF5 pero ninguna fila del metadata la referencia. Backup del metadata sin filtrar en `fase2_metadata_sin_filtrar.parquet`.

Ese filtrado se hizo a mano una vez y por eso no era reproducible: si se regenera el parquet, el filtro se pierde. Desde el 2026-08-11 lo reemplaza `src/verify_preprocessed.py --fix`, que no hardcodea el `record_id` 2858700 sino que vuelve a hacer el barrido — encuentre lo que encuentre. Es el mismo chequeo que ya está dentro de `preprocess.py`, pero aplicado del otro lado: allá se valida lo que entra desde los archivos crudos, acá lo que quedó escrito en la salida.

**Verificado el 2026-08-11** con un barrido completo e independiente de los 362.364 registros (21 min): **1 fila con valores no finitos, huérfana, 0 referenciadas por el metadata**. Confirma las dos cosas que el arreglo manual no había demostrado — que esa fila era la única del corpus, y que el parquet que lee el entrenamiento no la nombra. El dataset de Fase 2 queda avalado para entrenar.

**Pendiente menor para Fase 4:** el HDF5 está en float32 (48,7 GB). Pasarlo a float16 lo deja en ~24 GB, seguro *porque* ya está z-scoreado (los valores quedan en un rango chico alrededor de 0). Relevante solo al momento de copiarlo a la máquina de entrenamiento; no cambia nada del diseño.

---

## Fase 3 — Definición operacional de la tarea ✅

Esta fase es más de decisión que de código, pero es bloqueante para las siguientes. **Cerrada el 2026-08-08** con las 4 decisiones tomadas y el punto de operación definido.

**Tareas:**
- Definir qué cuenta como "positivo" (indicio a derivar) vs. "negativo", en línea con los 3 patrones objetivo del ROADMAP (BRD+HAI, extrasístoles ventriculares, zonas eléctricamente inactivas).
- Elegir la métrica de éxito para un caso de uso de screening: priorizar sensibilidad/recall (falso negativo = paciente con Chagas no derivado, mucho más costoso que un falso positivo que solo deriva a un test gratuito). La métrica la definimos nosotros según el caso de uso; **no** adoptamos la forma de evaluar del challenge.

### Las 4 decisiones — RESUELTAS el 2026-08-08

Resumen de lo acordado; el detalle de cada una queda abajo.

1. **Multi-tarea Chagas + RBBB**, con loss enmascarada para SaMi-Trop/PTB-XL (que no tienen label de RBBB). **AUPRC loggeado por cabeza y por separado**: si la cabeza de RBBB anda mal, el modelo estaría "explicando" con una señal poco confiable, que es peor que no explicar. La explicación por BRD se muestra solo si esa cabeza supera un umbral de confianza propio.
2. **Punto de operación: 95% de sensibilidad** (se acepta perder 1 de cada 20 enfermos), implementado con **tres bandas** en vez de un sí/no, y con un **criterio de go/no-go fijado antes de entrenar**. Detalle completo en la sección "El punto de operación, decidido" más abajo. Métrica de calidad: **AUPRC**, no AUC-ROC.
3. **Evaluación por paciente**, agregando por `patient_id`. Agregación por máximo como default, **comparada contra promedio y voto** por el sesgo cuantificado abajo.
4. **Entrenar con las tres fuentes ponderadas, evaluar solo contra serología**, más un análisis aparte que mide cuánto cae el recall al evaluar también contra autorreporte — como evidencia de que separar las evaluaciones no es prolijidad sino necesidad.

### El detalle

**1. Qué predice el modelo.** Tres opciones, y el hallazgo de `RBBB` en la Fase 1 abre la tercera:
   - (a) **Chagas directo.** Un solo target binario. Simple, pero mezcla serología, autorreporte y presunción en la misma columna.
   - (b) **Los 3 patrones ECG.** Requiere anotación por cardiólogo que no existe. Inviable hoy salvo para un subconjunto.
   - (c) **Multi-tarea: Chagas + RBBB** (y opcionalmente AF, 1dAVb). Aprovecha que RBBB = el BRD del ROADMAP está disponible en 343.424 registros con enriquecimiento 8,22× en Chagas+. Da una señal de supervisión mucho más densa que el 1,9% de positivos, y hace el modelo explicable ("se deriva porque hay BRD"), que es lo que el ROADMAP promete. Costo: solo CODE-15% tiene estas columnas.

**2. El punto de operación.** No alcanza con "priorizar recall": hay que fijar un piso de especificidad. Con prevalencia 1,90%, sobre 100.000 personas (1.900 casos reales):

| sensibilidad | especificidad | detectados | perdidos | serologías | PPV | tests por caso |
|---|---|---|---|---|---|---|
| 95% | 80% | 1.805 | 95 | 21.425 | 8,4% | 11,9 |
| 95% | 90% | 1.805 | 95 | 11.615 | 15,5% | 6,4 |
| 95% | 95% | 1.805 | 95 | 6.710 | 26,9% | 3,7 |
| 95% | 99% | 1.805 | 95 | 2.786 | 64,8% | 1,5 |
| 90% | 95% | 1.710 | 190 | 6.615 | 25,9% | 3,9 |
| 80% | 95% | 1.520 | 380 | 6.425 | 23,7% | 4,2 |

Leer la columna "tests por caso" como la pregunta real: **cuántas serologías está dispuesto a pagar el sistema de salud por cada caso que encuentra.** Bajar la especificidad de 95% a 80% triplica el costo sin detectar un solo caso más.

**Corrección importante a esa tabla.** Presenta sensibilidad y especificidad como dos perillas independientes, y **no lo son**: las ata la capacidad discriminativa del modelo. Sobre un modelo binormal (negativos ~N(0,1), positivos ~N(d,1), d = √2·Φ⁻¹(AUC)) — es una aproximación para planificar, no una promesa:

| AUC | espec @ sens 90% | espec @ sens 95% | espec @ sens 98% | espec @ sens 99% |
|---|---|---|---|---|
| 0,80 | 46,4% | 32,5% | 19,4% | 12,8% |
| 0,85 | 57,3% | 42,9% | 27,8% | 19,5% |
| 0,90 | 70,2% | 56,7% | 40,5% | 30,4% |
| 0,95 | 85,2% | 75,2% | 60,7% | 50,0% |
| 0,98 | 94,8% | 89,6% | 80,3% | 71,8% |

Con un AUC realista de 0,85, exigir 99% de sensibilidad deja la especificidad en **19,5%**: sobre 100.000 personas son **80.878 serologías** (43 por caso detectado), o sea derivar al 81% de la población. La herramienta deja de filtrar. Para sostener 98-99% de sensibilidad con carga manejable hace falta AUC ≥ 0,95, que sería estado del arte.

**Por eso el punto de operación no se fija mirando solo la sensibilidad.** La pregunta a responder con el sistema de salud es **cuál es la capacidad de serología disponible**, que es la restricción que manda.

### El punto de operación, decidido (2026-08-08)

#### En castellano, sin jerga

El modelo no dice "sí" o "no": devuelve **un número del 0 al 100** que indica cuán sospechoso le parece el ECG. Alguien tiene que elegir **dónde cortar** — arriba del corte se deriva a análisis de sangre, abajo no. Toda esta sección es sobre dónde poner ese corte.

Es la perilla de sensibilidad de un detector de metales. **Subirla al máximo** no deja pasar ni un cuchillo, pero suena con hebillas y monedas, la fila se frena y a la semana los guardias ignoran la alarma. **Bajarla** hace fluir la fila pero deja pasar un cuchillo. No existe la perilla que suene con todos los cuchillos y con nada más.

Sobre un pueblo de 1.000 personas (19 con Chagas sin saberlo, 981 sanas), hay dos errores posibles:
- **Se escapa un enfermo:** saca un puntaje bajo, le decimos "sin indicios", se va tranquilo y no se trata. **Grave.**
- **Se asusta a un sano:** saca puntaje alto, va al análisis, da negativo. Se preocupó al pedo y se gastó un análisis. **Molesto pero barato.**

Bajar el corte reduce el primer error y aumenta el segundo. Siempre. Los dos a la vez no se puede.

Los nombres técnicos son solo eso: **sensibilidad** = de los 19 enfermos, a cuántos agarré. **Especificidad** = de los 981 sanos, a cuántos dejé tranquilos.

Y el **AUC** es la calidad del aparato, no del corte: un detector barato confunde monedas con cuchillos y para atrapar todos los cuchillos obliga a revisar medio aeropuerto; uno caro atrapa los mismos revisando a 20 personas. Los dos atrapan todo — la diferencia es a cuánta gente inocente molestan para lograrlo.

#### La decisión

**Se acepta perder 1 de cada 20 enfermos (95% de sensibilidad).**

**Por qué no cero.** Para no perder a ninguno, con un modelo realista habría que derivar al **81% de la población**. Ahí el modelo no filtra nada: es igual a no tenerlo y testear a todos. Y peor — nadie sostiene un programa así, la herramienta se abandona, y una herramienta abandonada no pierde 1 de cada 20 enfermos: **los pierde a todos**. 95% desplegado le gana a 99% archivado.

**Por qué no menos.** Abajo de 95% se pierde más de 1 de cada 20 y el mensaje "sin indicios" pasa a ser peligroso — es la falsa tranquilidad marcada como riesgo en la Fase 6.

**Qué hace tolerable ese 5%.** Que **el modelo nunca es la única puerta**: quien tiene criterio epidemiológico (zona endémica, madre con Chagas) va a serología igual, dispare o no el ECG. El modelo es una red adicional para pescar a los que nadie hubiera mirado, no el único filtro.

#### Tres bandas, no un sí/no

| banda | umbral | deriva | mensaje |
|---|---|---|---|
| **Alta** | PPV ≥ 30% | ~2-3% de la población | Derivación prioritaria. 1 de cada 3 da positivo (contra 1 de cada 53 tirando al azar). Es la banda que hace que el médico confíe en la herramienta. |
| **Media** | hasta el umbral de 95% sens | el resto de los derivados | Derivación normal. |
| **Baja** | por debajo | no deriva | **"No se detectaron indicios. Esto no descarta Chagas."** El texto es obligatorio: ahí adentro va 1 de cada 20 enfermos. |

Especificidad necesaria para la banda alta: ~97,7% (con sensibilidad 30% en esa banda) o ~97,7-98,6% según el corte exacto; se calibra sobre validación.

#### Go/no-go, fijado ANTES de entrenar

Se deja escrito de antemano para no racionalizar después el número que toque:

- **AUC ≥ 0,93** → se despliega completo, las tres bandas. Deriva ≤34% de la población, ~19 serologías por caso detectado.
- **AUC 0,85–0,93** → **no se despliega como descarte.** Solo la banda alta, como priorizador de a quién testear primero cuando la capacidad no alcanza. Ese uso no requiere sensibilidad alta.
- **AUC < 0,85** → no se usa.

Cuánta gente se deriva al fijar la sensibilidad en 95%, según lo bueno que salga el modelo:

| AUC | especificidad | **derivados** | serologías por caso | perdidos c/100k |
|---|---|---|---|---|
| 0,85 | 42,9% | 57,8% | 32,0 | 95 |
| 0,90 | 56,7% | 44,3% | 24,6 | 95 |
| **0,93** | 67,1% | **34,1%** | 18,9 | 95 |
| 0,95 | 75,2% | 26,1% | 14,5 | 95 |

#### Regla operativa

**Umbral bajo** = el que da 95% de sensibilidad sobre etiqueta serológica, a nivel paciente. **Umbral alto** = el que da PPV ≥ 30%. Los dos se calculan **post-hoc sobre validación, nunca sobre test**.

**3. La unidad de evaluación: paciente o examen.** Con 32% de exámenes repetidos, evaluar por examen infla las métricas. Clínicamente lo que importa es el paciente. Se agrega por `patient_id` y se reporta a nivel paciente.

**El sesgo de agregar por máximo, cuantificado.** Un paciente sano con k exámenes tiene FPR = 1−(1−p)^k: a más exámenes, más chances de que uno dispare por ruido. Sobre la distribución real de CODE-15% (230.894 pacientes sanos, hasta 31 exámenes cada uno):

| espec. por examen | espec. por paciente (max) | espec. por paciente (promedio\*) |
|---|---|---|
| 99,0% | 98,5% | 99,3% |
| 95,0% | **92,9%** | 96,2% |
| 90,0% | 86,2% | 92,1% |
| 80,0% | 73,6% | 82,9% |

\* si el ruido es independiente entre exámenes del mismo paciente, promediar lo reduce ~√k. Es un límite optimista, pero muestra que **el sesgo va al revés**: `max` castiga al paciente con muchos exámenes, `mean` lo premia.

El costo de `max` es real pero acotado: ~2 puntos de especificidad a un nivel de trabajo de 95%. Lo llamativo es la concentración: con 95% de especificidad por examen, el 28% de los pacientes (los que tienen más de un examen) genera el **49,3% de todos los falsos positivos**. Se implementan las tres agregaciones (`max`, `mean`, voto) y se comparan; `max` queda como default por ser el conservador clínicamente (no perder un examen anómalo aislado).

**4. Qué hacer con las etiquetas de confianza distinta.** Entrenar con las tres ponderadas, pero **evaluar solo contra etiqueta fuerte** (serología). Reportar el desempeño sobre etiqueta débil por separado, nunca mezclado.

**Agregado:** incluir un análisis aparte (apéndice del reporte de Fase 5) que mida **cuánto cae el recall al evaluar también contra las etiquetas de autorreporte**. Si la caída es grande, es la evidencia de que separar las evaluaciones no es prolijidad metodológica sino que el autorreporte mete ruido medible. Si es chica, el supuesto de que el autorreporte es "débil" queda debilitado y habría que revisar la ponderación del entrenamiento — sirve en los dos sentidos.

**Enmienda a la decisión 4 (2026-08-12, ver detalle en Fase 4):** "evaluar solo contra etiqueta fuerte" no es computable tal cual está escrito — los 1.545 registros `strong` (SaMi-Trop) son 100% positivos, y sobre una sola clase no existe AUC, AUPRC, especificidad ni PPV, solo recall. El punto de operación y el go/no-go de más arriba (que sí necesitan las dos clases) se calculan sobre **CODE-15% sola** en su lugar, y la serología entra por dos vías separadas y explícitas: recall puro sobre SaMi-Trop, y un par artificial SaMi-Trop(+) vs. CODE-15%(−) que resulta válido porque esas dos fuentes son difíciles de distinguir entre sí (AUC 0,598, ver Fase 4). El detalle completo, con el argumento de por qué CODE-15% es la arena correcta, está en Fase 4.

**Dificultades:**
- ~~**SaMi-Trop: 93% vs 100% positivos, sin resolver.**~~ **Resuelto (2026-08-10):** SaMi-Trop es la cohorte de pacientes con Chagas confirmado por serología, 100% positivo — es *el* dataset de Chagas por serología, no una muestra poblacional con el ~93% que menciona el ROADMAP para "según la fuente". El `exams.hdf5`/`exams.csv` (1.631 registros) no trae columna `chagas` porque no hace falta: todos los registros son positivos por diseño de la cohorte. `build_metadata.py` ya lo tenía hardcodeado a `True`; el hardcodeo queda confirmado, no es un supuesto sin validar.
- **No hay ground truth directo de los 3 patrones ECG.** Las etiquetas disponibles son "Chagas sí/no" (por serología o autorreporte), no "presenta BRD+HAI sí/no". Esto significa que el modelo aprenderá a predecir la enfermedad de forma indirecta, no los patrones específicos mencionados en el objetivo — hay que decidir si eso es aceptable o si se necesita anotación adicional (posiblemente manual, por cardiólogo) para un subconjunto.
- **Tensión sensibilidad vs. especificidad.** Optimizar solo para recall puede disparar los falsos positivos a un nivel que vuelva la herramienta inútil en la práctica (saturar el sistema de testeo serológico). Hay que fijar un piso de especificidad aceptable, no solo maximizar sensibilidad.

---

## Fase 4 — Modelado (baseline → iteración) 🟡

**Tareas:**
- Definir enfoque: señal cruda (deep learning, ej. CNN/RNN sobre las 12 derivaciones) vs. features clásicas de ECG (intervalos, morfología de onda) + modelo clásico. **Decidido (2026-08-12): señal cruda**, CNN 1D estilo ResNet1D (Ribeiro et al., la red de referencia para este dataset exacto), multi-tarea Chagas + RBBB con loss enmascarada, tal como definió la Fase 3.
- Levantar un baseline simple primero (aunque sea débil) antes de ir a arquitecturas complejas.
- Iterar arquitectura/hiperparámetros con validación cruzada respetando los splits por dataset definidos en la Fase 2.

**Dificultades:**
- **Desbalance extremo de clases** (2% positivos en el dataset más grande) — requiere técnicas específicas (pesos de clase, resampling, focal loss) y complica la elección de umbral de decisión.
- **Recursos de cómputo.** Entrenar sobre cientos de miles de señales de 12 derivaciones no es liviano; hay que dimensionar si se necesita GPU y de dónde sale.
- **Riesgo de overfitting a la fuente del dato, no a la enfermedad.** Si el modelo aprende a distinguir "viene de SaMi-Trop" vs. "viene de PTB-XL" (por diferencias de equipo, región, calidad de señal) en vez de patrones clínicos reales, va a tener buen desempeño en test pero fallar en producción con datos nuevos. **Cuantificado el 2026-08-08 y peor de lo que suena:** como SaMi-Trop es 100% positivo y PTB-XL 100% negativo, *saber de qué dataset viene un registro equivale a saber la etiqueta*, y las señales delatan su origen con una regla de dos líneas (ver tabla en Fase 1: PTB-XL no tiene padding en ningún registro y vive en otra escala de amplitud). El preprocesado de la Fase 2 está diseñado para borrar las dos pistas, pero **toda métrica sospechosamente buena se audita contra esto primero**.

### Tercera pista de fuente encontrada, y por qué no se filtra (2026-08-12)

Antes de escribir código de entrenamiento se midió si las dos pistas de la Fase 2 (padding, escala de amplitud) eran las únicas. **No lo eran.** Sobre 200 registros por dataset del HDF5 ya preprocesado (z-scoreado), la fracción de energía en la banda 40-100 Hz por sí sola separa PTB-XL de CODE-15% con **AUC 0,880**:

| fracción de energía en 40-100 Hz (mediana) | code15 | samitrop | ptbxl |
|---|---|---|---|
| | 0,00228 | 0,00073 | **0,00909** (4× code15) |

Hipótesis inicial: ruido de línea eléctrica (PTB-XL es europeo, 50 Hz; CODE-15%/SaMi-Trop son brasileños, 60 Hz). **Medida y descartada**: el ratio de potencia en 48-52 Hz vs. 58-62 Hz no separa los datasets de forma consistente con esa hipótesis. El delator es una diferencia más general de ancho de banda/contenido de alta frecuencia entre el equipo de PTB-XL (Alemania, años 90) y el de CODE-15%/SaMi-Trop (Brasil, 2010s), no específicamente la red eléctrica.

**Se probó la solución obvia (pasabajos a 40 Hz) y no alcanza**: baja el AUC de separación de 0,880 a 0,803, sigue siendo una pista fuerte. No se implementa — no vale el costo (reprocesar y volver a versionar los datos) por una mejora parcial. Contraparte importante: **SaMi-Trop y CODE-15% son difíciles de distinguir entre sí** (AUC 0,598, casi azar) — esto es lo que habilita la arena serológica de la sección siguiente.

**Decidido (2026-08-12): PTB-XL se saca del entrenamiento por completo, no solo con peso reducido.** Sigue procesándose (val/test lo necesitan), pero el DataLoader de entrenamiento lo excluye por default — así no hay ninguna oportunidad, ni parcial, de aprender el atajo del origen. Pasa a ser exclusivamente el conjunto de la **arena C, especificidad cruzada de población** (más abajo), que con esta decisión mide generalización real: el modelo nunca vio una señal de PTB-XL entrenando. El costo en volumen es chico (15.286 negativos menos, contra ~232.000 que ya aporta CODE-15% solo). Queda un flag `--con-ptbxl` para correr la ablación inversa y confirmar que sacarlo no perdía nada.

**Decidido (2026-08-12): el desbalance de clases (1,9%-2,8% positivos) se maneja con `pos_weight` derivado de las masas ponderadas, no con oversampling, como default.** Sobremuestrear los 1.083 registros de SaMi-Trop en train para verlos ~15-20 veces por época acumula cientos de exposiciones a un puñado de grabaciones en 30 épocas, con augmentación (ruido gaussiano, dropout de derivación) que no está pensada para borrar la pista espectral recién encontrada — el ruido aditivo sube contenido de alta frecuencia, no lo baja. Oversampling queda como ablación (`--sampler balanceado`, con `pos_weight` recalculado para no apilar los dos mecanismos), a correr después del primer resultado y bajo el mismo test de señal barajada de más abajo.

**Acordado (2026-08-12): el primer modelo que se corre no busca un resultado, busca confirmar que no hay errores.** Antes de la primera corrida con volumen real (~2 h), se corre sobre una muestra muy chica (`--limit-train 500 --epocas 1`, <2 min) — el único objetivo es que el pipeline entero funcione de punta a punta sin romperse: carga de datos, forma del tensor, pérdida enmascarada sin NaN, checkpoint escrito, loop de validación completo. **El AUC de esa corrida no se interpreta como señal**, la muestra es demasiado chica para significar algo.

### Las arenas de evaluación (enmienda a Fase 3, decisión 4)

Fase 3 pedía "evaluar solo contra etiqueta fuerte (serología)". No es computable así como está escrito: SaMi-Trop, la única fuente con etiqueta fuerte, es 100% positiva (1.545 de 1.545) — sin un grupo sano para comparar no hay AUC, AUPRC, especificidad ni PPV, solo recall. El go/no-go (AUC ≥0,93 / 0,85-0,93 / <0,85, Fase 3) y el punto de operación (95% sensibilidad, banda PPV≥30%) sí necesitan las dos clases. Se resuelve con **arenas separadas, nunca una métrica agregada entre datasets**:

- **Arena A — CODE-15% sola (decide el go/no-go).** Dos clases, una sola fuente ⇒ el atajo de fuente es imposible por construcción dentro de la arena. Prevalencia a nivel paciente 1,91% (val) / 1,92% (test) — coincide con el 1,90% que asume toda la tabla de operación de Fase 3, lo que hace interpretables el PPV y las "serologías por caso". AUC-ROC, AUPRC, sens/espec/PPV en los dos umbrales.
- **Arena B — SaMi-Trop (solo recall).** De los pacientes confirmados por serología, cuántos se detectan en el umbral calibrado en A. No se reporta AUC/especificidad/PPV: no hay negativos.
- **Arena C — PTB-XL (solo especificidad).** Población no endémica, otro equipo/década — mide robustez cruzada de población, no discriminación.
- **Arena D — serológica ampliada, SaMi-Trop(+) vs. CODE-15%(−).** La única forma de obtener un AUC con positivos confirmados por serología. Válida porque esas dos fuentes tienen baja separabilidad entre sí (AUC 0,598 medido arriba) — si esa sonda de fuente diera alto, esta arena se reportaría como cota superior, no como resultado.
- **Diagnóstico de atajo (no es performance)**: AUC pooled entre todos los datasets menos AUC de arena A. Ese delta es la magnitud del atajo de fuente en las unidades del go/no-go.

Umbrales de operación (bajo = 95% sensibilidad, alto = PPV≥30%) se calibran **siempre en validación, arena A, nunca en test**.

### Implementación (2026-08-12)

Cuatro módulos nuevos, todos corriendo sobre la salida de Fase 2 sin tocarla:

- **`src/dataset.py`** — `ECGDataset` de PyTorch sobre `fase2_preprocessed.hdf5`, más el merge del label de RBBB y los pesos por confianza. Excluye PTB-XL del train por default (`--con-ptbxl` para la ablación inversa).
- **`src/model.py`** — `ResNet1D`, CNN 1D residual con dos cabezas (Chagas + RBBB) sobre el mismo cuerpo.
- **`src/evaluar.py`** — las 4 arenas + calibración de umbrales, a nivel paciente. Separado de `train.py` a propósito: la Fase 5 lo reusa tal cual.
- **`src/train.py`** — loop de entrenamiento con loss enmascarada, `pos_weight`, AMP, checkpoints.

**Bug encontrado antes de que llegara al modelo: los `record_id` no son únicos entre datasets.** El primer merge del label de RBBB se hizo por `record_id` contra `code15/exams.csv`, y le pegó label de RBBB ajeno a **3.100 registros de PTB-XL** — el `record_id` de PTB-XL es su `ecg_id` (1..21.837) y esos mismos números existen como `exam_id` en CODE-15%. Nada falla cuando pasa: el registro queda con máscara 1 y un label inventado, y encima justo en la arena C, que es la que mide generalización. La clave del merge ahora lleva `dataset` además de `record_id`. Vale como advertencia general: **`record_id` solo es único dentro de su dataset** (ya está documentado así en `build_metadata.py`, pero es fácil de olvidar al hacer un join).

**La arquitectura, y por qué no es exactamente la del paper.** Se calca ResNet1D de Ribeiro et al. 2020 (*Nature Communications*), que es la red entrenada sobre CODE — el dataset de donde sale el 93% de nuestro train: stem convolucional + 4 bloques residuales que dividen el largo por 4 y suben canales (128, 196, 256, 320), flatten, cabeza lineal. ~6,4M de parámetros.

La diferencia forzada es el largo de entrada: **2.800 muestras, no 4.096**, porque esa es la ventana de 7,0 s de la Fase 2. Como 2.800 no es múltiplo de 4⁴, los largos por bloque quedan `2800 → 700 → 175 → 43 → 10`, y ahí la rama principal y el atajo residual dejan de coincidir salvo que redondeen igual. Medido: con el **kernel 17 del paper** en la convolución con stride, el tercer bloque da **44 contra 43** del atajo y el forward revienta. Con **kernel 16 / padding 6** la convolución da exactamente `floor(L/4)`, el mismo largo que `MaxPool1d(4)`, y coincide en los cuatro bloques. No es un detalle cosmético: es la razón por la que no se puede copiar el paper y pegarlo.

**Ponderación por confianza: `peso_strong = 3,0`.** La decisión 4 de Fase 3 pide entrenar con las tres fuentes ponderadas, sin fijar cuánto. El primer instinto —dejarlo en 1,0 porque SaMi-Trop es apenas 0,45% de los registros de train y subirle el peso equivaldría a sobremuestrearlo— **está mal planteado, y medirlo lo muestra**: como *todos* los registros de SaMi-Trop son positivos, `pos_weight` ya le corrige el submuestreo, y con peso 1,0 ya se lleva el **19,1%** del gradiente de la clase positiva. El eje que `peso_strong` mueve es otro distinto: cuánto vale una etiqueta serológica contra una autorreportada.

Medido sobre el train real (238.027 registros: 4.574 positivos autorreportados + 1.083 serológicos):

| peso_strong | pos_weight | masa+ de SaMi-Trop | exposición efectiva |
|---|---|---|---|
| 1,0 | 41,1 | 19,1% | 41× |
| 2,0 | 34,5 | 32,1% | 69× |
| **3,0** | **29,7** | **41,5%** | **89×** |
| 5,0 | 23,3 | 54,2% | 116× |
| 20,0 | 8,9 | 82,6% | 177× |

("masa+" = qué fracción del gradiente de la clase positiva aportan los 1.083 registros de SaMi-Trop; "exposición efectiva" = cuánto pesa uno de ellos contra un negativo de CODE-15%.)

**Se elige 3,0**: deja a la serología como co-protagonista de la señal positiva sin dominarla. De 5,0 para arriba, 1.083 grabaciones aportan más de la mitad de todo lo que el modelo aprende sobre "positivo", y *eso* sí es memorizar la cohorte — el riesgo por el que se descartó el oversampling. **Efecto lateral a tener presente:** subir `peso_strong` baja `pos_weight` (41 → 29,7), o sea que de paso les baja el peso a los positivos de CODE-15%.

Es la **primera ablación a correr** (`--peso-strong`), y es barata de resolver empíricamente porque la arena A es CODE-15% sola: es inmune a este cambio y hace de juez honesto.

**Qué se elige como mejor checkpoint: AUPRC de arena A en validación, no la loss.** La loss mezcla las dos tareas y está dominada por el desbalance; la arena A es la que decide el go/no-go.

### Portabilidad a Linux (2026-08-12)

El armado del dataset (Fases 0-2) corre en la notebook con Windows, pero **el entrenamiento corre en la máquina con la RTX 4090, que tiene Ubuntu**, con el mismo SSD enchufado. Tres cosas que había que arreglar para eso, ninguna visible desde Windows:

- **Autodetección del SSD.** `config.py` buscaba la carpeta de datos solo en la raíz de cada punto de montaje (`/media/DECA-datasets`), pero Ubuntu automonta los medios externos en `/media/<usuario>/<etiqueta>/`, o sea dos niveles más abajo. Ahora la búsqueda baja hasta 2 niveles bajo cada raíz (`/media`, `/run/media`, `/mnt`, `/Volumes`, `/data`, `$HOME`) y en cada candidata prueba las dos formas: que el montaje *contenga* la carpeta y que el montaje *sea* la carpeta. Verificado sobre 7 layouts (automount de Ubuntu, de Fedora/Arch, montaje manual, copia en el home, minúsculas, y uno demasiado profundo que debe fallar): 7/7.
- **Bloqueo de archivos de HDF5.** En Linux, HDF5 pide un lock al abrir y sobre exFAT —que es como está formateado el SSD— eso falla con `unable to lock file` aun en solo lectura y con un solo proceso. En Windows no pasa, así que el síntoma habría aparecido recién al mover el entrenamiento. El `ECGDataset` abre con `locking=False`, que es seguro porque nadie escribe el archivo mientras se entrena.
- **La rueda de torch con CUDA no viene de PyPI.** `pip install torch` a secas instala la versión sin CUDA. Hay que usar el índice de PyTorch (ver `requirements.txt`).

No hace falta convertir nada del armado del dataset: con el SSD conectado, `train.py` lee `code15/exams.csv` para el label de RBBB igual que acá.

Se agregó [SETUP.md](SETUP.md): instructivo de puesta en marcha desde cero para una PC nueva, con los dos escenarios (solo entrenar / rehacer el dataset), Windows y Linux, y una sección de problemas conocidos con el síntoma exacto y el comando que lo arregla.

### Corrida de humo, primer bug post-implementación, y primera corrida real (2026-08-12)

**Corrida de humo (`--limit-train 500 --epocas 1`): pasó.** Cargó datos, corrió forward/backward sin NaN, calibró umbrales sobre las 4 arenas sin explotar (incluido el caso límite de banda alta sin umbral alcanzable, que devuelve PPV `nan` en vez de romper), y escribió el checkpoint. Confirmó además, de paso, que `num_workers>0` no cuelga en Windows con este dataset.

**Bug real, chico, atrapado por el propio warning de PyTorch.** La corrida de humo tiró:
```
UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
```
En `entrenar_epoca()` (`src/train.py`), `suma_chagas += float(l_chagas)` convertía a escalar un tensor que todavía requería gradiente, después de `backward()` pero sin haberlo desprendido del grafo explícitamente. No cambiaba el resultado numérico, pero es la clase de descuido que en un loop de miles de batches puede retener memoria de más sin motivo. Arreglado con `.detach().item()` en las dos acumulaciones (chagas y RBBB); la corrida de humo repetida después del fix salió limpia, sin warnings.

**Benchmark de throughput, antes de comprometerse a un número de épocas.** Sobre la RTX 3500 Ada Generation (laptop, 12 GB): con `batch=128` el train converge a **~7,4-8 it/s** en régimen estable (los primeros 1-2 batches tardan 15-20s por el warmup de cuDNN/CUDA, después cae a <150ms/batch). Con 1.860 batches de train (238.027 registros) y 429 de val (54.846), eso da **una época completa del dataset entero en ~5 minutos** — mucho más rápido de lo previsto en el plan original (~2h la corrida completa de 30 épocas se estimaba antes de medir; a este ritmo son ~2,5h para 30 épocas, pero cada época individual ya es barata de inspeccionar).

**Primera corrida real: 1 época, dataset completo (238.027 train / 54.846 val), batch 128, 5,2 min:**

| métrica | valor |
|---|---|
| loss chagas / rbbb | 1,2383 / 0,3604 |
| **Arena A (code15)** — AUC / AUPRC | **0,8023 / 0,1362** |
| banda media (sens 95%) | especificidad 32,0%, PPV 2,7%, deriva 68,6% |
| banda alta (PPV≥30%) | sensibilidad 9,3%, PPV 30,2%, deriva 0,6% |
| Arena B (samitrop) — recall | 75,2% (umbral bajo) / 4,3% (umbral alto) |
| Arena C (ptbxl) — especificidad | 21,8% (umbral bajo) / 99,4% (umbral alto) |
| Arena D (serológica ampliada) — AUC / AUPRC | 0,6385 / 0,0342 |
| cabeza RBBB — AUPRC | 0,7516 |
| diagnóstico de atajo (delta pooled vs. A) | **−0,0450** |

**Lectura, con la misma cautela con la que se lee cualquier resultado de época 1:**

- **AUC 0,80 en arena A tras una sola época es una señal fuerte de que el modelo está aprendiendo algo real**, no ruido — muy por debajo del umbral de go/no-go (0,93 fijado en Fase 3), pero el punto de comparación correcto es la trayectoria en las próximas épocas, no el criterio final todavía.
- **El delta del atajo de fuente dio negativo (−0,045)**: el AUC agrupando los 3 datasets es *menor* que el de arena A sola. Es la dirección correcta — si el modelo estuviera usando la pista de origen (ver "Tercera pista de fuente" más arriba) para inflar la métrica, el pooled saldría *más alto* que A, no más bajo. Primera evidencia de que la exclusión de PTB-XL del train y el resto del diseño anti-atajo están funcionando, aunque con una sola época no alcanza para descartarlo del todo.
- **La cabeza de RBBB (AUPRC 0,75) ya rinde muy por encima de la de Chagas.** Coherente con lo esperado: 9.457 positivos densos y bien anotados contra ~2% de prevalencia de Chagas, y RBBB es morfológicamente más directo de leer que "Chagas" en general (que es una inferencia indirecta, per Fase 3).
- **Arena C (PTB-XL, nunca vista en entrenamiento) da 99,4% de especificidad en la banda alta** desde la primera época — buena señal temprana de generalización cruzada de población, que es justamente lo que esta arena existe para medir.
- Las bandas de operación (68,6% de derivación en la media, 9,3% de sensibilidad en la alta) todavía están lejos de ser útiles clínicamente — esperable con 1 época; los umbrales se recalibran solos en cada época a medida que la discriminación mejora.

**Pendiente inmediato:** correr más épocas (estimado ~5 min/época a este throughput) para ver si arena A converge cerca del umbral de despliegue, y correr la ablación de `--peso-strong` (1,0 vs. 3,0 vs. 5,0) una vez que haya una corrida de varias épocas de referencia.

### Corrida de 8 épocas, cortada en la 6: overfitting detectado a mano (2026-08-13)

Misma configuración (dataset completo, batch 128, `peso_strong=3,0`), pensada para 8 épocas. Se fue siguiendo época por época en vivo:

| | ép. 1 | ép. 2 | ép. 3 | ép. 4 | ép. 5 | ép. 6 |
|---|---|---|---|---|---|---|
| loss chagas (train) | 1,238 | 1,151 | 1,087 | 1,011 | 0,980 | **0,981** |
| Arena A AUC | 0,8023 | 0,7970 | 0,8238 | 0,8397 | 0,8338 | 0,8135 |
| Arena A AUPRC | 0,1362 | 0,1279 | 0,1328 | **0,1500** | 0,1369 | 0,1152 |
| Arena B recall (bajo) | 75,2% | 83,5% | 89,1% | 88,7% | 92,2% | 93,0% |
| Arena D AUC | 0,6385 | 0,7230 | 0,7995 | 0,8061 | 0,8445 | 0,8405 |
| atajo (`delta vs A`) | −0,045 | −0,024 | −0,009 | −0,011 | −0,001 | **+0,0059** |
| RBBB AUPRC | 0,7516 | 0,7962 | 0,7915 | 0,7818 | 0,8032 | 0,7793 |

**En la época 6 coincidieron los tres indicios que se venían vigilando desde la 5, en la misma época:** la loss de train dejó de bajar por primera vez en toda la corrida (0,980 → 0,981), el AUPRC de arena A tocó su mínimo (0,1152, el peor de las 6 épocas), y el diagnóstico de atajo **cruzó de negativo a positivo** (+0,0059) por primera vez — señal de que el modelo empezó a apoyarse un poco en la pista de origen del dato en vez de en la enfermedad. Individualmente cualquiera de los tres podría ser ruido (arena B/D se miden sobre solo 230 positivos de SaMi-Trop y 2.830 de PTB-XL en val, chico y ruidoso), pero los tres juntos en la misma época es la combinación acordada de antemano para frenar y mirar.

**Se cortó la corrida ahí** (proceso interrumpido a mano, no llegó a completar la época 7). No se perdió nada de valor: el mejor checkpoint quedó fijado en la **época 4** (AUC 0,8397, AUPRC 0,1500, atajo −0,011 — el mejor resultado de las 6 épocas en las tres dimensiones a la vez) y ese es el que `mejor.pt` tenía guardado. Se copió a `models/real8ep/mejor.pt` en el repo (fuera de `.gitignore`, ~26 MB) para no depender de que el SSD esté siempre conectado para tener el mejor modelo a mano; el resto de los artefactos de la corrida (`historia.json`, `args.json`, `ultimo.pt` — que ni llegó a escribirse, se escribe solo al completar todas las épocas) se quedan en el SSD, no en git.

**Nota sobre `ultimo.pt`:** al cortar a mano antes de que el loop termine, ese archivo nunca se llega a escribir (se guarda una sola vez, después de la última época). Es una limitación real: si se corta la corrida, la única forma de recuperar algo es `mejor.pt`, y si nunca hubo un "nuevo mejor" antes de cortar, no queda nada guardado. Es la motivación directa de `--resume` (más abajo).

### Tres mejoras a `train.py`, motivadas directamente por esta corrida (2026-08-13)

**1. Parada temprana automática (`--paciencia-atajo`, default 2).** Codifica el criterio que se acaba de aplicar a mano: cuenta épocas *seguidas* donde la época no fue un nuevo mejor **y** el atajo de fuente dio positivo; al llegar a 2, corta sola. El default es 2 y no 1 a propósito — lo que se hizo a mano fue reaccionar al primer cruce, pero arena B/C son chicas y un cruce aislado por ruido no debería tirar abajo toda una corrida de 30 épocas; pedir 2 seguidas es el punto medio entre reaccionar rápido y no ser hipersensible a una sola época ruidosa. En 0 se desactiva.

**2. `--resume RUTA.pt`.** Hasta ahora cortar a mano tiraba todo el progreso: no había forma de seguir entrenando desde un checkpoint, solo volver a arrancar de cero. Ahora los checkpoints (`mejor.pt`/`ultimo.pt`) guardan también el estado del optimizador y del scheduler, no solo los pesos del modelo, y `--resume` los carga y sigue desde `epoca_del_checkpoint + 1`. **Limitación conocida, aceptada por tiempo:** si se resume desde un checkpoint grabado antes de la última época que llegó a correr (el caso típico de cortar a mano unas épocas después del último "nuevo mejor" — exactamente lo que pasó acá, cortado en la 6 con el mejor en la 4), las épocas intermedias ya vistas se vuelven a correr. No es grave — cuesta unos minutos repitiendo trabajo, no corrompe nada — pero es una asimetría a tener presente: `--resume` retoma desde el **mejor** checkpoint, no desde el último intento.

**3. `historia.json` se extiende en vez de pisarse** si ya existe en la carpeta de la corrida (relevante al resumir con el mismo `--nombre`): antes se sobreescribía completo en cada corrida, así que resumir hubiera borrado el registro de las épocas ya hechas.

Verificado con corridas de humo antes de usar: `--resume` cargó correctamente el checkpoint de la época 4 de esta corrida (`AUPRC arena A 0.1500 -> sigue desde epoca 5`) y arrancó del punto correcto.

### Ablación `--peso-strong 1,0` vs. `3,0`: el atajo NO depende del peso de SaMi-Trop (2026-08-13)

Hipótesis a probar: que `peso_strong=3,0` (que le da a los 1.083 registros de SaMi-Trop ~89× de exposición efectiva, ver más arriba) fuera lo que empujaba al modelo a apoyarse en rasgos específicos de esa cohorte en vez de en la enfermedad en general — coherente en su momento con que arena B (recall SaMi-Trop) siguiera subiendo fuerte (75%→93%) en las mismas épocas en que arena A se estancaba y el atajo cruzaba a positivo. Se corrió `--peso-strong 1.0`, mismo resto de hiperparámetros, dataset completo, 8 épocas, con la parada temprana (`--paciencia-atajo`, ver abajo) ya activada por default.

| | ép. 1 | ép. 2 | ép. 3 | ép. 4 | ép. 5 | ép. 6 | ép. 7 | ép. 8 |
|---|---|---|---|---|---|---|---|---|
| Arena A AUPRC | 0,1374 | 0,1310 | 0,1688 | **0,1755** | 0,1538 | 0,1525 | 0,1490 | 0,1615 |
| atajo (`delta vs A`) | −0,051 | −0,016 | −0,015 | −0,002 | −0,004 | **+0,0021** | −0,002 | −0,008 |

**El cruce a positivo pasó en la época 6 en las dos corridas** (+0,0059 con peso 3,0; +0,0021 con peso 1,0) — casi el mismo punto exacto, con las dos corridas siendo completamente independientes salvo por ese único hiperparámetro. **Esto descarta la hipótesis**: el atajo no es un efecto del peso de SaMi-Trop, aparece en algún punto del entrenamiento independientemente de cuánto se pondere la etiqueta serológica. Sigue sin identificarse la causa real (candidatos para más adelante: el punto en que el learning rate todavía no bajó por el scheduler, o simplemente que el modelo cruza cierto umbral de capacidad donde empieza a ser rentable usar la pista espectral de PTB-XL descripta más arriba). Con `peso_strong=1,0` el cruce fue una sola época aislada (no dos seguidas), así que la parada temprana no se disparó y la corrida completó las 8 épocas.

**Resultado neto: `peso_strong=1,0` ganó.** Mejor checkpoint de las dos corridas, misma época (4) en ambas:

| | `peso_strong=3,0` | **`peso_strong=1,0`** |
|---|---|---|
| Arena A AUPRC (mejor) | 0,1500 | **0,1755** |
| Arena A AUC (esa época) | 0,8397 | 0,8378 |
| atajo (esa época) | −0,011 | −0,002 |

El checkpoint de esta ablación (`models/abl-peso1/mejor.pt` en el repo) reemplaza al de `real8ep` como mejor resultado de referencia.

### Tercer punto del barrido — `peso_strong=5,0` (2026-08-13): confirma la tendencia y triangula el cruce

Mismos hiperparámetros, misma parada temprana. Resultado:

| `peso_strong` | mejor AUPRC arena A | época del mejor |
|---|---|---|
| **1,0** | **0,1755** | 4 |
| 3,0 | 0,1500 | 4 |
| 5,0 | 0,1460 | 7 |

**El barrido queda cerrado con `peso_strong=1,0` como ganador claro**, con una tendencia monótona en los tres puntos: a más peso a la etiqueta serológica, peor arena A. Esto revisa la decisión original de la sección "Implementación": la ponderación por confianza que pide la Fase 3 (decisión 4) **no ayuda en este dataset** — `pos_weight` ya corrige el submuestreo de SaMi-Trop por sí solo (documentado más arriba), y pedirle un peso adicional lo hace sobreexpresarse en el gradiente sin mejorar la métrica que importa. **Nuevo default recomendado: `peso_strong=1,0`.**

**El cruce del atajo en la época 6 se repitió por tercera vez, en las tres corridas independientes:**

| `peso_strong` | atajo en época 6 |
|---|---|
| 1,0 | +0,0021 |
| 3,0 | +0,0059 |
| **5,0** | **+0,0077** |

Con tres corridas independientes cruzando en el mismo punto exacto, ya no es plausible que sea ruido — es un fenómeno estructural del entrenamiento en esa época, no un efecto de `peso_strong` (que ya se había descartado como causa del *momento* del cruce). Lo que sí correlaciona con `peso_strong` es la **magnitud** del cruce (0,002 → 0,006 → 0,008, también monótona) — más peso a SaMi-Trop no cambia cuándo pasa, pero sí lo agrava un poco cuando pasa. En las tres corridas el cruce fue transitorio (1-2 épocas) y la parada temprana (que pide 2 seguidas) no se disparó en ninguna; con `peso_strong=5,0` incluso se recuperó a un nuevo mejor checkpoint en la época 7 inmediatamente después.

**Pendiente:** investigar la causa real del cruce en la época 6 (candidatos: el punto en que el LR scheduler todavía no actuó — `patience=3` de `ReduceLROnPlateau` significa que recién podría bajar el LR después de 3 épocas sin mejora, así que la 6 cae justo en esa ventana; o que el modelo cruza un umbral de capacidad donde empieza a ser rentable usar la pista espectral de PTB-XL) antes de escalar a una corrida de 30 épocas con `peso_strong=1,0`.

### El patrón real no es "época 6", es "2 épocas seguidas sin mejorar" — y las tres corridas no son tan independientes como parecía (2026-08-13)

Antes de gastar más GPU, se reanalizaron los `historia.json` de las tres corridas ya hechas (sin entrenar nada nuevo). Resultado, alineando por "épocas sin mejorar" en vez de por número de época:

```
              épocas sin mejorar (por época):     atajo delta en esas épocas
peso 1,0:     0,1,0,0,1,2,3,4                     ...,-0.004,+0.0021,-0.0024,-0.0075
peso 3,0:     0,1,2,0,1,2,3,4                     ...,-0.001,+0.0059,-0.0002,-0.0019
peso 5,0:     0,1,2,0,1,2,0,1                     ...,-0.002,+0.0077,+0.0002(nuevo mejor)
```

**El cruce pasa siempre exactamente con 2 épocas seguidas sin mejorar**, no en la "época 6" en sí — coincidía con la época 6 en las tres corridas porque las tres tuvieron su último "nuevo mejor" en la época 4 antes de estancarse (4+2=6). Pero el patrón **no es monótono**: si fuera "cuanto más estancado, más se apoya en el atajo", tendría que seguir empeorando en las épocas 3 y 4 sin mejorar, y en cambio se recupera solo (vuelve a negativo, o directamente a un nuevo mejor checkpoint como en `peso_strong=5,0`). No encaja con una historia simple de "se estanca y busca la salida fácil".

**El problema metodológico, más importante que el patrón en sí:** las tres corridas usan **la misma semilla (`seed=42`)** — mismo peso inicial, mismo orden de datos. `peso_strong` solo reescala el gradiente de 1.083 registros de SaMi-Trop sobre 238.027 en total, así que es totalmente plausible que las tres trayectorias de optimización sean casi la misma para el grueso del modelo, y que "siempre cruza con 2 épocas sin mejorar" sea un artefacto de la semilla compartida — no una ley del entrenamiento. **No fueron tres corridas independientes como se venía asumiendo, y esa es la causa más probable de por qué el patrón se ve tan limpio.**

**Prueba pendiente, acordada pero no lanzada por falta de tiempo:** correr `peso_strong=1,0` (el mejor default hasta ahora) con una semilla distinta (`--seed` ≠ 42), 8 épocas, y ver si el cruce se sigue disparando a las 2 épocas sin mejorar (evidencia de que es real) o se mueve/desaparece (evidencia de que era ruido de la semilla compartida). **No se escala a una corrida de 30 épocas hasta tener esta respuesta** — decisión explícita del usuario, no se entrena "a ciegas" sin entender la causa.

**Oversampling (`--sampler balanceado`) evaluado como posible ruta y descartado para esta pregunta puntual.** No ataca la misma incógnita: PTB-XL está excluido del train siempre, y el atajo que se mide depende de cómo el modelo generaliza a PTB-XL en validación — oversampling solo cambia la mezcla de code15/SaMi-Trop *dentro* de train, sin razón directa para esperar que mueva el momento del cruce. Además tiene un riesgo conocido en la dirección contraria: repetir las 1.083 grabaciones de SaMi-Trop ~15-20× por época (contra verlas una sola vez con `pos_weight` reescalado) es la forma más directa de memorizar cualquier resto de pista de origen que haya sobrevivido al preprocesado de Fase 2, específica de esos archivos puntuales — podría empeorar el atajo, no mejorarlo. Sigue siendo una ablación válida para ver si mejora arena A en general, pero es una pregunta distinta a la del cruce, y no es la siguiente a correr.

### Dos utilidades agregadas a `train.py`, motivadas por la corrida cortada de la sección anterior (2026-08-13)

**1. Parada temprana automática (`--paciencia-atajo`, default 2).** Codifica el criterio que se aplicó a mano para cortar la corrida de `peso_strong=3,0` en la época 6: cuenta épocas *seguidas* donde la época no fue un nuevo mejor **y** el atajo de fuente dio positivo; al llegar a 2, corta sola. El default es 2 y no 1 a propósito — lo que se hizo a mano fue reaccionar al primer cruce, pero arena B/C son chicas y un cruce aislado por ruido no debería tirar abajo toda una corrida de 30 épocas (la ablación de `peso_strong=1,0` es la prueba: cruzó en la época 6, pero al ser una sola época aislada, no dos seguidas, la corrida siguió sola hasta el final sin intervención). En 0 se desactiva.

**2. `--resume RUTA.pt`.** Hasta esta corrida, cortar a mano tiraba todo el progreso: no había forma de seguir entrenando desde un checkpoint, solo volver a arrancar de cero (fue justo lo que pasó con la corrida de `peso_strong=3,0`, cortada en la época 6 sin `ultimo.pt` porque ese archivo solo se escribe al completar el loop entero). Ahora los checkpoints (`mejor.pt`/`ultimo.pt`) guardan también el estado del optimizador y del scheduler, no solo los pesos del modelo — por eso `mejor.pt` pasó de pesar ~26 MB a **~79 MB** (los dos buffers de momento de Adam pesan lo mismo que el modelo cada uno) — y `--resume` los carga y sigue desde `epoca_del_checkpoint + 1`. `historia.json` se extiende en vez de pisarse si ya existe en la carpeta de la corrida, para que resumir con el mismo `--nombre` no borre el registro de las épocas ya hechas. **Limitación conocida, aceptada por tiempo:** si se resume desde un checkpoint grabado antes de la última época que llegó a correr, las épocas intermedias ya vistas se vuelven a correr — no es grave, cuesta unos minutos repitiendo trabajo, pero es una asimetría a tener presente (`--resume` retoma desde el **mejor** checkpoint, no desde el último intento). Verificado con una corrida de humo antes de usarlo en la ablación real: cargó correctamente el checkpoint de la época 4 y arrancó de la 5.

### Corrida de control con otra semilla (`abl-peso1-seed123`, 2026-08-14): el streak es real, el cruce no es determinista

Prueba pendiente de la sección anterior: `peso_strong=1,0`, 8 épocas, `--seed 123` (las tres corridas previas usaban `seed=42`).

```
Epoca  AUPRC arena A   delta atajo   sin mejorar
1      0.1481          -0.0464       (nuevo mejor)
2      0.1435          -0.0027        1
3      0.1598          -0.0124       (nuevo mejor)
4      0.1515          -0.0036        1
5      0.1612          -0.0086       (nuevo mejor)
6      0.1318          -0.0097        1
7      0.1481          -0.0011        2   <- mismo punto donde cruzaba con seed=42
8      0.1654          -0.0052       (nuevo mejor)
```

Mejor AUPRC arena A: 0.1654 (vs. 0.1755 de la corrida original `peso_strong=1,0` con seed=42 — mismo orden de magnitud).

**Conclusión:** el streak de "2 épocas seguidas sin mejorar" se repite (época 7) — es una dinámica de entrenamiento real, no un artefacto de la semilla. Pero el cruce del atajo **no se repite**: llegó muy cerca de cero (-0.0011, el valor más chico de todo el streak) y se recuperó solo en la época siguiente con un nuevo mejor, sin cruzar a positivo. Con seed=42 cruzó las tres veces exactamente en ese punto; con seed=123, no.

Esto separa las dos partes de la hipótesis original: la coincidencia de fase ("se estanca cada ~2 épocas") es estructural y se explica por la dinámica de optimización en sí (plausible: el scheduler y la superficie de loss producen mesetas periódicas independientemente de la semilla). Pero si en esa meseta el modelo específicamente cae en el atajo de fuente es sensible a la inicialización — no hay una fuerza determinística empujándolo ahí, es más cerca de un roce que de una atracción. Con una sola semilla adicional no se puede descartar del todo (podría cruzar 1 de cada 2-3 semillas), pero baja bastante la urgencia: no parece ser el tipo de problema estructural que garantiza que una corrida larga termine explotando el atajo.

**Implicancia práctica:** con `--paciencia-atajo 2` activo por default, una corrida larga real está protegida igual — si cruza en una meseta de 2 épocas, corta sola; si no cruza (como esta corrida), sigue de largo. Con esto ya se puede considerar habilitada una corrida más larga (sujeto a decisión del usuario), aunque seguiría siendo valioso en algún momento correr una tercera semilla para tener más que 2 puntos de dato.

### Corrida larga de 30 épocas (`real30ep`, 2026-08-25): confirma la meseta, no supera al mejor modelo

Corrida pendiente habilitada por la sección anterior: dataset completo, `peso_strong=3,0` (default), 30 épocas, sin cortar por `--paciencia-atajo`. Corrida en la máquina de entrenamiento (RTX 4080 SUPER, no la 4090 mencionada en SETUP.md — batch ajustado a 128 por los 16 GB de VRAM en vez de 256). Los datos se copiaron del SSD externo al disco interno (`~/DECA-datasets`, ~47 GB: solo `fase2_preprocessed.hdf5`, `fase2_metadata.parquet`, `code15/exams.csv` y `modelos/` — los HDF5 crudos de code15 no hacen falta para entrenar, ya está todo fusionado en la fase 2) para poder desconectar el SSD sin cortar el entrenamiento.

```
| ep | loss chagas (train) | AUC arena A | AUPRC arena A | delta atajo | AUPRC RBBB |
|---|---|---|---|---|---|
| 1  | 1.2071 | 0.8285 | 0.1302 | -0.0578 | 0.7784 |
| 2  | 1.0782 | 0.8258 | 0.1378 | -0.0320 | 0.7946 |
| 3  | 1.0294 | 0.8349 | 0.1705 | -0.0157 | 0.7911 |  <- nuevo mejor (queda como mejor.pt)
| 4  | 1.0097 | 0.8398 | 0.1549 | -0.0086 | 0.7973 |
| 5  | 0.9905 | 0.8404 | 0.1433 | -0.0083 | 0.8015 |
| 6  | 0.9791 | 0.8357 | 0.1567 | +0.0009 | 0.7823 |  <- unico cruce del atajo, aislado, sin repetirse
| 7  | 0.9805 | 0.8410 | 0.1500 | -0.0019 | 0.7862 |
| 8  | 0.9108 | 0.8457 | 0.1645 | -0.0082 | 0.7919 |
| 9  | 0.8966 | 0.8465 | 0.1563 | -0.0093 | 0.7908 |  <- maximo AUC de toda la corrida
| 10 | 0.8901 | 0.8464 | 0.1673 | -0.0105 | 0.7889 |
| 11-30 | ~0.85-0.88 | 0.842-0.843 (estable) | 0.164-0.167 (estable) | -0.010 a -0.011 (estable) | 0.784-0.785 (estable) |
```

Historial completo por época en `~/DECA-datasets/modelos/real30ep/historia.json`.

**Lectura:**

- **La meseta que ya se veía en la corrida de 8 épocas se confirma con datos hasta la época 30**: AUC arena A converge a ~0,843 desde la época ~8 y no se mueve más en las 22 épocas restantes. No era falta de entrenamiento — era un techo real del modelo/features actuales.
- **Sin atajo de fuente sostenido**: el delta cruzó a positivo una sola vez (época 6, +0,0009, ínfimo) y no se repitió — coherente con lo que predecía la sección anterior sobre `--paciencia-atajo 2`, que no tuvo que activarse.
- **El mejor checkpoint (por AUPRC, criterio de selección de Fase 4) quedó en la época 3: AUPRC 0,1705, AUC 0,8349.** No es el máximo AUC de la corrida (0,8465 en la época 9) porque se selecciona por AUPRC, no por AUC.
- **No supera al mejor modelo hasta ahora.** Comparando AUPRC arena A entre todas las corridas guardadas en `~/DECA-datasets/modelos/`:

  | corrida | hiperparámetro relevante | AUPRC (mejor.pt) | AUC (misma época) | max AUC visto |
  |---|---|---|---|---|
  | **`abl-peso1`** | `peso_strong=1,0`, 8 ép. | **0,1755** | 0,8378 | 0,8428 |
  | `real30ep` | `peso_strong=3,0`, 30 ép. | 0,1705 | 0,8349 | 0,8465 |
  | `abl-peso1-seed123` | `peso_strong=1,0`, seed 123 | 0,1654 | 0,8395 | 0,8422 |
  | `real8ep` | `peso_strong=3,0`, 8 ép. cortada | 0,1500 | 0,8397 | 0,8402 |
  | `abl-peso5` | `peso_strong=5,0`, 8 ép. | 0,1460 | 0,8310 | 0,8413 |

  `abl-peso1` sigue siendo el mejor modelo global. 30 épocas con el `peso_strong` default (3,0) no superaron a 8 épocas con `peso_strong=1,0` — otra pista de que lo que mueve la aguja es ese hiperparámetro, no la cantidad de épocas.

- **Contra el criterio de go/no-go fijado en Fase 3 (AUC ≥0,93 despliegue completo / 0,85-0,93 solo priorizador / <0,85 no se usa): esta corrida da NO-GO.** Ni el checkpoint guardado (AUC 0,8349) ni el máximo AUC visto en toda la corrida (0,8465) llegan al piso de 0,85 — no califica ni para el uso limitado como priorizador.

**Pendiente:** correr la ablación `--sampler balanceado` (todavía no probada) y, dado que `peso_strong=1,0` viene ganando en las tres corridas que lo probaron, considerarlo el nuevo default candidato en vez de 3,0.

### Ablación `--sampler balanceado` (`real30ep-sampler`, 2026-08-26): overfitting rápido, pierde contra `abl-peso1`

Corrida pendiente de la sección anterior: dataset completo, `peso_strong=1,0` (default del código, `dataset.py:57`), 30 épocas, `--sampler balanceado`, `--batch 128`, sin `--paciencia-atajo` disparada. Con el sampler, `pos_weight` de Chagas se apaga (queda en 1,0) para no apilar los dos mecanismos de corrección de desbalance, tal como documenta `train.py`.

**Terminó en ~27 minutos — mucho más rápido que las ~2,5h de las corridas sin sampler**, pese a usar el mismo `num_samples=len(train)` (238.027) por época. La causa más probable: `WeightedRandomSampler` con reemplazo le da a cada clase la mitad de la masa de probabilidad, y con solo ~5.657 positivos totales (4.574 code15 + 1.083 samitrop) contra 233.453 negativos, la mitad "positiva" de cada batch sale una y otra vez del mismo pool chico — mejor localidad de cache de disco que el barrido aleatorio sobre el dataset completo que hacen las corridas sin sampler.

```
| ep | loss chagas (train) | AUC arena A | AUPRC arena A | delta atajo |
|---|---|---|---|---|
| 1  | 0.5362 | 0.8365 | 0.1626 | -0.0135 |  <- mejor.pt (mejor de toda la corrida)
| 2  | 0.4563 | 0.8196 | 0.1525 | -0.0047 |
| 3  | 0.3027 | 0.8166 | 0.1499 | -0.0036 |
| 4  | 0.1786 | 0.7976 | 0.1241 | +0.0009 |  <- unico cruce del atajo, aislado
| 5  | 0.1215 | 0.8106 | 0.1532 | -0.0027 |
| 6  | 0.0734 | 0.8074 | 0.1509 | -0.0005 |
| 10 | 0.0359 | 0.8086 | 0.1514 | -0.0005 |
| 20 | 0.0316 | 0.8093 | 0.1541 | -0.0005 |
| 30 | 0.0322 | 0.8092 | 0.1527 | -0.0007 |
```

Historial completo en `~/DECA-datasets/modelos/real30ep-sampler/historia.json`.

**Lectura:**

- **La loss de train se hunde a casi cero en ~10 épocas** (0,54 → 0,03) y se queda ahí planchada las 20 épocas restantes — la firma clásica de memorización que ya se había anotado como riesgo antes de correr esto: repetir un pool chico de positivos (~5.657 registros) muchas veces por época en vez de verlos una sola vez con `pos_weight`.
- **El mejor resultado de toda la corrida fue la época 1**, antes de que el overfitting pegara fuerte (AUC 0,8365, AUPRC 0,1626). Después de eso el modelo se estabiliza en un piso más bajo (AUC ~0,808-0,810, AUPRC ~0,153) y no vuelve a mejorar en 29 épocas.
- **Sin atajo de fuente sostenido** (un solo cruce aislado en la época 4, +0,0009, no se repite) — el oversampling no empeoró ese diagnóstico, pero tampoco lo mejoró.
- **Pierde contra el mejor modelo actual:**

  | corrida | AUPRC (mejor.pt) | AUC (misma época) |
  |---|---|---|
  | **`abl-peso1`** (peso_strong=1,0, sin sampler, 8 ép.) | **0,1755** | 0,8378 |
  | `real30ep` (peso_strong=3,0, sin sampler, 30 ép.) | 0,1705 | 0,8349 |
  | `real30ep-sampler` (peso_strong=1,0, con sampler, 30 ép.) | 0,1626 | 0,8365 |

**Conclusión: ninguna de las dos rutas para el desbalance de clases (subir `peso_strong`, oversampling) mueve el techo de AUC ~0,84 en arena A.** `abl-peso1` (peso_strong=1,0, sin oversampling) sigue siendo el mejor modelo de todas las corridas hechas hasta ahora, y sigue sin cruzar el piso de 0,85 de Fase 3. El cuello de botella no parece ser el manejo del desbalance — es capacidad del modelo o de las features, tema para la próxima sección de trabajo.

---

## Sesión del 2026-08-26 — seis hallazgos que reencuadran toda la Fase 4

Sesión de análisis, sin corridas nuevas más allá de la ablación de sampler de arriba. El disparador fue una pregunta simple ("¿por qué no mejora?") y terminó revirtiendo tres supuestos que veníamos arrastrando. Se documenta en orden de importancia, no cronológico.

### 1. El techo NO es del modelo: es de la etiqueta (el hallazgo central)

El paper de referencia de esta tarea exacta —[PLOS NTD 2023](https://journals.plos.org/plosntds/article?id=10.1371%2Fjournal.pntd.0011118), del mismo grupo que produjo CODE-15% y SaMi-Trop, usando **el mismo ResNet1D que nosotros copiamos**— reporta un experimento que nunca habíamos considerado: mismo modelo, mismos pesos, cambiando *únicamente qué cuenta como positivo*.

| positivos definidos como | REDS-II | ELSA-Brasil |
|---|---|---|
| todos los seropositivos | 0,68 | 0,59 |
| **solo cardiopatía chagásica crónica (CCC)** | **0,82** | **0,77** |

**+0,14 a +0,18 de AUC sin tocar una línea de código.** La explicación de los autores: "el modelo es capaz de detectar pacientes con CCC a partir del trazado con alta discriminación; para los pacientes sin CCC la discriminación es menor". La razón clínica es que la mayoría de los seropositivos nunca desarrolla compromiso cardíaco — su ECG es genuinamente normal. Son **positivos inaprendibles**: estamos penalizando al modelo por no ver algo que no está en la señal.

**Esto aplica de lleno a DECA y hay un desalineamiento en nuestro propio planteo.** El ROADMAP dice, desde el título, que el objetivo es detectar **cardiopatía chagásica**. Nuestras etiquetas (`chagas_label`) son serología/autorreporte, o sea **infección**. Entrenamos y evaluamos contra un blanco más ancho que el que declaramos querer, y la brecha entre ambos es exactamente donde se pierde el AUC.

**Evidencia propia, que ya teníamos y no habíamos leído así:** en todas nuestras corridas, misma red, mismo cuerpo compartido, mismos datos:

| cabeza | AUPRC |
|---|---|
| RBBB (patrón ECG, anotado desde la señal) | **0,76 – 0,80** |
| Chagas (serología/autorreporte) | 0,15 – 0,17 |

El modelo lee patología del ECG ~5× mejor que "esta persona tiene anticuerpos". Ese cociente es la medida de nuestro propio ruido de etiqueta, y estaba a la vista desde la primera corrida.

### 2. Estamos en el estado del arte, no debajo — el go/no-go de Fase 3 es inalcanzable

Nunca habíamos contrastado nuestros números contra la literatura. Al hacerlo:

| | AUC | fuente |
|---|---|---|
| PLOS NTD 2023 (mismo ResNet, CODE+SaMi-Trop) | **0,80** (IC 0,79-0,82) | validación |
| Moody Challenge 2025, 5° de 40 (equipo Ahus AIM) | **0,840** | cross-validation |
| **DECA, `abl-peso1`, arena A** | **0,838** | validación |
| DECA, máximo visto (`real30ep`, ép. 9) | 0,847 | validación |

**El piso de 0,93 fijado en Fase 3 para "despliegue completo" no lo alcanza nadie en el campo**, ni siquiera el grupo dueño de los datos. El de 0,85 para "solo priorizador" está justo en el borde de lo que logra el estado del arte mundial. Ambos se escribieron a ciegas (2026-08-08), antes de tener un solo número propio y sin contraste bibliográfico — decisión metodológicamente correcta en su momento (fijar el criterio antes de ver resultados evita moverlo a conveniencia), pero calibrada con información que no teníamos.

**Otros datos del challenge que reencuadran la evaluación:**
- **La métrica principal del Moody Challenge 2025 NO es AUC**: es TPR a capacidad fija de derivación (cuántos positivos reales capturás dentro de un cupo fijo de gente que podés mandar a serología). Es un primo directo de nuestras bandas de operación de Fase 3, no del AUC.
- Puntajes absolutos bajos en todo el campo: mejor equipo (Biomed-Cardio) 0,323 en test oculto; mediana de validación 0,279; el 5° puesto sacó 0,269.
- **La performance cae ~64% de validación a un test externo (ELSA-Brasil)** — el problema de generalización entre poblaciones es del campo entero, no de nuestro pipeline.

**Corolario sobre AUC vs. AUPRC.** Con prevalencia 1,9%, el AUC-ROC *maquilla* el desbalance: el eje FPR = FP/(FP+TN) tiene un TN gigantesco, así que se pueden acumular muchos falsos positivos sin que la curva se mueva. El AUPRC sí se mueve con cada falso positivo, que es el costo real (una serología de más). Ya elegíamos checkpoints por AUPRC (ver "Qué se elige como mejor checkpoint"), pero el go/no-go sigue escrito en AUC — **inconsistencia a resolver al revisar Fase 3**.

### 3. El scheduler de LR venía apagando el entrenamiento solo

`train.py` tenía `ReduceLROnPlateau(mode="max", factor=0.1, patience=3)` **hardcodeado**, y hace `step()` sobre el AUPRC de arena A. Con `patience=3`, tras 4 épocas seguidas sin un nuevo mejor el LR se corta ×10, y puede volver a cortarse cada 4 épocas.

Ahora crucemos eso con el patrón real de las corridas: **el mejor checkpoint siempre cae muy temprano** (época 1, 3 o 4 según la corrida) y después no hay ningún nuevo mejor en las 26-29 épocas restantes. O sea que el scheduler estuvo recortando el LR casi desde el principio: en `real30ep` (mejor en época 3), entre la 7 y la 30 hay margen para ~5-6 reducciones → **LR de 1e-3 a ~1e-9**.

**La huella que lo confirma está en los propios datos de `real30ep`:** de la época 11 a la 30, AUC arena A queda en 0,842-0,843 y AUPRC en 0,164-0,167 — *valores prácticamente idénticos época tras época*. Un modelo que entrena fluctúa; uno con LR ≈ 0 devuelve el mismo número. **Las últimas ~20 épocas de cada corrida larga no estaban entrenando nada.**

Esto reencuadra la conclusión de la sección anterior: donde decíamos "es un techo real del modelo/features", parte de lo que veíamos era **un techo autoinfligido**. No invalida el hallazgo de que ni `peso_strong` ni oversampling mueven la aguja (esas comparaciones son entre sí, con el mismo scheduler), pero sí invalida la lectura de "30 épocas confirman que no hay más para sacar".

**Acción tomada:** se expusieron `--paciencia-lr` (default 3, retrocompatible) y `--factor-lr` (default 0,1) como flags en `train.py`. Se lanzó la corrida `real30ep-paciencialr8` (`peso_strong=1,0`, sin sampler, `--paciencia-lr 8`, 30 épocas) — **quedó corriendo y no se pudo recuperar el resultado**: la máquina de entrenamiento se volvió inalcanzable por red antes de terminar. Verificar `~/DECA-datasets/modelos/real30ep-paciencialr8/historia.json` cuando vuelva.

### 4. Sumar síntomas al target lo empeora — medido, no razonado

Propuesta evaluada: en vez de predecir serología, predecir **los patrones que genera el Chagas** (la idea es que el detector sea de la cardiopatía, no de la infección). Medido sobre los 343.424 registros de CODE-15% con label de Chagas:

| target | prevalencia | P(Chagas \| target) | lift sobre base 1,91% | recall de Chagas+ |
|---|---|---|---|---|
| **RBBB solo** | 2,77% | **13,79%** | **7,2×** | 19,97% |
| unión de anormalidades enriquecidas (RBBB/AF/1dAVb/LBBB/SB) | 8,87% | 6,66% | 3,5× | 30,91% |
| "ECG no normal" (`normal_ecg` invertido) | 60,87% | 2,63% | 1,4× | 83,75% |

**Cada anormalidad extra sube el recall pero diluye la especificidad más rápido.** AF, LBBB, SB y 1dAVb rondan el 2% de prevalencia en gente sin Chagas, así que la unión se llena de no-Chagas: RBBB solo (enriquecimiento 8,22×) es mejor discriminador que RBBB combinado con las otras cinco. **La versión naive de "detectar los síntomas" es medible y es peor que lo que ya tenemos.**

**Corrección probabilística importante para cómo se presenta el producto:** "quien tenga el patrón probablemente tenga Chagas" **es falso** en población general. P(Chagas | RBBB) = 13,79%, o sea que ~86% de los RBBB no son Chagas (tiene mil causas: isquemia, hipertensión, edad, EPOC). Lo que sí vale, y mucho: pasar de 1,91% a 13,79% es **7,2× de enriquecimiento**, exactamente lo que necesita un embudo que deriva a un test serológico gratuito. Y en el norte argentino endémico ese multiplicador se aplica sobre una prevalencia base más alta — es la "integración con datos epidemiológicos" que el paper de PLOS recomienda explícitamente como trabajo futuro.

**Dato lateral encontrado en el mismo chequeo:** `code15/exams.csv` trae además `age` e `is_male` (y `nn_predicted_age`, `death`, `timey`). **Edad y sexo están disponibles y no los usamos como entrada del modelo.** El paper de PLOS tampoco los usó (solo estratificó post-hoc) — es una palanca sin explorar en el campo. Cautela: las distribuciones etarias difieren por dataset, así que es candidato a atajo de fuente y habría que vigilarlo con el mismo diagnóstico de siempre.

### 5. PTB-XL tiene los 3 patrones objetivo del ROADMAP como etiqueta — y lo estamos descartando

**Corrección de un error propio cometido en esta misma sesión:** se afirmó que HBAI, extrasístoles ventriculares y zonas eléctricamente inactivas "no existen como label en ningún dataset que tengamos". **Es falso.** Se miraron las 7 columnas de CODE-15% y no el vocabulario SCP de PTB-XL, que trae **71 códigos** con anotación clínica validada.

Medido sobre los 21.799 registros de `ptbxl_database.csv`:

| patrón objetivo del ROADMAP | código SCP | registros | % |
|---|---|---|---|
| **BRD + HBAI** (el patrón clásico de Chagas) | `CRBBB`/`IRBBB` **+** `LAFB` | **284** | 1,30% |
| HBAI solo — *la mitad que nos faltaba* | `LAFB` | 1.623 | 7,45% |
| BRD (completo o incompleto) | `CRBBB`/`IRBBB` | 1.658 | 7,61% |
| **Extrasístoles ventriculares** | `PVC`/`BIGU`/`TRIGU`/`PRC(S)` | 1.205 | 5,53% |
| **Aneurisma ventricular** | `ANEUR` ("ST-T changes compatible with ventricular aneurysm") | 104 | 0,48% |
| Zona eléctricamente inactiva (ampliada) | `ANEUR`/`QWAVE`/infartos localizados | 5.400 | 24,77% |
| **cualquiera de los 3 patrones** | | **6.242** | **28,63%** |

**Los tres patrones que el ROADMAP nombra como objetivo clínico están disponibles como ground truth, en el dataset que excluimos del entrenamiento por completo** (ver "Tercera pista de fuente", 2026-08-12). PTB-XL está hoy relegado a arena C.

Que sean alemanes y no chagásicos **no importa para este uso**: un BRD+HBAI es un BRD+HBAI sea por Chagas o por isquemia. Sirve como etiqueta *del patrón ECG*, no como label de Chagas — que es justamente lo que falta.

**Tensión a resolver antes de usarlo:** meter PTB-XL al entrenamiento reabre el riesgo de atajo de fuente que motivó excluirlo. La mitigación plausible es que alimente **solo las cabezas de patrón**, con la de Chagas enmascarada (mismo mecanismo que ya usa RBBB), de modo que la cabeza de Chagas nunca reciba gradiente que diga "PTB-XL → negativo". El riesgo residual es que el cuerpo compartido codifique el origen igual. **No está resuelto — se decide con el diagnóstico de atajo midiendo, no a priori.**

**Otro dato de `ptbxl_database.csv` sin explotar:** trae anotaciones de calidad de señal por registro (`baseline_drift`, `static_noise`, `burst_noise`, `electrodes_problems`, `extra_beats`, `pacemaker`) más `device` y `site`. Sirve para filtrado de calidad y, potencialmente, para atacar de frente la pista espectral de fuente (`device` permitiría medir cuánto de la separabilidad es equipamiento).

### 6. Datasets externos: qué conviene y qué no existe

**[PhysioNet/CinC Challenge 2021](https://physionet.org/content/challenge-2021/1.0.3/) — el recomendado.** ~88.000 ECG públicos, 30 clases SNOMED puntuadas, **12,6 GB**, WFDB (`.mat`+`.hea`) con el diagnóstico en el campo `#Dx:` del header, licencia CC-BY 4.0 **abierta** (sin credencial).

El argumento a favor no es solo volumen: **ataca el atajo de fuente por construcción.** Hoy la fuente es proxy de la etiqueta (SaMi-Trop=100% positivo, PTB-XL=100% negativo). Con 6 fuentes más, todas enmascaradas para la cabeza de Chagas pero ricas en etiquetas de patrón, "de qué dataset viene esto" deja de predecir la etiqueta.

**Inventario real de lo que agrega** (fuentes públicas: CPSC 6.877, CPSC-Extra 3.453, INCART 74, PTB/PTB-XL 21.837, Georgia 10.344, Chapman-Shaoxing 10.247, Ningbo 34.905). Como **PTB-XL ya lo tenemos**, el aporte neto es ~66.000 registros; descontándolo, en nuestros patrones objetivo:

| patrón | clase | total | ya tenemos | **nuevo** |
|---|---|---|---|---|
| BRD (todas las variantes) | `CRBBB`/`IRBBB`/`RBBB` | 6.687 | 1.660 | **~5.027** |
| **HBAI** — *el label escaso* | `LAnFB` | 2.186 | 1.626 | **560** |
| Extrasístoles ventriculares | `PVC`/`VPB` | 1.938 | 0 en este mapeo | **1.938** |
| Onda Q anormal | `QAb` | 2.076 | 548 | **1.528** |
| Mala progresión de onda R | `PRWP` | 638 | 0 | **638** (patrón nuevo) |

**El cuello de botella sigue siendo HBAI**: apenas +560, y el patrón clásico requiere BRD **+** HBAI simultáneos (en PTB-XL esa coocurrencia son 284 registros). Descargar Challenge 2021 no resuelve la escasez del patrón más específico, aunque sí multiplica todo lo demás.

**Fricciones concretas a resolver:** (a) WFDB = ~176.000 archivos chicos, justo lo que el SSD exFAT hace mal — mitigación: son 12,6 GB, se descomprime en disco interno, se convierte a HDF5 replicando `convert_ptbxl.py` y recién ahí se mueve; (b) duraciones de 5 a 144 s, las de <7,0 s las descarta la Fase 2 tal cual está, e INCART (74 registros, 257 Hz, 30 min) conviene descartarlo entero; (c) las etiquetas son SNOMED-CT dentro del header, hace falta un parser más el mapeo de [`dx_mapping_scored.csv`](https://github.com/physionetchallenges/evaluation-2021).

**Espacio disponible medido (2026-08-26):** SSD `D:` con **771,9 GB libres** de 931,5; disco interno `C:` con 116,6 GB libres. El espacio no es restricción para ninguno de los dos datasets.

**[MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/) — la receta del top-5.** ~800.000 ECG de ~160.000 pacientes, **90,4 GB**, 10 s a 500 Hz, WFDB. Trae mediciones automáticas (intervalo RR, inicio/fin de QRS) y ~600.000 informes de cardiólogo en texto libre (vía MIMIC-IV-Note). El equipo Ahus AIM (5° de 40) preentrenó el extractor de features para predecir **biomarcadores de sangre discretizados por percentiles** y recién después hizo fine-tuning a Chagas, con ensemble de 5 modelos.

**Corrección sobre el acceso:** en esta misma sesión se afirmó que requiere credencial de PhysioNet con curso CITI. **Es falso** — el proyecto pasó a acceso abierto (ODbL v1.0). El obstáculo real no es el acceso sino que las etiquetas son texto libre, y por eso el equipo del top-5 preentrenó contra biomarcadores en vez de contra diagnósticos.

**Lo que NO existe: no hay ningún dataset público de ECG con Chagas de Argentina ni de otro país endémico** más allá de los tres que ya tenemos. Los datasets extra de zonas endémicas del Moody Challenge son privados/ocultos, y REDS-II y ELSA-Brasil son cohortes de estudio, no descargas públicas. **El riesgo transversal del ROADMAP ("dependencia de un solo dominio geográfico") no se resuelve con datos públicos** — se resuelve consiguiendo datos locales, que es gestión institucional, no trabajo técnico.

### 7. Diferencias concretas contra la implementación de referencia

Del repo oficial del paper ([carji475/ecg-chagas](https://github.com/carji475/ecg-chagas)) y su sección de métodos, contra lo nuestro:

| | PLOS NTD 2023 | DECA |
|---|---|---|
| dropout | **0,5** | 0,2 (`train.py:181`) |
| weight decay | **0,001** | **ninguno** (`Adam` sin `weight_decay`, `train.py:237`) |
| batch | 32 | 128 |
| ensemble | **15 modelos, distintas semillas** | 1 modelo |
| early stopping | val loss | AUPRC arena A |
| ventana | 4.096 muestras @400 Hz, **rellenada con ceros** | 2.800 @400 Hz, **descarta** (Fase 2) |
| edad/sexo como entrada | no | no |

Dos observaciones:
- **Estamos regularizando bastante menos que la referencia** (dropout 0,2 vs 0,5, sin weight decay) en una tarea donde el overfitting aparece rápido — consistente con la loss que se desploma en la ablación de sampler.
- **El ensemble nunca se probó y es la ganancia más barata disponible.** El paper usó 15 modelos; el 5° puesto del challenge usó 5. Nosotros corrimos siempre 1 — y ya hay **5 modelos entrenados** en `~/DECA-datasets/modelos/` (`abl-peso1`, `abl-peso1-seed123`, `real30ep`, `real8ep`, `abl-peso5`). Promediar sus logits y evaluar arena A **no cuesta ni una época de GPU**.

### Qué queda pendiente de esta sesión

Ordenado por relación impacto/costo:

1. **Ensamblar los 5 checkpoints existentes** y medir arena A. Costo cero de entrenamiento, nunca probado.
2. **Recuperar el resultado de `real30ep-paciencialr8`** (hallazgo 3) — es la prueba de si el techo de 0,84 era del scheduler o real.
3. **Expandir las cabezas auxiliares** de 1 (RBBB) a los 3 patrones del ROADMAP usando los códigos SCP de PTB-XL (hallazgo 5), con la cabeza de Chagas enmascarada y vigilando el diagnóstico de atajo. Da supervisión densa alineada al objetivo clínico real *y* la explicabilidad que el ROADMAP promete ("se deriva porque hay BRD+HBAI") y hoy no entregamos.
4. **Igualar la regularización de la referencia** (dropout 0,5, weight decay 1e-3): una corrida.
5. **Revisar el go/no-go de Fase 3** a la luz del hallazgo 2: los umbrales están fuera del alcance del estado del arte, y están expresados en AUC cuando la decisión real se toma sobre AUPRC/punto de operación. **Requiere decisión del usuario, no se cambia unilateralmente** — un criterio prefijado que se ajusta después de ver resultados pierde su función, así que el cambio tiene que ser explícito, argumentado y fechado.
6. Evaluar la descarga de Challenge 2021 (hallazgo 6), previo chequeo de espacio en el SSD.

---

## Sesión del 2026-08-26/27 (noche) — descarga y conversión de Challenge 2021

Ejecutada en modo autónomo (Axel durmiendo, sin commits). Cubre la acción concreta del punto 6 de arriba: bajar Challenge 2021 al SSD y dejarlo en un HDF5 unificado, listo para una futura integración a Fase 4.

### Corrección sobre el hallazgo 6: INCART y PTB/PTB-XL no están en el dataset real

La sesión del 2026-08-26 (hallazgo 6, tabla de composición) listaba INCART (74 registros) y PTB/PTB-XL (21.837) entre las fuentes públicas de Challenge 2021. **Es incorrecto.** El manifiesto real (`SHA256SUMS.txt` de la v1.0.3, 125.820 líneas) no trae ninguna carpeta `training/incart/`, `training/ptb/` ni `training/ptb-xl/`. Las únicas 5 fuentes bajo `training/` son:

| fuente | archivos (.hea+.mat) | registros (÷2) |
|---|---|---|
| chapman_shaoxing | 20.505 | ~10.252 |
| cpsc_2018 | 13.762 | ~6.881 |
| cpsc_2018_extra | 6.910 | ~3.455 |
| georgia | 20.699 | ~10.349 |
| ningbo | 63.881 | ~31.940 |
| **total** | **125.758** | **~62.879** |

Consecuencia práctica: no hizo falta descartar INCART a mano ni deduplicar contra el PTB-XL que ya tenemos — ninguno de los dos estaba en el dataset para empezar. Los números de "aporte neto" de la tabla de ROADMAP.md (que restaban PTB-XL) hay que releerlos con esto en mente; quedan pendientes de recalcular sobre los datos reales (ver más abajo).

### Descarga: de 273 a 366 archivos/min optimizando conexión, no concurrencia

Axel pidió explícitamente que la descarga fuera directo al SSD (`D:\DECA-datasets\challenge2021`), no a disco interno primero: la notebook no se iba a usar al día siguiente y el destino final es el SSD portable.

Primer intento (`requests.get()` de módulo, sin reuso de conexión): 12 threads dieron 273 archivos/min; subir a 40 threads lo **empeoró** a 193/min (más threads sin pooling = más contención de handshakes TLS nuevos, no más throughput). Fix real: `requests.Session()` compartida + `HTTPAdapter(pool_connections=N, pool_maxsize=N)`. Con eso, 20 threads dieron 366/min (+34% sobre el mejor intento anterior), y subir a 48 threads volvió a empeorar (345/min) — **el techo de concurrencia útil está en ~20**, sin errores de servidor visibles, así que es saturación de enlace o del lado de PhysioNet, no un bug local.

Comparación pedida por Axel (bajar a disco interno primero vs. directo al SSD): **prácticamente igual** (368/min en `C:` vs. 366/min en `D:`) — a ~6 archivos/seg exFAT no es el cuello de botella, la latencia de red domina en los dos casos. Se descartó el paso extra de "bajar y mover".

Config final: `src/../scratchpad/download_challenge2021.py` (script de descarga, no vive en el repo — es de un solo uso), `Session` compartida, 20 workers, manifiesto = `SHA256SUMS.txt` filtrado a `training/`. Log en el mismo scratchpad de la sesión.

### Conversor: `src/convert_challenge2021.py`

Nuevo módulo, análogo a `convert_ptbxl.py` pero con parseo de diagnóstico SNOMED-CT en vez de asumir una etiqueta ya provista. Decisiones de diseño:

- **Frecuencia y ventana: 500 Hz / 5.000 muestras, igual convención que `ptbxl.hdf5`** (no la duración nativa de cada registro, que varía: la mayoría de las 5 fuentes son 10 s nativos, pero CPSC/CPSC-Extra van de ~6 a ~60 s). Los registros más largos se recortan **centrados** a 5.000 muestras; los más cortos se rellenan con cero al final. Es un recorte sin pérdida para el pipeline tal como está hoy: Fase 2 (`preprocess.py`) también recorta centrado (a 7,0 s) y descarta el padding de ceros antes de usar la señal, así que el centro que termina viendo Fase 2 es el mismo que vería si se le diera el registro nativo completo — recortar centrado en la conversión no cambia el resultado final, solo ahorra espacio.
- **Mapeo SNOMED → patrón**, subconjunto de `dx_mapping_scored.csv` (repo `physionetchallenges/evaluation-2021`, no versionado en el repo — son 8 códigos, quedaron hardcodeados en el módulo con cita a la fuente): BRD = {CRBBB 713427006, RBBB 59118001, IRBBB 713426002}, HBAI = {LAnFB 445118002}, PVC = {PVC 427172004, VPB 17338001}, QAb = {164917005}, PRWP = {365413008}. Las equivalencias (CRBBB≡RBBB, PVC≡VPB) son las que el propio challenge puntúa como el mismo hallazgo clínico.
- **No toca `metadata.parquet` ni `build_metadata.py`.** Escribe su propio `challenge2021_labels.csv` (análogo a `exams.csv`/`ptbxl_database.csv` por fuente) más `challenge2021.hdf5`. Integrarlo al `metadata.parquet` consolidado y a Fase 4 es la decisión de arquitectura pendiente (ver "Qué falta" más abajo).

**Prueba de humo** (contra los ~4.300 registros de `chapman_shaoxing` ya descargados en ese momento): 4.286/4.306 convertidos, 20 salteados por `FileNotFoundError` (pares .hea/.mat todavía incompletos por la descarga en curso, no un bug — se resuelven solos en la corrida final). Validación cruzada contra la tabla oficial: `hbai=0` y `prwp=0` en chapman_shaoxing coinciden **exactamente** con las columnas `Chapman_Shaoxing` de `dx_mapping_scored.csv` (esa fuente tiene 0 casos anotados de esos dos patrones específicamente) — confirma que el parseo de SNOMED y el mapeo están bien. Señal en rango físico correcto (mV, std ~0,3), shape (5.000, 12), sin nulos inesperados en ningún campo del CSV. Se borraron los artefactos de la prueba antes de la corrida final.

### Qué falta (decisión de Axel, no se improvisa de noche)

**Integrar Challenge 2021 a Fase 4 requiere más que repetir el mecanismo de RBBB.** Repasando `train.py`: hoy la cabeza de Chagas **no tiene máscara** — `l_chagas = (loss_chagas(...) * peso).sum() / peso.sum()`, sin factor de máscara, porque hasta ahora los tres datasets siempre traen `chagas_label` concreto (True/False). Sumar Challenge 2021 como supervisión auxiliar (que es todo el punto: atacar el atajo de fuente agregando orígenes sin etiqueta de Chagas) obliga a **introducir por primera vez un `chagas_mask`** — no solo copiar el patrón de `rbbb_mask` a las cabezas nuevas. Eso toca `model.py` (cabezas nuevas), `dataset.py` (merge de `challenge2021_labels.csv` + máscara de Chagas) y `train.py` (término de loss nuevo por cabeza). Es exactamente lo que FASES.md ya marcaba como "no resuelto, se decide midiendo" (punto 3 de la lista de pendientes, arriba) — no es una tarea mecánica de una noche, y tocar la arquitectura del modelo sin que Axel lo revise despierto no correspondía al pedido de dejar "el SSD listo en formato correcto".

Pendiente para cuando Axel decida encararlo:
1. Agregar `chagas_mask` (hoy inexistente) antes de poder sumar cualquier fuente sin label de Chagas.
2. Elegir cuántas cabezas nuevas (¿una por patrón, o BRD+HBAI combinado ya que es el patrón clínico que importa?) y su peso en la loss total.
3. Correr el diagnóstico de atajo de fuente con Challenge 2021 adentro antes de confiar en cualquier mejora — es el mismo riesgo que motivó excluir PTB-XL originalmente, ahora con 5 orígenes nuevos en vez de 1.
4. Recalcular la tabla de "aporte neto" de ROADMAP.md con los conteos reales de `challenge2021_labels.csv` (la tabla actual asumía PTB-XL adentro del dataset, que resultó no estar).

### Estado al cortar la sesión (2026-08-27, madrugada) — qué falta

**Descarga: 100% completa y verificada.** 125.757/125.758 archivos en `D:\DECA-datasets\challenge2021\training\` (el único que falta, `training/ningb`, es una línea corrupta del manifiesto de PhysioNet — no es un archivo real, nunca va a existir). Desglose por fuente: chapman_shaoxing 20.505, cpsc_2018 13.762, cpsc_2018_extra 6.910, georgia 20.699, ningbo 63.881 archivos (.hea+.mat). 62.846 `.hea` vs 62.845 `.mat` (1 registro con par incompleto, ruido esperado — el conversor lo salta solo).

**Conversión: escrita y validada por prueba de humo, pero la corrida completa NO terminó.** Se intentó dos veces:
1. Primera vez, corriendo como task en background de la sesión de Claude Code: se cortó sola sin completar ni el 1% — la sesión se reinició (reconexión) y mató el proceso, que no sobrevive a eso.
2. Segunda vez, lanzada como proceso de Windows desatado de la sesión (`Start-Process` de PowerShell, para que sobreviva a un reinicio de sesión): Axel avisó que se iba a otra compu con el SSD en 10 minutos, así que se cortó a propósito para no dejar el HDF5 a medio escribir mientras se desconecta el disco. **Se limpiaron los archivos parciales** (`challenge2021.hdf5.tmp`) — no quedó nada corrupto ni a medias en el SSD.

**Para retomar (en esta compu o en otra con el SSD conectado):**
```
python src/convert_challenge2021.py
```
Tarda ~55 minutos (~19 registros/seg, medido). No necesita red — todo el trabajo es local (leer WFDB de `training/`, escribir `challenge2021.hdf5` + `challenge2021_labels.csv`). Requiere el mismo entorno que el resto del repo (`h5py`, `wfdb`, `pandas`, `numpy`, `tqdm` — ya en `requirements.txt`/`.venv`). Si se corta a la mitad, borrar `challenge2021.hdf5.tmp` antes de re-correr (el script no pisa un `.hdf5` final que ya exista, pero tampoco retoma un `.tmp` parcial — arranca de cero).

Una vez que termine, falta: (a) completar esta sección con los números reales de conteo por patrón (hoy son solo la muestra de humo sobre chapman_shaoxing), (b) recalcular la tabla de ROADMAP.md con los conteos reales, (c) la decisión de arquitectura para integrarlo a Fase 4 (sección de arriba, "Qué falta" — sigue sin resolver).

---

## Fase 5 — Validación y evaluación 🔲

**Tareas:**
- Evaluar con la métrica definida en la Fase 3, reportando por separado el desempeño en cada dataset de origen (no solo agregado).
- Análisis de errores: revisar falsos negativos en particular, dado el alto costo clínico.
- Comparar contra el baseline y contra cualquier resultado publicado del Moody Challenge como referencia (sin ser el objetivo a batir).

**Dificultades:**
- **Generalización entre poblaciones.** Buen desempeño en los datasets de entrenamiento no garantiza nada sobre una población distinta (otro país, otro equipo de ECG) — el objetivo final es uso en Argentina, con datos de entrenamiento mayormente brasileños/europeos.
- **PTB-XL como "negativo" es una simplificación.** Es una población de zona no endémica, no pacientes con serología negativa confirmada para Chagas — puede introducir sesgo si el modelo aprende diferencias poblacionales/equipamiento en vez de ausencia de patología.

---

## Fase 6 — Validación clínica y contrato con el resto de DECA 🔲

**Tareas:**
- Definir estrategia de validación clínica/estadística antes de considerar el modelo apto para uso real (mencionado como pendiente en el ROADMAP).
- Definir el contrato de entrada/salida entre este módulo de IA y el Backend: formato de ECG que sube el usuario, formato de respuesta (score, indicio sí/no, nivel de confianza).

**Dificultades:**
- **Brecha entre métricas offline y validación clínica real.** Un buen AUC/recall en test no reemplaza una validación con profesionales de salud sobre casos reales antes de exponer la herramienta a pacientes.
- **Formato de entrada del usuario final vs. formato de entrenamiento (WFDB).** Es poco probable que el usuario final suba archivos `.dat`/`.hea` — hay que definir de qué formato realista (ej. exportación de un equipo de ECG comercial, PDF, imagen) se va a partir en producción, lo cual puede requerir una etapa adicional de conversión no contemplada en el training.
- **Expectativas del usuario/paciente.** Comunicar claramente que es una herramienta de *screening*, no de diagnóstico, es un problema de producto tanto como de IA (evitar que un resultado "sin indicios" se interprete como "no tengo Chagas").

---

## Riesgos transversales (aplican a todas las fases)

- **Cambio de alcance del challenge de origen.** Si el Moody Challenge 2025 actualiza su código de referencia o métrica mientras avanzamos, hay que decidir si seguimos esa actualización o congelamos una versión propia.
- **Dependencia de un solo "dominio" geográfico de entrenamiento** (Brasil/Europa) para una herramienta pensada para Argentina — validar cuanto antes con cualquier dato local disponible, aunque sea chico.
- **Confusión entre "detectar Chagas" y "detectar los 3 patrones ECG objetivo".** Vale la pena recordar en cada fase que el alcance explícito es *indicios/screening*, no diagnóstico (ver ROADMAP).
