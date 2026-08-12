# IA — Fases del proyecto y dificultades

Este documento complementa a [ROADMAP.md](ROADMAP.md): mientras el roadmap define el **qué** (contexto, objetivo, datasets), acá se define el **cómo y en qué orden**, fase por fase, junto con las dificultades esperadas en cada una. Estado actual: **Fase 0 completa y verificada** (los 3 datasets descargados, unificados en HDF5, consolidados en `metadata.parquet` y validados por round-trip). **Fase 1 completa** (las 4 tareas de EDA cubiertas por notebooks reproducibles en `notebooks/`); la **Fase 3 quedó cerrada** (4 decisiones tomadas el 2026-08-08, punto de operación en 95% de sensibilidad con tres bandas y go/no-go escrito antes de entrenar). **Fase 2 completa (2026-08-10)**: diseño corregido y código implementado (`src/split_patients.py`, `src/preprocess.py`), corrido sobre el corpus completo: 362.363 registros preprocesados en `fase2_preprocessed.hdf5`, con el split por paciente verificado sin fuga. **Fase 4 en progreso (arrancada 2026-08-12)**: pipeline de entrenamiento implementado y validado de punta a punta (`src/dataset.py`, `src/model.py`, `src/evaluar.py`, `src/train.py`), corre en Windows y Linux; primera corrida real de 1 época sobre el dataset completo dio AUC 0,80 en la arena de go/no-go, con el diagnóstico de atajo de fuente en la dirección correcta. Falta correr más épocas y las ablaciones antes de llegar al criterio de despliegue.

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
