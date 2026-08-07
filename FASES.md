# IA — Fases del proyecto y dificultades

Este documento complementa a [ROADMAP.md](ROADMAP.md): mientras el roadmap define el **qué** (contexto, objetivo, datasets), acá se define el **cómo y en qué orden**, fase por fase, junto con las dificultades esperadas en cada una. Estado actual: **Fase 0 completa** (los 3 datasets están descargados, unificados en HDF5 y consolidados en `metadata.parquet`).

Convención de estado: 🔲 no iniciada · 🟡 en progreso · ✅ completa.

---

## Fase 0 — Entorno y obtención de datos ✅

Antes de escribir cualquier línea de modelado hay que poder leer un ECG.

**Tareas:**
- [x] Crear entorno de Python para IA (venv/conda) con `h5py`, `numpy`, `pandas`, `scipy` como base, más `wfdb` (solo se usa para convertir PTB-XL de WFDB→HDF5). `.venv` creado en el repo (2026-08-07) con `requirements.txt`.
- [x] Descargar los datasets desde Zenodo/PhysioNet: CODE-15%, SaMi-Trop, PTB-XL (ver tabla en el ROADMAP). Se guardan en el **SSD externo**, no en git. CODE-15% y SaMi-Trop ya estaban; PTB-XL (zip + WFDB) se completó y confirmó el 2026-08-07.
- [x] Unificar a **HDF5** (formato canónico, ver ROADMAP): CODE-15% y SaMi-Trop solo se descomprimen; PTB-XL se convierte. Consolidar labels/demografía en `metadata.parquet`. `python src/convert_ptbxl.py` corrido el 2026-08-07: 21.799 registros, `ptbxl.hdf5` (4.87 GB), sin NaN. Los WFDB de `records500/` ya convertidos se borraron para liberar espacio (se conserva `records100/` y el zip original). `metadata.parquet` regenerado con los 3 datasets (`build_metadata.py`).
- [x] Definir cómo se referencian los datos desde el código (variable de entorno / config apuntando a la ruta del SSD, no paths hardcodeados). `src/config.py` ya usa `DECA_DATA_DIR`. **Nota:** el default hardcodeado es `D:/DECA-DATASETS`, pero el SSD actual monta en `E:\DECA-datasets` (minúsculas) — hay que setear `DECA_DATA_DIR=E:/DECA-datasets` al correr los scripts, o el default va a fallar.

**Dificultades:**
- **Peso.** CODE-15% son ~46 GB comprimidos (cientos de miles de registros); no entra en git ni conviene en el disco interno. Por eso vive en el SSD externo.
- **exFAT y archivos chicos.** El SSD es exFAT: rinde bien con pocos archivos grandes (HDF5) y mal con cientos de miles de archivos chicos (motivo extra para no usar WFDB por-registro en CODE-15%).
- **Formato HDF5 al principio.** Primer contacto con `h5py` y lectura de señales multicanal por bloques (`x[start:end]`) — curva de aprendizaje corta pero real. Para PTB-XL además hay que leer WFDB una vez para convertirlo.
- **Datasets privados de Centro/Sudamérica** mencionados en el ROADMAP no son públicos — esta fase solo cubre CODE-15%, SaMi-Trop y PTB-XL, y queda pendiente gestionar el acceso a los privados más adelante (posible cuello de botella institucional, no técnico).

---

## Fase 1 — Exploración de datos (EDA) 🔲

**Tareas:**
- Cargar una muestra de cada dataset y visualizar señales crudas (las 12 derivaciones).
- Revisar distribución de edades, sexo, frecuencia de muestreo real vs. nominal, duración real de cada registro.
- Cuantificar el desbalance de clases por dataset (CODE-15% ~2% positivos, SaMi-Trop ~93% positivos).
- Revisar calidad de señal: ruido, artefactos de movimiento, derivaciones faltantes o corruptas.

**Dificultades:**
- **Tres datasets, tres realidades distintas.** No se pueden explorar como si fueran uno solo: CODE-15% es population-based con etiqueta autorreportada, SaMi-Trop es una cohorte de enfermos con etiqueta serológica, PTB-XL es de una región no endémica y sirve como negativo "por presunción" más que por diagnóstico confirmado. Cualquier estadística agregada sin discriminar por origen va a ser engañosa.
- **Ruido de señal real.** Estos son ECG de campo (telesalud en zonas rurales para CODE-15%/SaMi-Trop), no señales de laboratorio — esperar artefactos, baseline wander, ruido de línea eléctrica.
- **Volumen para EDA manual.** Con cientos de miles de registros no se puede inspeccionar todo a ojo; hay que definir criterios automáticos de calidad de señal antes de poder confiar en agregados.

---

## Fase 2 — Preprocesamiento y unificación 🔲

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
- **Fuga de información (data leakage).** Si un mismo paciente aparece en más de un registro/dataset, hay que evitar que termine partido entre train y test.

---

## Fase 3 — Definición operacional de la tarea 🔲

Esta fase es más de decisión que de código, pero es bloqueante para las siguientes.

**Tareas:**
- Definir qué cuenta como "positivo" (indicio a derivar) vs. "negativo", en línea con los 3 patrones objetivo del ROADMAP (BRD+HAI, extrasístoles ventriculares, zonas eléctricamente inactivas).
- Elegir la métrica de éxito para un caso de uso de screening: priorizar sensibilidad/recall (falso negativo = paciente con Chagas no derivado, mucho más costoso que un falso positivo que solo deriva a un test gratuito). La métrica la definimos nosotros según el caso de uso; **no** adoptamos la forma de evaluar del challenge.

**Dificultades:**
- **SaMi-Trop: 93% vs 100% positivos, sin resolver.** El ROADMAP indica ~93% positivos para SaMi-Trop, pero el `exams.hdf5`/`exams.csv` descargado (1.631 registros, columnas: `exam_id, age, is_male, normal_ecg, death, timey, nn_predicted_age`) **no trae columna `chagas`** — no hay forma de leer el label por registro desde este archivo. Dos hipótesis sin confirmar: (a) este es ya el subconjunto curado 100% confirmado positivo (1.631 coincide con el extremo bajo del rango 1.631–5.019 que da el ROADMAP "según la fuente"), o (b) falta un archivo de labels adicional que no se descargó y el 93% aplica también acá. **No asumir 100% en `build_metadata.py` hasta confirmar** — por ahora el script lo deja hardcodeado a `True` pero es un supuesto sin validar, marcado explícitamente en el código.
- **No hay ground truth directo de los 3 patrones ECG.** Las etiquetas disponibles son "Chagas sí/no" (por serología o autorreporte), no "presenta BRD+HAI sí/no". Esto significa que el modelo aprenderá a predecir la enfermedad de forma indirecta, no los patrones específicos mencionados en el objetivo — hay que decidir si eso es aceptable o si se necesita anotación adicional (posiblemente manual, por cardiólogo) para un subconjunto.
- **Tensión sensibilidad vs. especificidad.** Optimizar solo para recall puede disparar los falsos positivos a un nivel que vuelva la herramienta inútil en la práctica (saturar el sistema de testeo serológico). Hay que fijar un piso de especificidad aceptable, no solo maximizar sensibilidad.

---

## Fase 4 — Modelado (baseline → iteración) 🔲

**Tareas:**
- Definir enfoque: señal cruda (deep learning, ej. CNN/RNN sobre las 12 derivaciones) vs. features clásicas de ECG (intervalos, morfología de onda) + modelo clásico.
- Levantar un baseline simple primero (aunque sea débil) antes de ir a arquitecturas complejas.
- Iterar arquitectura/hiperparámetros con validación cruzada respetando los splits por dataset definidos en la Fase 2.

**Dificultades:**
- **Desbalance extremo de clases** (2% positivos en el dataset más grande) — requiere técnicas específicas (pesos de clase, resampling, focal loss) y complica la elección de umbral de decisión.
- **Recursos de cómputo.** Entrenar sobre cientos de miles de señales de 12 derivaciones no es liviano; hay que dimensionar si se necesita GPU y de dónde sale.
- **Riesgo de overfitting a la fuente del dato, no a la enfermedad.** Si el modelo aprende a distinguir "viene de SaMi-Trop" vs. "viene de PTB-XL" (por diferencias de equipo, región, calidad de señal) en vez de patrones clínicos reales, va a tener buen desempeño en test pero fallar en producción con datos nuevos.

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
