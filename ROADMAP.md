# IA — Detección de indicios de cardiopatía chagásica en ECG

Este documento marca el rumbo del módulo de **IA** dentro de DECA. Es el único alcance de este `.md`: no cubre backend, frontend ni infraestructura del proyecto.

## Contexto del problema

En Argentina se estima que hay aproximadamente **1.400.000** personas contagiadas con la enfermedad de Chagas. De ellas:

- Unas **300.000** presentan cardiopatía chagásica (afecta el funcionamiento del corazón).
- Unos **1.300 niños por año** nacen con Chagas congénito.

### Por qué el diagnóstico falla en la práctica

1. **Nadie se testea, aunque el test es gratuito.** La ley n.° 26.281 garantiza acceso gratuito al test serológico en cualquier hospital público del país. El problema no es el acceso, sino que la enfermedad puede cursar años o décadas de forma asintomática, por lo que la mayoría de los contagiados no sabe que lo está. Solo el **30%** de los infectados son conscientes de su enfermedad.
2. **Alta automedicación retrasa la consulta médica.** Argentina tiene una de las tasas de automedicación más altas del mundo: más del **70%** de la población consume medicamentos sin consulta médica. Síntomas que podrían ser indicio de algo mayor (palpitaciones, cansancio, etc.) se pasan por alto y nunca llegan a un médico, retrasando aún más el acceso al test.
3. **Falta de especialistas para interpretar el ECG.** Aunque se realice un ECG, identificar en él los patrones propios de cardiopatía chagásica requiere un cardiólogo entrenado, recurso escaso en el norte del país (la región de mayor prevalencia).

## Objetivo de la IA

Desarrollar un modelo que **lea los datos de un ECG y detecte patrones asociados a cardiopatía chagásica**, para poder derivar al paciente a realizarse el test serológico correspondiente.

**Importante — alcance explícito:**
- ❌ La IA **no diagnostica** la enfermedad de Chagas.
- ✅ La IA **detecta indicios/screening** en un estudio de rutina (ECG que ya se realiza mucha gente por trabajo, chequeos, o algún malestar) para redirigir al paciente hacia el test serológico, que es el que efectivamente diagnostica.

Esto convierte un examen ya masivamente utilizado en una herramienta de detección temprana, sin depender de la disponibilidad de un cardiólogo especializado.

## Patrones electrocardiográficos objetivo

El modelo debe apuntar a reconocer, en particular, la combinación de hallazgos característicos de cardiopatía chagásica:

- **Bloqueo de rama derecha (BRD)** combinado con **hemibloqueo anterior izquierdo (HAI)** — la combinación es especialmente sugestiva de Chagas.
- **Extrasístoles ventriculares.**
- **Zonas eléctricamente inactivas** compatibles con el aneurisma apical típico de la enfermedad.

## Dataset y formato de datos

**Nota:** DECA no es una participación en el Moody Challenge 2025 ni busca resolver su tarea tal cual está planteada. Lo usamos como **plataforma**: nos aprovechamos de sus datasets, del formato de datos ya estandarizado y del código de referencia, para no partir de cero.

### Datasets disponibles

| Dataset | Origen | Tamaño | Etiquetas | Frecuencia | Duración |
|---|---|---|---|---|---|
| **CODE-15%** | Brasil (Minas Gerais, telesalud), 2010–2016 | ~300.000+ registros | **Autorreportadas** ("weak labels"), ~2% positivos | 400 Hz | 7.3s o 10.2s |
| **SaMi-Trop** | Brasil (cohorte de pacientes con Chagas), 2011–2012 | 1.631–5.019 registros según la fuente | **Validadas serológicamente** ("strong labels"), ~93% positivos (es una cohorte de enfermos) | 400 Hz | 7.3–10.2s |
| **PTB-XL** | Europa, 1989–1996 | 21.799 registros | Presumidos **negativos** (controles, zona no endémica) | 500 Hz | 10s |
| Datasets privados | Zonas endémicas de Chagas en Centro/Sudamérica | No públicos | — | — | — |

Puntos a tener en cuenta para el diseño del modelo:
- Las frecuencias de muestreo **no son uniformes** entre datasets (400 Hz vs 500 Hz) → va a hacer falta resamplear/normalizar.
- El balance de clases es muy distinto entre datasets (CODE-15% casi todo negativo y con ruido en el label; SaMi-Trop casi todo positivo con label confiable) → afecta el diseño del split train/val/test y de la función de pérdida.
- Las etiquetas autorreportadas (CODE-15%) son más ruidosas que las validadas serológicamente (SaMi-Trop) — distinguir "weak" vs "strong" labels al ponderar el entrenamiento.

### Formato de archivos

Los datasets **no vienen todos en el mismo formato**:

| Dataset | Formato nativo | Señal | Metadata / labels |
|---|---|---|---|
| CODE-15% | **HDF5** | `exams_part{0..17}.hdf5` (tracings, 400 Hz) | `exams.csv` + `code15_chagas_labels.csv` |
| SaMi-Trop | **HDF5** | `exams.hdf5` (tracings, 400 Hz) | `exams.csv` (todos positivos, serología) |
| PTB-XL | **WFDB** (`.dat`/`.hea`) | `records500/` (500 Hz) | `ptbxl_database.csv` (todos negativos por presunción) |

**Decisión de formato canónico: HDF5.** Unificamos los tres en HDF5 porque dos de los tres ya lo son (para ellos es solo descomprimir) y porque en el SSD externo (exFAT) unos pocos archivos grandes rinden mucho mejor que cientos de miles de archivos chicos. Solo **PTB-XL** requiere conversión real WFDB→HDF5 (`wfdb` + `h5py`), y es el dataset más chico.

- Se guarda **un HDF5 por dataset** (las frecuencias 400 Hz vs 500 Hz se unifican recién en el resampleo de la Fase 2, no al almacenar).
- Las etiquetas y demografía se consolidan en una **tabla de metadata unificada** (`metadata.parquet`) con columnas: `record_id`, `dataset`, `edad`, `sexo`, `frecuencia`, `duracion`, `chagas_label`, `confianza` (weak / strong / negativo-presunto).
- 12 derivaciones estándar: I, II, III, aVR, aVL, aVF, V1–V6.

Nota: el formato de almacenamiento del dataset **no** condiciona la inferencia sobre ECGs reales — un ECG entra al modelo como un tensor `(12 derivaciones × N muestras)` a una frecuencia fija, se defina como se defina el contrato de entrada (Fase 6).

### Código de referencia (Moody Challenge 2025)

DECA es un proyecto **aparte**; el challenge se usa solo como plataforma (datasets + formato ya estandarizado). No adoptamos su tarea ni su forma de evaluar. El código de referencia queda como material del que *podemos robar* piezas útiles si conviene, no como base obligada:
- [python-example-2025](https://github.com/physionetchallenges/python-example-2025) — utilidades de carga y pipeline de ejemplo. Está pensado para WFDB; como nosotros vamos a HDF5, sirve de referencia conceptual, no para usar tal cual.

## Referencias

- [PhysioNet — Moody Challenge 2025](https://moody-challenge.physionet.org/2025/): desafío específico sobre detección de Chagas en ECG, planteado por PhysioNet (MIT/Harvard) en 2025. Lo usamos como plataforma (datasets + formato + código de referencia), no como objetivo a resolver.
- [PhysioNet — repositorio general de datasets](https://physionet.org/content/): fuente de datasets de señales fisiológicas (ECG y otros) para complementar o validar más allá del challenge.
- [arXiv:2510.02202](https://arxiv.org/pdf/2510.02202) — "Detection of Chagas Disease from the ECG: The George B. Moody PhysioNet Challenge 2025": paper de referencia técnica que describe la tarea y los datasets en detalle.

## Próximos pasos a definir

Estos son los temas que quedan abiertos para las próximas conversaciones, antes de escribir código:

- [x] Revisar en detalle los datasets del challenge que usamos como plataforma (CODE-15%/SaMi-Trop/PTB-XL, formatos nativos). Ver sección "Dataset y formato de datos".
- [x] Definir el formato canónico de almacenamiento: **HDF5 unificado** (ver "Formato de archivos").
- [ ] Definir qué se considera "positivo" (indicio a derivar) vs. "negativo" en términos operables para el modelo, en línea con los 3 patrones objetivo.
- [ ] Elegir métricas de éxito acordes al caso de uso de screening (priorizar sensibilidad/recall sobre precisión, dado que el costo de un falso negativo es mucho mayor que el de un falso positivo que termina derivando a un test serológico gratuito).
- [ ] Definir el enfoque de modelado (señal cruda vs. features clásicas de ECG, arquitectura, etc.) y el stack técnico.
- [ ] Definir el contrato de entrada/salida entre este módulo de IA y el resto de DECA (Backend), incluyendo formato de datos de ECG que subirá el usuario.
- [ ] Definir estrategia de validación clínica/estadística antes de considerar el modelo apto para uso real.
