# Investigación: reemplazar la ResNet1D por un transformer fundacional de ECG

Fecha: 2026-10-22. **Alcance: solo investigación.** No se entrenó nada, no se tocó código del repo y no se corrió `evaluar_test.py`.

**Convención de evidencia**
- **[V]** = verificado en la fuente citada (URL al lado), en esta sesión.
- **[R]** = ya registrado en `INVESTIGACION-LITERATURA.md` o `FASES.md` del repo, no re-verificado acá.
- **[I]** = inferencia mía. Puede estar mal; está marcada para que se pueda discutir.

---

## 0. Resumen en una pantalla

- **Si se hace, el candidato es ECG-FM** (wanglab / bowang-lab), en su checkpoint `mimic_iv_ecg_finetuned.pt`.
  - Es un transformer (wav2vec 2.0, 90,9M parámetros) con licencia MIT.
  - **Es el único transformer con pesos públicos cuyo preentrenamiento no incluye CODE / CODE-15% ni SaMi-Trop.** Sí incluye PTB-XL.
  - Su entrada es 500 Hz con z-score por derivación, en el orden de derivaciones canónico del proyecto.
  - Su checkpoint ajustado ya trae cabezas de **BRD, bloqueo bifascicular (BRD+HBAI), extrasístole ventricular e infarto**, que son los tres patrones del ROADMAP.
- **La referencia fuerte no-transformer es ECGFounder** (Net1D, una CNN de 76,3M parámetros, MIT).
  - Está preentrenado de forma supervisada sobre 10,7M ECG estadounidenses, sin CODE.
  - En el benchmark independiente más serio (ICLR 2026) está entre los 3 FM que le ganan consistentemente a un supervisado fuerte. ECG-FM quedó 4º.
- **Descartados por contaminación**, porque su preentrenamiento incluye CODE-15% o CODE completo, que contiene nuestros val y test:
  - HuBERT-ECG
  - ST-MEM, tanto la versión original como la del ganador del Challenge
  - ECG-JEPA
  - xECG
  - Dos de ellos tienen además licencia no comercial (CC BY-NC 4.0).
- **Expectativa honesta [I]: lo más probable es una mejora de 0 a 3 puntos de TPR@5% en val arena A.** Eso cae dentro del ruido entre semillas y del intervalo de confianza.
  - La evidencia externa es débil. El mejor FM del Challenge 2025 sacó 0,323 en test oculto contra 0,280 de la mejor ResNet. ECG-FM congelado sacó 0,245, por debajo de esa ResNet.
  - El techo documentado en FASES (82,7% de los perdidos sin ningún diagnóstico ECG etiquetable) no es un problema de capacidad del backbone.
  - **Si se corre, que sea como experimento con criterio de éxito pre-registrado** (sección 6) y con un paso barato primero: sonda lineal sobre embeddings cacheados.

---

## 1. Tabla comparativa

Orden de las columnas:
- **Arq.**: arquitectura.
- **Params**: tamaño.
- **Preentrenamiento**: datos y método.
- **CODE / SaMi / PTB-XL**: si cada uno está en el preentrenamiento.
- **Licencia**: de los pesos.
- **Entrada nativa**: formato que espera el modelo.
- **Chagas (test oculto)**: score del Challenge 2025 = TPR@5% medio en los 3 sets ocultos.

| Modelo | Arq. | Params | Preentrenamiento | CODE / SaMi / PTB-XL | Licencia | Entrada nativa | Chagas (test oculto) |
|---|---|---|---|---|---|---|---|
| **ECG-FM** | Transformer (wav2vec2: 4 conv + 12 capas BERT-base) | 90,9M | SSL (W2V+CMSC+RLM), 1,4M segmentos: MIMIC-IV-ECG + PhysioNet 2021 | **No / No / Sí** | MIT | 500 Hz, 5 s, z-score por derivación, orden I..V6 | 0,245 congelado + MLP (#416, 8º) |
| **ECGFounder** | **CNN** (Net1D, tipo RegNet) | 76,3M | Supervisado, 150 etiquetas, HEEDB 10,7M ECG (EE.UU.) | **No / No / No** (CODE-test solo como test externo) | MIT | 500 Hz, 10 s (5000), z-score | Sin entrada rankeada verificada (ver §2.2) |
| HuBERT-ECG | Transformer (HuBERT) | 30M / 93M / 188M | SSL, 9,1M ECG; **incluye CODE completo (2,3M) y PTB-XL**; SaMi-Trop excluido | **Sí (CODE completo) / No / Sí** | **CC BY-NC 4.0** | 100 Hz, 5 s, banda 0,05–47 Hz, escala [-1,1] | 0,204 (#197, 22º) |
| ST-MEM (original) | ViT-B | ~90M | SSL (MAE espaciotemporal), 188k: Chapman, Ningbo, **CODE-15% (143k)** | **Sí / No / No** | **CC BY-NC 4.0** | 250 Hz, 10 s, banda 0,67–40 Hz, z-score | — |
| ST-MEM del 1º puesto (KU Leuven) | ViT-1D, 12 bloques | ~90M | SSL ST-MEM, 400.365: CinC2020 (contiene PTB-XL), Chapman, CODE-test, **CODE-15% (345.799)** | **Sí / No / Sí** | **Pesos no publicados** | 100 Hz, 10 s, parche 50 | **0,323 (1º)** |
| ECG-JEPA | Transformer (JEPA, ViT-B) | 87M | SSL, ~174k: Chapman, Ningbo, **CODE-15% (130.900)** | **Sí / No / No** | MIT | 250 Hz, 10 s, 8 derivaciones | 0,217 (#219, 18º) |
| xECG | **xLSTM** (recurrente) | 57M | SSL (SimDINOv2): **CODE completo (2,3M)**, Chapman, Ningbo, INCART | **Sí / No / No** | MIT ("no para uso clínico ni comercial") | 100 Hz | Parte del 2º puesto (0,283) |
| MERL | CNN o ViT pequeño (ECG-texto) | 4,6M (CNN) | Contrastivo ECG-informe, MIMIC-IV-ECG | No / No / No | MIT | — | — |
| ECG-CPC | SSM (S4) | 3,8M | CPC sobre HEEDB | No / No / No | Pesos no confirmados | 240 Hz | — |

Fuentes de la tabla:
- [V] Benchmark Al-Masud, Lopez Alcaraz, Strodthoff, ICLR 2026: https://arxiv.org/html/2509.25095. De ahí salen los params de las 8 filas y la lista de preentrenamiento de HuBERT-ECG y ECG-FM.
- Las demás URL están en la sección 2.

---

## 2. Notas por modelo

### 2.1 ECG-FM (University Health Network / Vector, Bo Wang lab)

**Arquitectura** [V]
- wav2vec 2.0 con 90,9M parámetros.
- Codificador de features: 4 bloques convolucionales (256 canales, stride 2), lo que da submuestreo ×16.
- Embedding posicional relativo convolucional.
- Transformer de 12 capas, dimensión 768, 12 cabezas, FFN de 3072.
- Fuentes: https://arxiv.org/html/2408.05178 y https://github.com/bowang-lab/ECG-FM
- **[I]** Como el embedding posicional es convolucional y relativo, el modelo acepta largos distintos de 5 s. El equipo #416 lo usó con ventanas de 10 s (texto del paper CinC2025-416, leído en esta sesión).

**Preentrenamiento** [V]
- Híbrido: wav2vec2 + CMSC (contrastivo entre segmentos contiguos) + RLM (enmascarado aleatorio de derivaciones).
- 1.405.625 segmentos de MIMIC-IV-ECG y de PhysioNet 2021, con los subsets CPSC, CPSC-Extra, **PTB-XL**, Georgia, Ningbo y Chapman. Excluye PTB e INCART.
- **No incluye CODE, CODE-15% ni SaMi-Trop.**
- Fuente: JAMIA Open 2025, https://academic.oup.com/jamiaopen/article/8/5/ooaf122/8287827

**Pesos y licencia** [V]
- Repositorio: https://huggingface.co/wanglab/ecg-fm, licencia MIT.
- `mimic_iv_ecg_physionet_pretrained.pt` es el checkpoint solo SSL.
- `mimic_iv_ecg_finetuned.pt` está ajustado sobre MIMIC-IV-ECG con 17 etiquetas (`num_labels: 17` en el yaml).
- Las etiquetas (notebook `infer_quickstart.ipynb` del repo) incluyen:
  - *Right bundle branch block*
  - ***Bifascicular block***
  - *Premature ventricular contraction*
  - ***Infarction***
  - *Left bundle branch block*
  - *1st degree AV block*
  - *Atrial fibrillation*
  - y otras
- Esas etiquetas cubren los tres patrones del ROADMAP.
- Los modelos de FEVI (UHN-ECG) **no** se publicaron.

**Entrada** [V]
- Remuestreo a 500 Hz por interpolación lineal.
- Z-score por derivación sobre el registro: resta de la media y división por el desvío.
- Segmentos de 5 s, en el orden `['I','II','III','aVR','aVL','aVF','V1'..'V6']`.
- Fuente: `scripts/preprocess/ecg/preprocess.py` de https://github.com/Jwoo5/fairseq-signals (constante `LEAD_ORDER` y funciones `resample` y `lead_std_divide`).
- **Es exactamente el orden canónico y la normalización de DECA.**

**Resultados publicados** [V]
- FEVI ≤ 40%: AUROC 0,929, AUPRC 0,825 (UHN-ECG).
- BRD: AUROC 0,991 en UHN-ECG y 0,987 en MIMIC.
- Ajuste fino del paper: lr 1e-6, batch 256, en una A100 de 80 GB.

**En Chagas**
- **Equipo #416, "chagas_detector"** [V, texto del paper]:
  - ECG-FM congelado (checkpoint MIMIC-finetuned) + MLP con edad y sexo, a 500 Hz y 10 s, batch 16.
  - Validación oculta 0,323, test oculto **0,245**.
  - **Descongelar empeoró**: 0,335–0,342 contra 0,426 en su test interno.
  - El multitarea Chagas+BRD+BAV1 también empeoró: 0,349 contra 0,426.
- **Alvarez-Florez et al. 2026** [R, arXiv:2609.08582]:
  - ECG-FM sin alinear, sonda lineal, CV en CODE-15%+SaMi-Trop: **TPR@5% 0,377**, AUROC 0,827.
  - Alineado con RMN: 0,427 / 0,851.
  - Test oculto: 0,269.
- **Benchmark ICLR 2026** [V]: ECG-FM quedó 4º de 8 en interpretación de adultos y rindió por debajo de los líderes en PTB-XL, Georgia y Chapman.

**Código** [V]
- Todo el entrenamiento pasa por `fairseq-signals` (fork de fairseq con configs hydra).
- Su `setup.py` arrastra dependencias viejas (referencias a torch 1.7 / py3.6) y no menciona Windows.
- **[I]** Montarlo en la laptop Windows va a costar. En el box Linux es razonable.
- Una alternativa es extraer el `state_dict` y reimplementar el forward de wav2vec2-ECG en PyTorch puro (≈200 líneas). Hay que validarla contra la salida del original.

**Contaminación** [I]
- PTB-XL está en el preentrenamiento SSL, sin etiquetas de Chagas, pero sus señales fueron vistas. **Arena C** (PTB-XL, negativos) queda contaminada para cualquier comparación.
- Arena A (CODE-15%) y SaMi-Trop no.

### 2.2 ECGFounder (Harvard-Emory / PKU, NEJM AI 2025)

**Arquitectura** [V]
- Net1D: CNN tipo RegNet con convoluciones de grupo y atención por canal.
- 76,3M parámetros para 12 derivaciones; existe además un checkpoint de 1 derivación.
- **No es transformer.**
- Fuentes: https://arxiv.org/html/2410.04133 y `finetune_model.py` en https://github.com/PKUDigitalHealth/ECGFounder (`filter_list=[64,160,160,400,400,1024,1024]`).

**Preentrenamiento** [V]
- **Supervisado** sobre HEEDB: 10.771.552 ECG de 1,8M sujetos, 150 etiquetas.
- Mayormente cohortes de EE.UU. **CODE-test aparece solo como test externo.**
- Las etiquetas incluyen *RIGHT BUNDLE BRANCH BLOCK*, *PREMATURE VENTRICULAR COMPLEXES* e infartos anterior, inferior y lateral.
- No pude confirmar HBAI/LAFB en la lista parcial que leí [V parcial].

**Pesos** [V]
- `12_lead_ECGFounder.pth` y `1_lead_ECGFounder.pth` en https://huggingface.co/PKUDigitalHealth/ECGFounder, licencia MIT.
- `config.json`: `input_size 5000, 500Hz, 10s`.

**Entrada** [V]
- 500 Hz, 10 s, z-score por segmento, derivaciones faltantes en cero.
- Filtro: el paper es inconsistente (0,5–50 Hz contra 1–30 Hz) y hay que mirar `dataset.py`.

**Resultados**
- [V] BRD en test externo: AUC 0,976.
- [V] FEVI < 50% en MIMIC: AUC 0,867.
- [V] Está entre los 3 FM que superan consistentemente al supervisado S4 en el benchmark ICLR 2026.
- [V parcial] Chagas: el equipo ChagasExplorers (#259, ECGFounder congelado + clasificador liviano) reportó **0,988** en validación oculta en su abstract. No aparece en la tabla final de resultados del Challenge (https://moody-challenge.physionet.org/2025/results/challenge_2025_results.tsv). **[I]** Es casi seguro un artefacto de la fase no oficial; no es un dato utilizable.
- [R] Probablemente es el FM de "más de 10M ECG" de MIWEAR (#477, test 0,238), y fue base de ECGNeXt en CinCo Amigos.

**Código** [V] PyTorch plano (`net1d.py`), notebook de fine-tuning y python 3.10. **Es el más fácil de integrar al `train.py` actual.**

**Contaminación** [I] Ninguna conocida con nuestros datasets.

### 2.3 HuBERT-ECG (Coppola et al., medRxiv 2024)

**Arquitectura** [V] Transformer estilo HuBERT en tres tamaños: 30M, 93M y 188M (tabla del preprint).

**Preentrenamiento** [V, texto del PDF en medRxiv 10.1101/2024.11.14.24317328]
- SSL sobre 9,1M ECG formados por:
  - "las particiones etiquetadas y no etiquetadas de **Ribeiro (también conocido como CODE)**", con más de 2,3M ECG etiquetados
  - CPSC
  - **PTB y PTB-XL**
  - Georgia, Chapman y Ningbo
  - Tianchi (Hefei)
  - SPH
  - MIMIC-IV-ECG
- SaMi-Trop se excluyó del SSL y se usó en el ajuste fino.
- **[I]** CODE-15% es un subconjunto del CODE completo, así que **nuestro val y test casi con certeza están en el preentrenamiento**.
- Los checkpoints "Cardio-Learning" están además ajustados de forma *supervisada* sobre CODE, con BRD entre otras etiquetas, y sobre SaMi-Trop. Eso contamina también la cabeza de BRD.

**Entrada** [V] 100 Hz, 5 s, filtro FIR 0,05–47 Hz, reescalado a [-1, 1].

**Licencia** [V] **CC BY-NC 4.0**, en https://github.com/Edoar-do/HuBERT-ECG y https://huggingface.co/Edoardo-Coppola/hubert-ecg-base.

**Chagas** [V]
- #197 HeartGoesOut2U, HuBERT a 100 Hz con cabeza CNN: test **0,204** (22º).
- El benchmark ICLR 2026 dice que no supera a S4.

**Veredicto: descartado** por contaminación, licencia y rendimiento.

### 2.4 ST-MEM (Na et al., ICLR 2024, VUNO) y la variante del ganador del Challenge

**ST-MEM original** [V]
- ViT-B.
- Preentrenado sobre 188.480 ECG: Chapman, Ningbo y **CODE-15% (143.328 registros de 10 s)**.
- Entrada a 250 Hz, 10 s, banda 0,67–40 Hz, z-score.
- Fuentes: https://arxiv.org/html/2402.09450 y https://github.com/bakqui/ST-MEM
- Licencia del repo: **CC BY-NC 4.0** (archivo LICENSE leído). El README agrega "© VUNO Inc. All rights reserved".

**1º puesto, Biomed-Cardio** [V, texto del paper CinC2025-019 y https://arxiv.org/html/2509.00102v3]
- ViT-1D de 12 bloques, entrada de 100 Hz × 10 s, parche de 50.
- Preentrenado con ST-MEM sobre 400.365 ECG: CinC2020 (43.093, que contiene PTB-XL), Chapman, CODE-test y **CODE-15% (345.799)**.
- Los propios autores escriben que "el solapamiento entre preentrenamiento y fine-tuning es sustancial, ya que CODE-15% y PTB-XL aparecen en ambas etapas".
- Recetas de ajuste:
  - Promedio de las 12 capas del encoder.
  - Edad y sexo por FiLM.
  - Etiquetas suaves 0,8/0,2 solo en CODE-15%.
  - BCE con peso 5 más sobremuestreo ×5.
  - Aumentación de ruido de red de 50/60 Hz.
  - 2.000 iteraciones con backbone congelado y luego 12.000 completas, con batch 64.
- Resultados:
  - CV 0,490 (0,381 solo con la cabeza).
  - Validación oculta 0,445.
  - **Test 0,323**.
- Código en https://github.com/Xuanphu108/ecg_ssl (MIT, pero incorpora código de ST-MEM bajo CC BY-NC). **No publica pesos.**

**Veredicto: descartado como backbone** por contaminación con CODE-15%, licencia no comercial y falta de pesos. **Sus recetas de ajuste sí son reutilizables** (sección 5).

### 2.5 ECG-JEPA (Kim, 2024)

- [V] Transformer de 12 capas, 768 de dimensión (≈87M).
- [V] Preentrenado sobre Chapman, Ningbo y **CODE-15% (130.900 registros de 10 s)**.
- [V] Entrada de 250 Hz, 10 s, 8 derivaciones (I, II, V1–V6).
- [V] MIT. Fuentes: https://arxiv.org/html/2410.08559 y https://github.com/sehunfromdaegu/ECG_JEPA
- [V] Es de los mejores del benchmark ICLR 2026.
- [V] En Chagas (#219 Seoul Mates): test **0,217** (18º).
- **Descartado por contaminación con CODE-15%.**

### 2.6 xECG (Innsbruck, base del 2º puesto)

- [V] xLSTM bidireccional, 57M.
- [V] SimDINOv2 sobre **CODE completo (2.322.513 registros)**, Chapman, Ningbo e INCART, a 100 Hz.
- [V] Pesos en https://huggingface.co/riccardolunelli/xECG_base_model_v1. Repositorio https://github.com/dlaskalab/bench-xecg (MIT; "no destinado a uso clínico ni comercial").
- Fuente: https://arxiv.org/html/2509.10151v1
- [V] El 2º puesto (DlaskaLabMUI) además preentrenó sus MAE-Transformers (86M y 177M) sobre CODE de 8M, que no es público. Test 0,283.
- **Descartado: no es transformer y está contaminado.**

### 2.7 Otros revisados

- **MERL** (ECG-texto, MIMIC-IV-ECG): CNN de 4,6M o ViT. [V] No supera a S4 en el benchmark. Sin interés.
- **ECG-CPC** (Strodthoff, S4 de 3,8M, HEEDB): [V] el mejor del benchmark ICLR 2026 en 5 de 7 categorías, pero no encontré pesos publicados. No es transformer.
- **Estrategias de preentrenamiento** (Al-Masud & Strodthoff, arXiv:2605.12241): [V] el backbone S4 superó al Transformer y a Net1D en los cinco objetivos; CPC fue el más transferible y JEPA quedó cerca. **Preentrena con CODE-15%.**
- **AnyECG** (arXiv:2601.10748): [V] es ECGFounder ajustado sobre 13,3M ECG multicéntricos para 1.172 condiciones. No encontré pesos.
- **SwissBeatsNet** (#160): [V] MAE multiderivación preentrenado sobre CODE-15%, SaMi-Trop y PTB-XL, con encoder congelado. Test 0,226.
- **Ahus AIM** (#298, 5º): [V] preentrenamiento con biomarcadores de MIMIC sobre InceptionTime (CNN). Código y pesos en https://github.com/Ahus-AIM/physionet-challenge-2025. Test 0,269. No hay ablación del preentrenamiento.

---

## 3. Lo que dice la evidencia sobre "FM contra CNN" en Chagas

| Comparación | Valor | Tipo |
|---|---|---|
| 1º (ViT ST-MEM, contaminado con CODE-15%) contra 3º (ResNet1D desde cero) en test oculto | 0,323 contra 0,280 | [R]/[V] |
| ECG-FM congelado (#416) contra la ResNet1D del 3º en test oculto | 0,245 contra 0,280 | [V] |
| HuBERT-ECG (#197) y ECG-JEPA (#219) en test oculto | 0,204 y 0,217 | [V] |
| ECG-FM sonda lineal, CV CODE-15%+SaMi-Trop (Alvarez-Florez) | TPR@5% 0,377 | [R] |
| **DECA `patrones-lr8`, val arena A (CODE-15% solo)** | **TPR@5% 39,9%**, AUPRC 0,1798, AUC 0,8314 | [R] FASES.md |
| Ganador del Challenge, CV interna (incluye SaMi-Trop y PTB-XL) | 0,490 | [V] |

**Lectura** [I]
1. **Los transformers fundacionales no ganan de forma limpia en Chagas.** El único FM que superó a la mejor CNN en test oculto (+4,3 puntos) está contaminado con CODE-15%. Su ventaja además viene mezclada con recetas que también aplican a una CNN: aumentación de ruido de red, etiquetas suaves y FiLM demográfico.
2. **ECG-FM, el candidato limpio, rindió peor que la ResNet en test oculto** cuando se usó congelado.
3. **La sonda lineal de ECG-FM en CV (0,377) no supera nuestro val (0,399).**
   - Los splits son distintos y el CV de ellos incluye SaMi-Trop.
   - Eso último debería *inflarlo*, no deprimirlo.
4. **Las CV que mezclan SaMi-Trop/PTB-XL no son comparables con nuestra arena A.** El 0,490 del ganador no es un número a perseguir.

---

## 4. Ranking y recomendación

1. **ECG-FM, checkpoint `mimic_iv_ecg_finetuned.pt`. Es el transformer a probar.**
   - Transformer con MIT.
   - Sin contaminación de arena A ni de SaMi-Trop.
   - La entrada coincide casi exacto con nuestra Fase 2: z-score por derivación, orden canónico; solo cambia 400 → 500 Hz.
   - Su espacio de features ya fue empujado a distinguir BRD, bloqueo bifascicular, extrasístole ventricular e infarto, que son los tres patrones del ROADMAP.
   - Cabe holgado en 12–16 GB.
   - En contra: depende de fairseq-signals, #416 vio que descongelarlo empeoraba y los benchmarks lo dejan 4º.
   - Conviene probar los dos checkpoints. El *pretrained* (solo SSL) es más genérico; el *finetuned* trae los patrones clínicos. **[I]** Mi apuesta es el *finetuned*.
2. **ECGFounder, como control no-transformer y quizá como mejor apuesta real.**
   - Sin contaminación conocida y con MIT.
   - Supervisado sobre 150 etiquetas, BRD, extrasístole e infartos entre ellas.
   - Es top-3 en el benchmark independiente.
   - Es PyTorch plano, el más barato de integrar.
   - **[I]** Si el objetivo real es "mejor TPR@5%" y no "usar un transformer", es el primero que correría. Lo dejo segundo solo porque el pedido es un transformer.
3. **No usar:** HuBERT-ECG, ST-MEM (original o del ganador), ECG-JEPA ni xECG. Todos vieron CODE-15% o CODE en el preentrenamiento.
   - Vieron registros de *val* además de los de test, así que contaminan también la selección de modelo, no solo la medición final.
   - HuBERT-ECG y ST-MEM además son no comerciales.

---

## 5. Plan de ajuste fino para ECG-FM (boceto, sin ejecutar)

### 5.0 Gate de entorno (½ día) [I]
- Instalar fairseq-signals en el box Linux, en un venv **separado** del `.venv` del servicio. El `requirements.txt` fija torch CPU y no hay que tocarlo.
- Cargar `mimic_iv_ecg_finetuned.pt` y reproducir la salida del notebook `infer_quickstart.ipynb`.
- **Prueba de cordura con nuestros datos:** correr la cabeza "Right bundle branch block" de ECG-FM sobre val CODE-15% y medir AUC contra `rbbb_label`.
  - Si no da ≥ 0,95, la adaptación de entrada está mal: orden de derivaciones, frecuencia o escala.
  - Nuestra cabeza actual da 0,98.
- Si fairseq resulta inmanejable, portar el forward a PyTorch puro y validar contra el original (diferencia máxima de logits < 1e-4).

### 5.1 Adaptación de la entrada
- Partir de `fase2_preprocessed.hdf5` `(2800, 12)` a 400 Hz con z-score. **No regenerar la Fase 2 ni el split.** `split_congelado.parquet` sigue mandando.
- Transformar al vuelo en el `Dataset`, en este orden:
  1. Transponer a `(12, 2800)`.
  2. Remuestrear 400 → 500 Hz con interpolación lineal (la misma que fairseq-signals) o `resample_poly(5, 4)`, lo que da `(12, 3500)`.
  3. Volver a aplicar z-score por derivación. Es casi un no-op, pero deja la convención idéntica a la de ECG-FM.
- Alternativas de longitud:
  - **(A)** Pasar los 7 s completos (3500 muestras, ≈219 tokens). El posicional relativo lo admite y #416 usó 10 s.
  - **(B)** Usar dos recortes de 5 s, `[0:2500]` y `[1000:3500]`, y promediar logits.
  - Probar (A) primero.
- El orden de derivaciones ya es `I, II, III, aVR, aVL, aVF, V1–V6` y coincide con el `LEAD_ORDER` de fairseq-signals. No hay que reordenar nada.
  - **Trampa conocida:** si más adelante se suma SaMi-Trop LVSD, su orden no es el canónico y viene a 300 Hz (ver CLAUDE.md).

### 5.2 Modelo y cabezas
- Backbone: encoder de ECG-FM. Pooling por media de los tokens de salida, o promedio de capas como hizo el 1º puesto.
- Las cabezas replican la interfaz de `model.ResNet1D.forward`, que devuelve `(logit_chagas, logit_rbbb, patrones)`. Así `train.py` y `evaluar.py` no cambian:
  - `cabeza_chagas = Linear(768, 1)`
  - `cabeza_rbbb = Linear(768, 1)`
  - `cabeza_patrones = Linear(768, 3)`
- Sin demográficos en la primera corrida, para comparar contra `patrones-lr8`, que no los usa.
- **Opcional [I]:** concatenar los 17 logits de la cabeza MIMIC original como features adicionales. Es un tipo de destilación, en la línea del hallazgo del 4º puesto.

### 5.3 Pérdida: reutilizar sin cambios
- BCE de Chagas enmascarada con `mask_chagas` y ponderada (`peso * mask_chagas`), con `pos_weight_chagas` calculado sobre train.
- BCE de BRD enmascarada con `rbbb_mask`, que solo es 1 en CODE-15%.
- BCE de patrones enmascarada con `mask_pat`.
- Los mismos pesos relativos de las cabezas que en `patrones-lr8`.
- PTB-XL queda fuera de train, como hoy. Recién en una segunda iteración probar etiquetas suaves 0,8/0,2 solo en CODE-15%, que es la receta del 1º y 2º puesto.

### 5.4 Protocolo de entrenamiento (en dos pasos, el primero es un gate barato)
1. **Sonda lineal.**
   - Extraer embeddings de 768 dimensiones una sola vez para train/val (~238k × 768 en fp16 ≈ 0,4 GB) y entrenar solo las cabezas.
   - Toma minutos de GPU por corrida una vez cacheados.
   - Hacerlo con los dos checkpoints (pretrained y finetuned) y con ECGFounder como control (su penúltima capa).
   - **Si ninguna sonda llega a ~35% de TPR@5% en val arena A, no seguir al ajuste completo** [I]. Como referencia, la sonda del ganador quedó 11 puntos por debajo de su ajuste completo.
2. **Ajuste completo**, solo si el paso 1 pasa.
   - Primero ~2k iteraciones con el backbone congelado y después todo descongelado.
   - Tasas de aprendizaje: backbone 1e-6 a 1e-5 con *layer-wise decay* 0,75; cabezas 1e-3.
   - AdamW, AMP bf16, batch 64 (acumular a 256 si hace falta), 6–8 épocas.
   - Mejor checkpoint por AUPRC de arena A en val, igual que `train.py`.
   - **[I] VRAM:** 90,9M parámetros con AdamW en fp32 son ≈1,5 GB de estado. Con ≈219 tokens × 12 capas y batch 64 en bf16, las activaciones deberían quedar en unos pocos GB. Entra en 12 GB (laptop) y en 16 GB (box).
   - **[I] Tiempo:** estimo 30–60 min por época sobre ~238k registros en la 4080 SUPER. Hay que medirlo con el smoke run (`--limit-train 500 --limit-val 2000 --epocas 1`).
   - Alternativa si #416 se repite (descongelar empeora): **LoRA** r=8 sobre q/v de las 12 capas (`peft`), unos 0,3M parámetros entrenables.
3. **Semillas:** como mínimo 2 por configuración, igual que se hizo con `ctrl-lr8` y `patrones-lr8`. Sin eso la diferencia no es interpretable (lección del 2026-08-28).

### 5.5 Cómo se juzga (solo validación)
- **Validación, arena A, a nivel paciente. Nunca test.** El test se midió una vez el 2026-09-14. Volver a mirarlo es decisión de Axel, no un paso del experimento.
- Métrica primaria: **TPR@5%** (`evaluar.tpr_a_capacidad`). Secundarias: AUPRC y AUC.
- Referencia: `patrones-lr8` en val, con **39,9% / 0,1798 / 0,8314**, y la dispersión entre semillas de los controles.
- Diagnósticos obligatorios además de la métrica:
  - **Atajo de fuente:** AUC pooled menos AUC de arena A. No debe crecer contra el control.
  - **Sensibilidad de la banda alta estratificada por BRD anotado.** Hoy es 73,3% con BRD y 10,4% sin él. **Es la pregunta que de verdad importa: si el FM encuentra positivos sin BRD.** Si mejora el TPR@5% solo detectando mejor BRD, no cambia el carácter del modelo.
  - Arena A-cardiopatía y disparidad por sexo (mujeres 18,5% contra varones 27,3% en test).
  - **Arena C (PTB-XL) se reporta, marcada como contaminada** por el preentrenamiento de ECG-FM.
- **Criterio de éxito pre-registrado** [I, a fijar con Axel antes de correr]:
  - Media de 2 semillas ≥ +3 puntos de TPR@5% sobre el control en val arena A.
  - IC bootstrap pareado por paciente que excluya 0.
  - Sensibilidad en positivos sin BRD que suba de forma visible.
  - Si no se cumple, se registra como la palanca nº 10, negativa, y se borra el código (memoria del proyecto: "experimento fallido: borrar código, dejar registro").

---

## 6. Riesgos y si es probable superar el techo

**Por qué probablemente no mueva el techo** [I, apoyado en hechos]
1. **El techo no es de capacidad.**
   - FASES (2026-09-16) midió que el 82,7% de los perdidos relevantes no tiene ninguno de los 6 diagnósticos de CODE-15%.
   - El ruido de etiqueta (serología sin cardiopatía) explica el 20%.
   - Wen & Kang (*Physiol Meas* 2026) [R] llegan independientemente a "un plateau impuesto por la fisiopatología": la forma indeterminada no se distingue de los controles.
   - Un backbone más expresivo no crea señal que no está en el trazado.
2. **La evidencia de FM en Chagas es floja.**
   - El único FM que superó a la mejor CNN en test oculto lo hizo por +4 puntos, contaminado y con recetas que no son del backbone.
   - El FM limpio (ECG-FM) rindió por debajo de la CNN.
3. **Todos los equipos cayeron de 0,40–0,55 interno a 0,20–0,32 externo** [R]. Una mejora en val arena A no garantiza nada fuera de Brasil, y Argentina (Fase 6) es justamente "fuera".
4. **Ruido de medición.** El IC de TPR@5% en test fue [33,9 – 41,0], unos ±3,5 puntos. Una mejora de 1–2 puntos no se distingue.

**Dónde sí podría ayudar** [I]
- **Representación de cardiopatía.** ECG-FM predice FEVI ≤ 40% con AUROC 0,929, aunque esa cabeza no es pública, y su checkpoint MIMIC trae "Infarction" y "Bifascicular block".
- Si algo de la señal de los perdidos es miocardiopatía sutil (zonas eléctricamente inactivas, bajo voltaje regional), un FM preentrenado en 1,4M ECG la puede codificar mejor que una ResNet entrenada solo con 238k registros de etiqueta débil.
- **Lo más informativo y barato** es la sonda lineal estratificada por BRD: dice en horas si hay señal nueva, antes de pagar un ajuste completo.
- **Uso alternativo sin reemplazar nada:** validar contra SaMi-Trop LVSD (FEVI por eco). Eso exige cuidar el orden de derivaciones y los 300 Hz.

**Riesgos concretos**

| Riesgo | Detalle | Mitigación |
|---|---|---|
| Contaminación | ECG-FM vio PTB-XL; los descartados vieron CODE-15% (val y test) | Usar ECG-FM o ECGFounder; arena C marcada como contaminada |
| Dependencia | fairseq-signals es un fork viejo de fairseq, hydra y torch antiguo, probablemente sin Windows | Venv separado en Linux, o portar el forward a PyTorch puro y validar |
| Descongelar empeora | #416 lo midió con ECG-FM | Gate de sonda lineal, LR muy bajo, LoRA |
| Servicio | 90,9M parámetros contra la CNN actual; el servicio corre en CPU (torch CPU fijado) | Medir latencia en CPU antes de considerar un reemplazo en producto |
| Congelamiento | El producto está congelado en `patrones-lr8/mejor.pt` con test medido | Cualquier reemplazo invalida la Síntesis de Fase 5 y requiere decidir si se mira test otra vez |
| Licencias | HuBERT-ECG y ST-MEM son CC BY-NC; xECG "no para uso clínico ni comercial" | Solo ECG-FM (MIT) y ECGFounder (MIT) son elegibles para un producto |
| Selección por val | El ranking de val no predijo el de test en el Challenge (ISIBrno: mejor val, 4º en test) | Criterio pre-registrado; no iterar 20 variantes contra val |

**Conclusión** [I]
- Probar ECG-FM es razonable como **un experimento acotado**: sonda lineal primero, con criterio pre-registrado y estratificado por BRD.
- No es razonable como "reemplazo" con la expectativa de romper el techo; la evidencia disponible no lo sostiene.
- Si se corre, **ECGFounder debe ir en paralelo como control**: es más barato, no tiene contaminación y en benchmarks independientes rinde igual o mejor. Si le gana a ECG-FM, el pedido de "un transformer" pierde su justificación técnica.

---

## 7. Fuentes principales (todas consultadas el 2026-10-22)

**Modelos**
- ECG-FM, paper: https://arxiv.org/html/2408.05178
- ECG-FM, versión JAMIA Open: https://academic.oup.com/jamiaopen/article/8/5/ooaf122/8287827
- ECG-FM, código: https://github.com/bowang-lab/ECG-FM
- ECG-FM, pesos: https://huggingface.co/wanglab/ecg-fm
- fairseq-signals: https://github.com/Jwoo5/fairseq-signals (`scripts/preprocess/ecg/preprocess.py`)
- ECGFounder, paper: https://arxiv.org/html/2410.04133
- ECGFounder, código: https://github.com/PKUDigitalHealth/ECGFounder
- ECGFounder, pesos: https://huggingface.co/PKUDigitalHealth/ECGFounder
- HuBERT-ECG, preprint: https://www.medrxiv.org/content/10.1101/2024.11.14.24317328
- HuBERT-ECG, código: https://github.com/Edoar-do/HuBERT-ECG
- HuBERT-ECG, pesos: https://huggingface.co/Edoardo-Coppola/hubert-ecg-base
- ST-MEM: https://arxiv.org/html/2402.09450 y https://github.com/bakqui/ST-MEM
- ECG-JEPA: https://arxiv.org/html/2410.08559 y https://github.com/sehunfromdaegu/ECG_JEPA
- xECG / BenchECG: https://arxiv.org/html/2509.10151v1 y https://github.com/dlaskalab/bench-xecg

**Benchmarks**
- Benchmark de FMs (ICLR 2026): https://arxiv.org/html/2509.25095
- Estrategias de preentrenamiento: https://arxiv.org/html/2605.12241v1

**Challenge 2025**
- Resultados: https://moody-challenge.physionet.org/2025/results/challenge_2025_results.tsv
- 1º Biomed-Cardio: https://cinc.org/archives/2025/pdf/CinC2025-019.pdf
- FM del 1º puesto: https://arxiv.org/html/2509.00102v3 y https://github.com/Xuanphu108/ecg_ssl
- #416 ECG-FM: https://cinc.org/archives/2025/pdf/CinC2025-416.pdf
- #236 DlaskaLabMUI: https://cinc.org/archives/2025/pdf/CinC2025-236.pdf
- #160 SwissBeatsNet: https://cinc.org/archives/2025/pdf/CinC2025-160.pdf
- #197 HuBERT: https://cinc.org/archives/2025/pdf/CinC2025-197.pdf
- #259 ChagasExplorers: https://cinc.org/2025/Program/accepted/259.html
- #219 Seoul Mates: https://cinc.org/2025/Program/accepted/219.html
- Ahus AIM: https://arxiv.org/abs/2604.09782
- AnyECG: https://arxiv.org/abs/2601.10748
