# Investigación bibliográfica y revisión cardiológica

*2026-10-14.* Lo escribieron agentes de búsqueda web y no se verificó cita por cita. Cada sección marca qué cifra está verificada contra la fuente, cuál viene de un abstract o de una fuente secundaria, y cuál no se pudo verificar. Antes de citar un número en otro documento, revisá esa marca.

Cómo se hizo: tres agentes relevaron literatura (clínica del ECG en Chagas, IA/ECG para Chagas y epidemiología y tamizaje en Argentina). Después, un agente con rol de cardiólogo leyó esos tres informes, FASES.md y el código, y propuso qué cambiar. Fue una revisión de solo lectura: no se reentrenó nada y no se volvió a medir test.

## Resumen

**Conclusión.** Contra serología, el modelo no puede subir más. Su AUC en test (0,835) está en el estado del arte, y ningún trabajo publicado supera AUC 0,85 o TPR@5% 0,40 contra serología en un test externo. En un estudio prospectivo de 59.154 personas en Brasil (Ribeiro, *Circulation* 2026), el ECG con IA solo dio AUC 0,645; sumándole tres preguntas epidemiológicas llegó a 0,776. Lo que falta está fuera del modelo.

**Por qué hay techo (literatura clínica).** Alrededor del 49% de los seropositivos de REDS-II tiene ECG normal. La forma indeterminada de Chagas no se ve en el ECG por definición. Fuera del BRD (OR 4,6), las demás alteraciones discriminan poco: HBAI, extrasístoles ventriculares, BAV de 1er grado y FA quedan en OR de 1,6 a 2,1. Bajo voltaje, potenciales tardíos y ST-T aislado no discriminan.

**Las 5 propuestas del cardiólogo.** Ninguna cambia el score ni la banda, así que no requieren reentrenar ni volver a mirar test.
1. **Preguntas epidemiológicas en el servicio** (`servidor.py`, `inferencia.py`). Campos opcionales; si alguno da positivo, la respuesta dice "serología indicada por guía". No es lo mismo que `--con-demograficos`: no entra al modelo.
2. **Prueba de robustez del score sobre validación:** red de 50 Hz contra 60 Hz, filtros de los equipos y frecuencia de muestreo.
3. **Explicación clínica** (`calibrar_servicio.py`, `inferencia.py`): patrones como presente o ausente con umbral calibrado en validación, la combinación BRD+HBAI con nombre, "alta sin patrón reconocible" cuando corresponda, y la aclaración de que la cabeza de HBAI no está validada en pacientes chagásicos.
4. **Seguridad del mensaje.** Marcapasos devuelto como "no interpretable". Valor predictivo calculado con la prevalencia local en vez del "30 de cada 100" fijo. Aviso en `no_alta` de que con síntomas o con BRD informado la serología corresponde igual. Un nivel de "calidad dudosa".
5. **Auditoría de sexo y edad** con `--solo-analisis` sobre el parquet de test ya generado: LR+ y LR− por grupo, y la disparidad declarada en `/contrato`.

**Lo que desaconseja.** Pérdida de ranking: es una variante de `hardneg-v1`, que dio −6,1 pp. Etiquetas suaves: obligan a reentrenar, y el experimento con `peso_strong` apunta en contra. SaMi-Trop LVSD como positivos contra CODE-15%: reabre el atajo de reconocer la fuente; propone usarlo solo como banco de evaluación según la fracción de eyección.

**Fase 6.** Trabajar con el Hospital Eva Perón y el INP Fatala Chabén, que tienen la cohorte de Viotti y Laucella. No existe ningún dataset público argentino con ECG y serología.

---

## Contenido

- [Parte 1 — Revisión cardiológica](#parte-1--revisión-cardiológica)
- [Anexo A — Clínica del ECG en Chagas](#anexo-a--clínica-del-ecg-en-chagas)
- [Anexo B — IA/ECG para Chagas](#anexo-b--iaecg-para-chagas)
- [Anexo C — Epidemiología y tamizaje en Argentina](#anexo-c--epidemiología-y-tamizaje-en-argentina)


---

# Parte 1 — Revisión cardiológica

## DECA: revisión clínica y técnica desde la cardiología

*Revisión hecha desde la mirada de un electrofisiólogo con experiencia en Chagas y en IA sobre ECG, del tipo que trabajó con las cohortes SaMi-Trop y REDS-II. Fecha: 2026-10-14.*

**Alcance.** Es una revisión de solo lectura: no se modificó ningún archivo del repo, no se entrenó nada y no se corrió `evaluar_test.py`.

**Fuentes.**
- La literatura que reunieron otros agentes, en el scratchpad: `lit_clinica_ecg.md`, `lit_ia_ecg.md` y `lit_epidemiologia.md`.
- La documentación del repo: `FASES.md` completo (de la Fase 3 hasta la sesión del 2026-10-01), `CLAUDE.md`, `ROADMAP.md`, `VALIDACION-CLINICA.md` y `API.md`.
- El código de entrenamiento: `model.py`, `train.py`, `dataset.py`, `analisis_errores.py` y `validar_patrones_samitrop.py`.
- El código de servicio: `main.py`, `servidor.py`, `inferencia.py` y `models/patrones-lr8/calibracion.json`.

**Marcas que uso.**
- **[FASES §x]** remite a la sección de `FASES.md` donde algo ya se midió.
- **[lit-clin]**, **[lit-IA]** y **[lit-epi]** remiten a los tres archivos de literatura.
- **[cálculo propio]** es aritmética mía sobre números ya publicados en el repo. No hay mediciones nuevas.

---

### 1. Lectura clínica del modelo actual

#### 1.1 Qué es, visto desde la consulta

`patrones-lr8` deriva a la banda alta al **1,41 %** de los ECG. De esos, **29,5 %** son seropositivos, contra 1 de cada 52 en la población general. El **LR+ es 21,5 [18,0–25,6]** y la sensibilidad es **21,8 %** (test, arena A [FASES Fase 5; VALIDACION-CLINICA §2]).

Para un cardiólogo, un LR+ de 21 coloca a la banda alta en la categoría de "hallazgo que cambia la conducta": convierte una prevalencia de 2 % en una de 30 %. La ley ya obliga a testear a toda embarazada con una positividad de alrededor de 1 % [lit-epi §4]. **Cualquier persona en banda alta tiene una probabilidad pretest 30 veces mayor que la de una gestante a la que ya testeamos por ley.** Como *rule-in* oportunista, la derivación está plenamente justificada.

#### 1.2 Lo que hace bien

1. **Lee lo que leería un cardiólogo, y lo lee mejor que la regla clínica.**
   - La banda alta es operativamente un detector de BRD: encuentra al 73,3 % de los positivos con BRD anotado y al 10,4 % de los que no lo tienen [FASES 2026-09-16].
   - Aun así le gana a la regla "todo BRD → serología": deriva al 1,41 % contra el 2,45 %, con VPP de 29,5 % contra 14,1 % y LR+ de 21,5 contra 8,4 [FASES 2026-09-23].
   - La razón es que dentro del BRD discrimina: deriva al 73 % de los BRD chagásicos y al 27 % de los no chagásicos. Es coherente con lo clínico. El BRD tiene el OR más alto del metaanálisis (4,60) [lit-clin §1.2], pero P(Chagas | BRD) es solo 13,8 % en CODE-15% [FASES §4, 2026-08-26]. Distinguir el BRD chagásico del BRD hipertensivo o isquémico es justamente lo que un clínico no puede hacer a ojo.
2. **La capa de explicación es honesta.**
   - La cabeza de BRD transfiere sin pérdida a chagásicos reales: AUC 0,9808 en val y 0,9797 en SaMi-Trop [FASES 2026-09-10 §7]. En test da 0,9838.
   - La cabeza `hbai` da 0,96 en test y `extra` 0,90.
   - `zona` (onda Q / zona inactiva) no transfiere (0,52 en chagásicos) y ya se sacó del servicio. Fue la decisión correcta.
3. **No hay sobreajuste ni atajo de fuente.** El AUC pasó de 0,8314 en val a 0,8348 en test, y el diagnóstico de atajo dio −0,042.

#### 1.3 Lo que hace mal, o lo que no puede hacer

1. **No ve la forma indeterminada, y nadie la ve.**
   - Por definición, el estadio A es ECG normal [lit-clin §1.1].
   - Entre seropositivos no seleccionados, del 49 % (REDS-II) al 86 % (Chaco boliviano) tienen ECG normal [lit-clin §1.3, §1.8].
   - La Guía nacional 2018 ubica al 70 % de los infectados en estadio 0 de Kuschnir [FASES 2026-09-23].
   - Que se pierda el ~78 % de los casos es en gran parte una propiedad de la enfermedad. Pero FASES corrigió algo importante el 2026-09-16: **solo el 20 % de los falsos negativos tiene ECG normal y ninguna cabeza encendida**. El resto tiene un ECG "anormal" que no corresponde a ninguno de los 6 diagnósticos de CODE-15% (82,7 % de los huérfanos).
   - Mi lectura clínica de ese 82,7 % es que son, con alta probabilidad, alteraciones inespecíficas que la literatura declara no discriminantes:
     - ST-T aislado (11,3 % contra 13,0 % en Bambuí);
     - bajo voltaje (OR 0,84);
     - BRD incompleto (OR 0,85);
     - eje entre 0 y −30°, y extrasístoles aisladas [lit-clin §0.3, §1.9].
   - Es decir: el ECG tiene *algo*, pero ese algo es lo mismo que tiene el 60 % de los seronegativos de CODE-15% [FASES 2026-09-10 §4].
2. **No gradúa el daño.**
   - El score no discrimina mortalidad dentro de seropositivos: AUC 0,457 [FASES 2026-08-27 §4].
   - Tampoco discrimina disfunción del VI con FEVI ≤ 40: AUC 0,614 [FASES 2026-09-10 §7].
   - Es esperable. El BRD aislado tiene poco valor pronóstico [lit-clin §1.8, Consenso SAC] y ningún score pronóstico usa el BRD (Rassi usa bajo voltaje y TVNS; el score SaMi-Trop usa FC ≥ 80 y QRS ≥ 150) [lit-clin §4]. **Pronóstico y tamizaje usan señales distintas.** El modelo es de tamizaje.
3. **Disparidad por sexo, real y con mecanismo clínico.**
   - Sensibilidad de la banda alta: 18,5 % en mujeres contra 27,3 % en varones.
   - El LR+ es igual en ambos sexos (22,1 contra 21,4) y el AUC dentro de cada sexo es el mismo.
   - El mecanismo es la prevalencia de BRD entre positivos: 26,6 % en mujeres contra 35,9 % en varones [FASES 2026-09-16; VALIDACION-CLINICA §2].
   - La literatura lo respalda. El sexo masculino tiene OR 1,91 de anomalía mayor en SaMi-Trop, y BRD+HBAI es de 15,3 % en varones contra 9,7 % en mujeres. La progresión a cardiomiopatía también es mayor en varones (Basquiera, RR 5,0) [lit-clin §3; lit-epi §1.4].
   - **No es un sesgo del algoritmo en sentido estricto: es la epidemiología de la cardiopatía chagásica expresada en el punto de operación.** Pero el resultado para el sistema de salud es que se encuentran menos mujeres, y eso hay que declararlo.
   - Agrego algo que el repo no tiene: en Bambuí las mujeres son el 67,5 % de los infectados [lit-clin §3], y en las gestantes argentinas el tamizaje ya es universal. El peso de esta disparidad depende de a quién se le haga el ECG. Si la población de uso es mayoritariamente femenina (consultorios de atención primaria), la sensibilidad efectiva cae hacia 18 %.
4. **La edad mueve el LR+, no la sensibilidad.**
   - La fórmula de mezcla de VALIDACION-CLINICA §2 (FPR ≈ 0,274·r₀ + 0,0043·(1−r₀)) predice que en poblaciones más viejas, con más BRD no chagásico, el LR+ cae: con r₀ = 5 % baja a ~12.
   - Es coherente con la clínica. El exceso atribuible a Chagas es máximo entre los 25 y 44 años y se diluye después de los 64 por la comorbilidad (HTA, isquemia) [lit-clin §3].
   - El repo reporta la sensibilidad por década, pero **no el LR+ por edad** (ver propuesta P6).
5. **Lo que el ECG de 10 s no puede dar.**
   - Las extrasístoles ventriculares solo se ven si caen en la tira, y el modelo analiza **7 s centrados**.
   - La VFC, los potenciales tardíos y la variabilidad de la onda T requieren Holter, ECG de alta resolución o análisis latido a latido, y la evidencia de que sirvan para tamizaje es negativa o preliminar [lit-clin §2.3–2.6].
   - No hay nada que rescatar ahí con esta entrada.
6. **El marcapasos no está contemplado.**
   - Hay marcapasos en el 3,6–4,1 % de SaMi-Trop [lit-clin §1.4; FASES §7].
   - El protocolo de la Etapa 1 lo excluye, pero **el servicio no lo detecta**: puntúa un QRS estimulado como si fuera conducción propia, y su resultado no es interpretable. `plausibilidad.py` lo cuida solo en sentido negativo, es decir, se diseñó para *no rechazar* marcapasos [FASES 2026-09-28/29].

---

### 2. La literatura contra el código

#### 2.1 Lo que la literatura recomienda y DECA ya hace

| Recomendación | Dónde está en DECA |
|---|---|
| Evaluar por fuente y no por un pool que infle (DlaskaLabMUI: AUC 1,00 en SaMi-Trop+PTB-XL) [lit-IA §1.4] | Arenas A–D y diagnóstico de atajo [FASES Fase 4] |
| Usar TPR a capacidad fija como métrica operativa (Moody 2025) | `evaluar.tpr_a_capacidad` [FASES 2026-08-27 §5] |
| Separar serología de cardiopatía (Jidling: 0,68 → 0,82) | Arena A-cardiopatía y `--target`. Medido: +0,024 con el proxy `normal_ecg` [FASES §4, 2026-08-27] |
| Supervisar con patrones (BRD/HBAI/EV) para explicar | Cabezas `rbbb`, `hbai` y `extra`, validadas en test y, la de BRD, en chagásicos |
| Mostrar el resultado como *rule-in*, nunca como descarte | Salida `alta`/`no_alta`; `no_alta` informa casos perdidos y no VPN [FASES 2026-09-23] |
| Reportar LR y no solo VPP | VALIDACION-CLINICA §1 |
| Comparar contra la regla clínica | "todo BRD → serología" como comparador obligatorio |
| Validar externamente con STARD/TRIPOD+AI | VALIDACION-CLINICA §4 |
| Usar el mismo preprocesamiento en entrenamiento y servicio | `ventana.py` compartido; verificado con diferencia máxima de 1,65e-04 |
| Leer las derivaciones por nombre (la trampa de SaMi-Trop LVSD) | `lectura_ecg.py` más `derivaciones_permutadas` |

Es un nivel de rigor metodológico superior al de la mayoría de los papers del Challenge.

#### 2.2 Lo que la literatura recomienda y DECA no hace

| Recomendación | Evidencia | Estado en DECA |
|---|---|---|
| **Combinar el ECG con preguntas epidemiológicas** | Circulation 2026, prospectivo: ECG solo AUC 0,645; 3 preguntas solas 0,700; combinación **0,776** [lit-IA §2.2; lit-epi §6.2] | **No existe en ningún lado**: ni en el modelo, ni en el servicio, ni en el protocolo como combinación. VALIDACION-CLINICA §4 "registra" el riesgo epidemiológico pero no lo analiza combinado |
| Aumentación contra confusores de adquisición (red de 50/60 Hz, resampleo, filtros) | 1º puesto: la validación externa subió de 0,395 a 0,445 mientras la CV interna apenas se movía; #192 "spurious features" [lit-IA §1.3] | `train.py` **no tiene ninguna aumentación**. Además Argentina usa red de 50 Hz y el entrenamiento Chagas es brasileño, a 60 Hz |
| Etiquetas suaves o *label smoothing* solo en CODE-15% | Puestos 1º, 2º y 3º [lit-IA §4.1] | **No medido.** Ver P9 |
| Ranking loss con *hard negatives* | 3º puesto: de 0,354 a 0,388 en su set interno [lit-IA §1.3] | Variante de algo ya descartado. Ver P9 |
| Fundación SSL (ST-MEM, MAE) | Puestos 1º y 2º | No hecho. La ventaja externa frente a una buena CNN es chica (0,323 contra 0,280) [lit-IA §3.3] |
| Reportar VPP local según la prevalencia del sitio | VALIDACION-CLINICA §1 lo pide | **El texto del servicio sigue diciendo "cerca de 30 de cada 100"** (`calibracion.json`, banda `alta`), que es cierto solo con 1,9 % de prevalencia |

#### 2.3 Lo que DECA ya descartó con su propia medición

Es la lista que no hay que volver a proponer.

| Palanca | Resultado | Sección |
|---|---|---|
| `peso_strong` (peso de la serología frente al autorreporte) | Monótona: más peso, peor arena A | FASES Fase 4, 2026-08-13 |
| Oversampling | Memoriza: la loss baja a 0,03 y el AUC cae | 2026-08-26 |
| Scheduler de LR / 30 épocas | Bug real, pero arreglarlo da +0,002 y después sobreajusta | 2026-08-27 §12 |
| **Edad y sexo como entrada (`--con-demograficos`)** | AUPRC nulo (t = 0,04); **TPR@5% peor −0,99 pp (t = −3,55)**; el sexo solo da AUC 0,487 | 2026-08-27 §13b, §14 |
| Ensemble de 6 checkpoints | No sube el techo | 2026-09-08 §2 |
| Cabezas de patrón para mejorar Chagas | Exactamente nulo en un 2×2 con semillas | 2026-08-27 §18 |
| Target "cardiopatía" con el proxy `normal_ecg` | Nulo | 2026-09-10 §1 |
| **Hard negative mining** | **Negativo**: −6,1 pp de TPR@5%. La cabeza RBBB sube y la de Chagas baja: el BRD chagásico no se separa del isquémico en la señal | 2026-09-10 §2 |
| Más vocabulario de patrones (LBBB, FA, BAV1, …) | El 82,7 % de los huérfanos no tiene ninguno; la sensibilidad en esos dx ya es baja | 2026-09-16, addendum |
| Boosting / focal loss / XGBoost sobre la señal | Contraindicado por argumento: amplifica positivos inaprendibles | 2026-09-08 §1 |
| Multitarea Chagas+BRD+BAV1 (literatura) | Empeoró en #416 [lit-IA] | — |

La conclusión de FASES ("como ML está terminado") **es correcta y la comparto**. Ninguna propuesta de abajo pretende subir el AUC de Chagas contra serología. Las que tocan el modelo lo dicen explícitamente y van al final.

---

### 3. Propuestas concretas de código, priorizadas

**Criterio de prioridad.** Valor clínico por unidad de costo y sin riesgo para la regla de "test medido una vez".

**Restricción transversal.** VALIDACION-CLINICA §6 congela para la Etapa 1 el checkpoint (`b0e2ecfc…`), el umbral de 0,9301 y el preprocesamiento. **Todo lo que propongo en P1–P8 deja intactos el score y la banda**: agrega información alrededor y no cambia la prueba índice.

#### P1. Integrar el criterio epidemiológico en la decisión, sin tocar el modelo

**Archivos:** `src/servidor.py`, `src/inferencia.py`, `src/calibrar_servicio.py` (textos) y `API.md`. Del lado de DECA-Back, el formulario.

**Qué cambia.**
1. `/analizar` acepta un bloque **opcional** `contexto_epidemiologico` con:
   - `residio_zona_endemica`;
   - `vivienda_con_vinchuca` (reconoce la vinchuca o vivió en vivienda rancho);
   - `familiar_con_chagas` (incluida la madre);
   - `transfusion_antes_1995` (año aproximado);
   - `chagas_ya_diagnosticado`.

   Cada campo vale `si`, `no` o `no_sabe`. Son los criterios de sospecha de la Guía nacional 2012/2018 [lit-epi §3.1], y los tres primeros coinciden con las preguntas del AI-ECG-EPI [lit-epi §3.3].
2. El modelo **no los recibe**: el score y la banda no cambian. Se usan para tres cosas:
   - **Regla de guía, previa al modelo.** Si cualquiera de los criterios vale `si`, la respuesta agrega `"serologia_indicada_por_criterio_epidemiologico": true` y un texto: *"Esta persona tiene indicación de serología por criterio epidemiológico (Guía nacional), independientemente del ECG."* Esto vale con `alta` y también con `no_alta`.
   - **`chagas_ya_diagnosticado = si`.** La respuesta lo marca como fuera del uso previsto: el producto busca a quien no sabe, y así lo excluye también VALIDACION-CLINICA §4.
   - **Registro.** El backend guarda los campos para el análisis de la Etapa 1 (ver P1-bis).

**Justificación clínica y numérica** [cálculo propio con el LR de test].
- Con un pretest epidemiológico de 10 %:
  - `alta` lleva la probabilidad a **70 %**;
  - `no_alta` la lleva a **8,1 %**, que sigue muy por encima del ~1 % al que ya testeamos por ley.
- O sea: **para quien tiene criterio epidemiológico, el ECG no cambia la decisión.** La serología corresponde igual.
- **Donde el ECG agrega valor real es en quien no tiene o no conoce sus antecedentes.** El migrante interno urbano no sabe si su casa de infancia tenía vinchuca; La Matanza y La Plata son series con mucha cardiopatía al diagnóstico [lit-epi §1.2].
- Hoy el producto no distingue esos dos escenarios, y el texto de `no_alta` remite a la guía solo en prosa.
- Circulation 2026 muestra que la epidemiología sola (AUC 0,700) supera al ECG solo (0,645) y que la combinación llega a 0,776 [lit-IA §2.2]. Es **la brecha más clara de DECA frente al estado del arte prospectivo** [lit-epi §6.3].

**Estado frente a FASES: nueva.** No es una variante de `--con-demograficos`. Ese experimento metió **edad y sexo** en la red y salió nulo o negativo [§13b]. El aporte de Circulation viene de **variables epidemiológicas**, que no están en ningún dataset de entrenamiento (CODE-15%, SaMi-Trop y PTB-XL no tienen esas preguntas), así que **no se pueden entrenar ni medir hoy**. Por eso la propuesta es una regla de guía (válida sin datos) más la recolección de datos para medir la combinación en Argentina.

**Impacto esperado.** No mueve las métricas del modelo. Clínicamente:
- elimina el riesgo de que un médico lea `no_alta` como "no hace falta" en alguien con criterio epidemiológico, que es el error de seguridad más grave posible del producto;
- deja al producto alineado con la guía.

**Costo.** Bajo. Son unos 60 a 100 líneas en el servicio más el formulario del backend.

**Riesgo frente a la regla del test.** Ninguno: no cambia el score ni la banda.

**P1-bis.** En VALIDACION-CLINICA §4, convertir "se registra" en **resultado secundario preregistrado**:
- LR+ y LR− de las preguntas solas;
- un modelo combinado logístico (logit del ECG + preguntas) ajustado en una mitad del sitio y confirmado en la otra, o en un segundo sitio.

Nunca ajustar y confirmar sobre los mismos datos (VALIDACION-CLINICA §6). Es documento, no código, pero define qué guarda el backend.

#### P2. Explicación clínica binaria por patrón, calibrada en val, en lugar de sigmoides crudas

**Archivos:** `src/calibrar_servicio.py`, `src/inferencia.py` (`_patron`, `analizar`) y `API.md`.

**Qué cambia.**
1. Hoy `patrones` devuelve `{"brd": {"score": 0.94, "auprc_test": 0.80}, "hbai": {"score": 0.11, …}}`. Un médico va a leer "hbai 0,11" como "11 % de probabilidad de hemibloqueo". Es el mismo error que el proyecto evitó con el score de Chagas usando el percentil, repetido en las cabezas.
2. En `calibrar_servicio.py`, calcular sobre **val, arena A** un umbral por cabeza. Lo natural es el percentil 95 de esa cabeza entre los negativos, que es exactamente el criterio que ya usa `analisis_errores.PERCENTIL_NEGATIVOS`, así que no hay que inventar uno. También se puede usar un umbral con especificidad fija para el patrón. Para `rbbb`, `hbai` y `extra` se usan las etiquetas de patrón disponibles en val (CODE-15% RBBB, PTB-XL y Challenge 2021 para HBAI y EV).
3. La salida pasa a ser `"brd": {"presente": true, ...}`. Hay que agregar la **combinación nombrada**: si `brd` y `hbai` están presentes, emitir `"brd_mas_hbai": true` con el texto *"BRD + hemibloqueo anterior izquierdo: la combinación más característica de la miocardiopatía chagásica"* [lit-clin §1.9; ROADMAP].
4. En banda `alta` **sin ningún patrón presente**, que es una fracción grande, el texto tiene que decirlo: *"Prioridad alta sin un patrón nombrable: el modelo reconoce rasgos asociados que no corresponden a BRD, HBAI ni extrasístoles."*
   - [cálculo propio sobre las tasas por estrato de VALIDACION-CLINICA §2] De los ~491 pacientes en alta, ~289 tienen BRD anotado (59 %) y ~201 no lo tienen (41 %). El VPP es casi igual en ambos grupos (30 % contra 28 %).
   - Forzar una explicación en ese 41 % sería mentir. No dar ninguna haría que el médico desconfíe de los casos más interesantes.
5. Marcar `hbai` con `"validada_en_chagasicos": false`. Hoy figura `usable: true` aunque su validación en población chagásica sigue pendiente porque `V21` está rota [FASES SÍNTESIS, pendiente 5]. Para `extra`, agregar `"nota": "solo detecta extrasístoles presentes en los 7 s analizados"`.

**Justificación.** El producto es un priorizador que se justifica ante un clínico [FASES 2026-08-27 §11]. La explicación tiene que hablar el idioma del informe del ECG ("presenta BRD completo + HBAI"), no el de los scores.

**Estado: nueva** como producto. Reusa un criterio de umbral ya definido en `analisis_errores.py`.

**Impacto.** Interpretabilidad y confianza del médico. No mueve métricas.

**Costo.** Bajo a medio: una pasada de inferencia sobre val (43 s en GPU, o minutos en CPU) más el JSON.

**Riesgo frente al test.** Ninguno si se calibra **solo en val**. Los AUC de test que ya están en `calibracion.json` se pueden seguir mostrando porque son números ya publicados. No calibrar el umbral de patrón sobre test.

#### P3. Detección de ritmo de marcapasos como estado "no interpretable"

**Archivos:** `src/plausibilidad.py`, `src/inferencia.py` (`_preparar`, `CODIGOS`) y `API.md`.

**Qué cambia.** Se detectan espigas de estimulación: pendiente muy alta y duración menor a 2 ms en varias derivaciones, alineadas con QRS anchos. Hay que cuidar que el resampleo a 400 Hz las atenúa, así que la detección va sobre la **señal cruda** (`lectura.señal`), que `plausibilidad.evaluar` ya recibe.

El resultado **no se rechaza con 422**. Se devuelve el score con `"ritmo_estimulado": true` y el texto: *"Ritmo de marcapasos: el QRS estimulado no refleja la conducción propia y el resultado del modelo no es interpretable. Si la persona es de zona endémica y no tiene serología, solicitarla por criterio clínico."*

**Justificación.**
- En Chagas el marcapasos es frecuente: entre 3,6 y 4,1 % en SaMi-Trop [lit-clin §1.4]. En CODE el OR del marcapasos para Chagas autorreportado es de 13,3 [lit-clin §1.6].
- El protocolo de la Etapa 1 ya lo excluye, pero el servicio no lo sabe. El modelo puede puntuar un QRS estimulado como BRI y dejarlo en `no_alta`. Clínicamente, un portador de marcapasos sin serología de zona endémica es un candidato de altísima probabilidad pretest.

**Estado: nueva.**

**Impacto.** Seguridad en un subgrupo chico pero de alta prevalencia de Chagas.

**Costo.** Medio. Hay que medir el falso positivo sobre val completo como se hizo con `no_parece_ecg` (`validar_plausibilidad.py` ya tiene la infraestructura). PTB-XL trae la columna `pacemaker` como verdad de referencia [FASES 2026-08-26 §5].

**Riesgo frente al test.** Ninguno: es un flag y no cambia el score.

#### P4. Escalón de calidad dudosa y estabilidad de la banda, medidos en val

**Archivos:** `src/plausibilidad.py`, `src/inferencia.py` y `src/validar_plausibilidad.py` (para la medición). Opcionalmente un script nuevo, `src/estabilidad_banda.py`.

**Qué cambia.**
1. Hoy la calidad es binaria: se analiza o se rechaza. Hay registros que pasan pero tienen indicadores marginales:
   - `concentracion_qrs` entre 0,30 y ~0,40 (el cuantil 0,1 % de los aceptados va de 0,40 a 0,58 [FASES 2026-09-28/29]);
   - R² de miembros por debajo de 0,95 (1,4 % de val);
   - de 1 a 6 derivaciones planas;
   - derivaciones reconstruidas desde I y II.

   Propongo un campo `calidad.nivel ∈ {buena, dudosa}`, con texto, para esos casos.
2. **Estabilidad de la banda.** Además de la ventana centrada (la que decide), puntuar 3 a 5 ventanas de 7 s desplazadas dentro del registro y reportar `"banda_estable": true/false` (la fracción de ventanas que coinciden con la banda oficial). **La banda oficial no cambia**: sigue siendo la de la ventana centrada.
   - Un score de 0,93 justo en el umbral, que cambia de banda al mover la ventana 1 s, merece decírselo al médico: *"resultado en el límite"*.
   - Además, `extra` sobre todas las ventanas (máximo) da una explicación más completa de extrasístoles sin tocar la banda.
3. **Antes de escribir los textos**, medir en **val** el LR+ de la banda alta dentro de cada estrato de calidad dudosa y la tasa de inestabilidad de la banda. Si en los dudosos el LR+ se mantiene, el texto dice "calidad límite, resultado igualmente válido". Si cae, dice "interpretar con cautela".

**Justificación.** Un médico en un hospital del norte con un equipo viejo necesita saber si el resultado vale.

**Estado: nueva.** La plausibilidad existe, pero en forma binaria.

**Costo.** Medio. La inferencia multiventana multiplica el CPU por 3 a 5, de ~0,5 s a ~2 s por análisis; es aceptable.

**Riesgo frente al test.** Ninguno si la medición es sobre val. La banda oficial queda intacta.

#### P5. Prueba de estrés de dominio argentino (red de 50 Hz, filtros, frecuencia de muestreo) sobre val, antes de la Etapa 0

**Archivo:** script nuevo, `src/estres_dominio.py`, que no toca el servicio.

**Qué cambia.** Sobre los ECG de val de arena A (no test) se aplican perturbaciones realistas del parque argentino y se re-puntúa con el checkpoint congelado:
- interferencia de **50 Hz** con armónicos, a SNR de 15 a 30 dB (el entrenamiento Chagas es de Brasil, con red de 60 Hz);
- filtros de equipo de 0,05–40 Hz, 0,5–100 Hz y notch de 50 Hz;
- muestreo nativo de 250, 300, 500 y 1000 Hz con resampleo;
- cuantización de 5 µV/LSB;
- deriva de línea de base.

Se mide cuánto se mueven el AUC, el TPR@5% y, sobre todo, **la fracción en banda alta y el LR+ de la banda alta**, más el AUC de la cabeza BRD.

**Justificación.**
- Biomed-Cardio (1º puesto) midió que los confusores de adquisición mueven 5 puntos en datos externos y casi nada internamente [lit-IA §1.3].
- #192 mostró que el cociente 50/60 Hz por sí solo tiene señal de fuente.
- ISIBrno filtró a 1–47 Hz justamente porque "Brasil usa 60 Hz y Alemania 50 Hz".
- DECA midió que PTB-XL (50 Hz) se separa de CODE-15% por contenido de alta frecuencia (AUC 0,880). Descartó que fuera la red eléctrica específicamente, pero **nunca midió la robustez del score de Chagas a un ECG brasileño con ruido de 50 Hz** [FASES 2026-08-12].
- **Argentina es 50 Hz.** Si el modelo asoció en algún grado "contenido de alta frecuencia tipo PTB-XL" con negativo, los ECG argentinos podrían caer sistemáticamente en `no_alta`. El criterio 0.4 de la Etapa 0 lo detectaría en un solo sentido, el de los falsos altos. **No detecta una caída de sensibilidad, que es el fallo silencioso.**

**Estado: nueva.** Es una medición, no una palanca de entrenamiento. Ninguna de las nueve palancas fue de aumentación ni de robustez.

**Impacto.** Diagnóstico de riesgo. Si sale robusto, la Etapa 0 se encara con confianza. Si no, se puede evaluar en val una mitigación en inferencia, por ejemplo un pasabanda a 0,5–40 Hz aplicado igual a entrenamiento y servicio. Eso implicaría recalibrar sobre val **antes** de congelar para la Etapa 1, no después.

**Costo.** Bajo: es un script de inferencia sobre val y minutos de GPU.

**Riesgo frente al test.** Ninguno, porque es solo val. Si llevara a cambiar el preprocesamiento del servicio, el número de test del 2026-09-14 **dejaría de aplicar** al servicio y haría falta una nueva medición independiente. Por eso el cambio tiene que decidirse antes de la Etapa 1 y validarse en la Etapa 0/1, no en el test viejo.

#### P6. Auditoría estándar de equidad por sexo y por edad, y declaración en el contrato

**Archivos:** `src/analisis_errores.py` (modo `--solo-analisis`, sobre el parquet de test ya generado), `src/calibrar_servicio.py`, `src/servidor.py` (`/contrato`) y `API.md`. Opcional: `sexo`/`edad` en el pedido, **solo para registro**.

**Qué cambia.**
1. Una tabla fija por sexo y por franja etaria (<45, 45–64, ≥65) con: sensibilidad, FPR, **LR+ y LR−** con IC, prevalencia de BRD entre positivos y entre negativos, y la **descomposición del mecanismo**: qué sensibilidad tendrían las mujeres con la prevalencia de BRD de los varones, aplicando la fórmula de mezcla de VALIDACION-CLINICA §2.
   - Es aritmética sobre `test_scores_20260916-152548.parquet`, por la vía que CLAUDE.md habilita.
   - **El LR+ por edad no está reportado** y la literatura predice que cae en mayores (§1.3.4 de este informe).
   - [cálculo propio] El LR− por sexo da ~0,82 en mujeres y ~0,74 en varones: en las dos, `no_alta` casi no informa. Conviene publicarlo para cerrar la pregunta de si `no_alta` es más tranquilizador en algún grupo. No lo es.
2. En `/contrato` y en `calibracion.json`, un bloque `limitaciones_conocidas` con la disparidad y su mecanismo, para que el backend lo muestre en la ficha del modelo, no en cada resultado.
3. Si el backend manda `sexo` y `edad`, el servicio los **devuelve sin usarlos** y el backend los guarda. Esto permite **monitorear la disparidad en uso real** (Etapa 2, DECIDE-AI).
   - **No se usan como entrada del modelo**: está medido que empeora el TPR@5% [§13b].
   - **No se usan para umbrales por sexo**: ver la sección 4.

**Estado: nueva** como artefacto estándar. La disparidad ya está medida [FASES 2026-09-16] y el LR por sexo también [2026-09-23]. Lo nuevo es la edad, el LR−, la descomposición y la declaración formal.

**Costo.** Bajo.

**Riesgo frente al test.** Ninguno: es el parquet existente, descriptivo, y no se retoca nada.

**Una nota.** Circulation 2026 reportó **mejor** rendimiento en mujeres [lit-IA §2.2], en sentido contrario a DECA. No está verificado si es por el componente epidemiológico, por el umbral o por la población [NV]. Conviene leer el texto completo antes de comunicar la disparidad como algo general.

#### P7. VPP del texto parametrizado por la prevalencia local

**Archivos:** `src/calibrar_servicio.py` (textos), `src/inferencia.py` (texto de `alta`) y una variable de entorno `DECA_PREVALENCIA_SITIO`.

**Qué cambia.** El texto de `alta` deja de afirmar "cerca de 30 de cada 100" en cualquier contexto:
- Con prevalencia del sitio configurada, se muestra el VPP local = f(LR+ 21,5, prevalencia), con su IC.
- Sin ella, se muestra el LR ("multiplica por ~20 las chances") y un aclarado: *"medido en población brasileña con 1,9 % de prevalencia"*.

**Justificación.** El mismo modelo da un VPP de 17,8 % con prevalencia de 1 % y de 70,5 % con 10 % [VALIDACION-CLINICA §1]. En Argentina va de ~1 % (gestantes) a 20–30 % (Pampa del Indio) [lit-epi §1.2]. VALIDACION-CLINICA ya lo identifica, pero el servicio no lo implementó.

**Estado: nueva** en código. Ya está identificada en el documento.

**Costo.** Muy bajo.

**Riesgo frente al test.** Ninguno: usa el LR ya publicado. VALIDACION-CLINICA §6 permite explícitamente cambiar los textos.

#### P8. SaMi-Trop LVSD: usarlo como banco de evaluación clínica, no como target de tamizaje

**Archivos:** `src/validar_patrones_samitrop.py`, que ya carga correctamente el dataset (reordena las derivaciones por nombre, resamplea 300 → 400 Hz, filtra registros incompletos o vacíos y toma el primer registro con las 12 derivaciones de igual largo y ≥ 7 s; 992 exámenes). Para el sondeo opcional, un script nuevo.

**Qué propongo, y qué no.**

1. **Sí, a costo bajo, como evaluación descriptiva y no como entrenamiento.** Extender `validar_patrones_samitrop.py` para reportar, dentro de chagásicos confirmados:
   - **sensibilidad de la banda alta estratificada por FEVI** (≤ 40, 41–49, ≥ 50) y por presencia de los hallazgos anotados (V20 BRD, V17 QRS ≥ 120, V18 Q, V19 ST-T, V26 FA);
   - el **porcentaje de pacientes con FEVI ≤ 40 que caen en `no_alta`**.

   La pregunta clínica es concreta: *¿a cuántos chagásicos con miocardiopatía dilatada no prioriza el modelo?* Si el número es alto, el texto de `no_alta` debería mencionar que una persona con síntomas de insuficiencia cardíaca requiere evaluación igual. Es descriptivo, no reemplaza a la arena B (que es SaMi-Trop original) y no toca test.

2. **Con cautela: una cabeza o sonda lineal de disfunción del VI** sobre el backbone congelado. Sería una regresión logística sobre los 3.200 features, con validación cruzada anidada por paciente. Notas:
   - El tamaño: **63 eventos** con FEVI ≤ 40 entre los 992 utilizables [FASES §7], o 93 en el total de 1.304. Con 63 eventos el IC del AUC va a ser de ±0,07 aproximadamente.
   - La literatura: Brito 2021 obtuvo AUC 0,84 **sin reentrenar en Chagas** (algoritmo de Attia), y Cardoso 2026 0,89 con fine-tuning en REDS-II [lit-IA §5].
   - **Responde otra pregunta**: priorizar ecocardiograma en un seropositivo conocido, que es otro producto (estadio B2). En Argentina sí sería valioso: el 54,6 % de los pacientes con miocardiopatía chagásica en varias instituciones no tenía completos el ECG, la radiografía y el eco [lit-epi §1.3, Chuit 2023].
   - **Estado frente a FASES: nueva.** El `--target cardiopatia-mask` usaba el proxy `normal_ecg`, no la FEVI.
   - **Impacto sobre el tamizaje de DECA: ninguno.** No es un camino para subir la detección de serología.

3. **No.** No usar LVSD como "positivo de cardiopatía" contra negativos de CODE-15% para entrenar la cabeza de Chagas:
   - Es otra exportación (300 Hz, otro formato, registros por grupos) contra otra fuente, o sea que **recrea el atajo de fuente** que el proyecto evitó desde el 2026-08-12.
   - Con ~60 a 90 positivos estructurales no hay potencia.
   - Las tres trampas (orden `DI,DII,DIII,AVL,AVF,AVR`, 300 Hz y filas vacías) ya están resueltas en el loader existente. **Reusarlo; no escribir otro.**

**Riesgo frente al test.** Ninguno en (1). En (2), es un artefacto nuevo que no toca el checkpoint de Chagas, pero si se lo quisiera mostrar en el servicio necesitaría su propia validación externa.

#### P9. Etiquetas suaves, *label smoothing* y ranking loss: qué dice FASES, y mi recomendación

**Estado.**
- **Etiquetas suaves / label smoothing solo en CODE-15%: NO medido** (`grep` sin resultados en `src/` ni en `FASES.md`). Pero hay **evidencia indirecta en contra**: el barrido de `peso_strong` cambiaba la confianza relativa entre etiquetas fuertes y débiles y dio resultados monótonamente peores al restarle peso relativo a CODE-15%. El target `cardiopatia-mask` (sacar positivos dudosos) dio nulo. El smoothing es otro mecanismo (regulariza la confianza en vez de reponderar), así que es una **variante cercana, no un duplicado**.
- **Ranking loss contra hard negatives (AIChagas): variante de algo ya descartado.** `hardneg-v1` concentró el entrenamiento en los mismos negativos difíciles y bajó 6,1 pp, con una firma inequívoca: **el BRD chagásico no se separa del no chagásico en la señal** [FASES 2026-09-10 §2]. Un margin ranking contra el 20 % superior del batch apunta exactamente a esos negativos. Además, #416 encontró que una pérdida de ranking orientada a TPR@5% fue peor que BCE [lit-IA]. FASES pide una hipótesis nueva antes de repetirlo, y yo no tengo una.
- La parte de AIChagas de **subir la etiqueta de los negativos con BRD (+0,15)** reforzaría la dependencia del BRD, que ya es el 80 % del modelo, y subiría el FPR entre los BRD no chagásicos, que es justo donde el modelo hoy discrimina (LR 2,7). Clínicamente va en la dirección equivocada.

**Por qué esta vez podría ser distinto, siendo justos.** En el Challenge, la ganancia de estas técnicas apareció en el **test externo** y no en la CV interna (en Biomed-Cardio los ajustes de robustez movieron +5 pts afuera y +2,7 % adentro) [lit-IA §1.3]. DECA solo puede medir en su arena interna, CODE-15%. Un nulo interno no prueba un nulo externo.

**Recomendación: no ahora.**
1. Reentrenar invalida el checkpoint congelado para la Etapa 1.
2. La única forma honesta de ver el beneficio es en datos externos, y esos son los argentinos que todavía no existen.

Si alguna vez se arma un "v2", el diseño es: label smoothing ε = 0,1–0,2 solo en CODE-15% más la aumentación de P5, con 2 semillas y medias de las últimas 10 épocas en val (la regla de FASES §13). Se juzga **en datos de la Etapa 0/1 reservados como confirmación**, no en el test de 2026-09-14, que ya está gastado.

**Costo.** Alto en gestión del congelamiento y bajo en GPU.

#### P10. Menor: criterio clínico explícito en el texto de `no_alta`

**Archivos:** solo textos (`calibrar_servicio.py`).

Agregar al texto de `no_alta`: *"Si hay síntomas cardiovasculares (palpitaciones, síncope, disnea) o el informe del ECG describe BRD, hemibloqueo o extrasístoles, la serología está indicada por criterio clínico."*

Con BRD anotado, el 26,7 % de los chagásicos con BRD cae igual en `no_alta` [cálculo propio: 1 − 0,733]. Un BRD informado por el cardiólogo en un paciente de zona endémica justifica serología con independencia del modelo, que es la regla comparadora de VALIDACION-CLINICA.

**Estado: nueva. Costo:** trivial. **Riesgo:** ninguno.

#### Tabla resumen

| # | Propuesta | Estado frente a FASES | Toca score o banda | Riesgo frente al test | Costo | Valor |
|---|---|---|---|---|---|---|
| P1 | Criterio epidemiológico en la decisión y registro para la Etapa 1 | Nueva (≠ demográficos, que están descartados) | No | Ninguno | Bajo | **Muy alto** (seguridad y estado del arte) |
| P2 | Patrones binarios calibrados en val, BRD+HBAI y "alta sin patrón" | Nueva (reusa el umbral de `analisis_errores`) | No | Ninguno si es val | Bajo-medio | Alto |
| P3 | Detección de marcapasos | Nueva | No | Ninguno | Medio | Medio-alto (subgrupo de alto pretest) |
| P4 | Calidad dudosa y estabilidad de banda | Nueva | No | Ninguno | Medio | Medio |
| P5 | Estrés de dominio de 50 Hz, filtros y muestreo, en val | Nueva (medición, no palanca) | No | Ninguno; si lleva a cambiar el preprocesamiento, hace falta medición nueva | Bajo | **Alto** (riesgo silencioso de sensibilidad) |
| P6 | Auditoría de sexo y edad (LR± y descomposición) y contrato | Nueva como artefacto | No | Ninguno (parquet) | Bajo | Alto (ética y transparencia) |
| P7 | VPP según la prevalencia local | Identificada, no implementada | No | Ninguno | Muy bajo | Alto |
| P8 | LVSD como banco de evaluación y, opcional, sonda de disfunción del VI | Nueva; usarlo como target de Chagas, no | No | Ninguno | Bajo / medio | Medio (otro producto) |
| P9 | Etiquetas suaves y ranking | Smoothing sin medir (cercano a `peso_strong`); ranking es variante de `hardneg-v1` | Sí | Alto (reentrenamiento) | Alto | Bajo hoy |
| P10 | Texto de `no_alta` con criterio clínico | Nueva | No | Ninguno | Trivial | Medio |

---

### 4. Qué NO hacer, y por qué

1. **No volver a correr `evaluar_test.py`**, ni calibrar nada nuevo sobre test: umbrales de patrón, umbrales por sexo ni percentiles. Todo umbral sale de val.
2. **No meter edad ni sexo como entrada del modelo.** Está medido: el TPR@5% empeora (t = −3,55) [§13b]. La ganancia de Circulation viene de la **epidemiología**, no de la demografía. No confundirlas.
3. **No poner umbrales distintos por sexo antes de la Etapa 1.**
   - Bajar el umbral en mujeres sube su sensibilidad, pero baja su LR+ y su VPP, y cambia la prueba índice congelada.
   - El LR+ ya es igual entre sexos: la banda informa lo mismo a una mujer cuando se enciende.
   - La disparidad es de mezcla (BRD), no de trato.
   - Si se quiere explorar, que sea en val, como análisis preregistrado y con una muestra de confirmación aparte.
4. **No reabrir el hard negative mining ni el margin ranking** sin una hipótesis nueva [FASES 2026-09-10 §6]. La firma (RBBB sube, Chagas baja) dice que la distinción no está en la señal.
5. **No agregar cabezas de LBBB, FA, BAV1, QT ni bajo voltaje para subir la sensibilidad.**
   - El addendum del 2026-09-16 lo midió.
   - La literatura dice que el bajo voltaje (OR 0,84), el BRD incompleto y el ST-T aislado no discriminan [lit-clin §0.3].
   - El bajo voltaje es pronóstico (Rassi), no de tamizaje.
6. **No invertir en VFC, potenciales tardíos, QRS fragmentado ni variabilidad de la onda T.** La evidencia de tamizaje es negativa o inexistente, y requieren registros que DECA no recibe [lit-clin §2].
7. **No usar SaMi-Trop LVSD como positivos contra negativos de CODE-15% para la cabeza de Chagas.** Recrea el atajo de fuente.
8. **No pedir REDS-II** [FASES 2026-09-17].
9. **No comunicar `no_alta` como descarte ni mostrar el percentil sin la banda.** En `no_alta` cae el 78 % de los casos, y el percentil 92–97 de un `no_alta` suena a urgencia.
10. **No ofrecer DECA sobre ECG preocupacionales** sin revisión legal (Ley 26.281, arts. 5–6) [lit-epi §2.1].
11. **No recalibrar con datos argentinos y confirmar sobre los mismos datos** (VALIDACION-CLINICA §6).
12. **No preentrenar un modelo fundacional (ST-MEM o MIMIC) para "romper el techo".**
    - La ventaja externa frente a una buena ResNet es chica: 0,323 contra 0,280 [lit-IA §3.3].
    - Ningún resultado publicado con serología llega a AUC 0,85 en test externo [lit-IA §6].
    - Rompe el congelamiento.
13. **No formular el problema como PU learning como si eso fuera a crear señal.** Corrige la contaminación de los negativos, pero no hace visible la forma indeterminada [lit-IA §4.2].

**Conclusión honesta.** El modelo no puede subir significativamente contra serología, y eso está medido. En el campo, el ECG solo, prospectivo, da AUC ~0,65 [Circulation 2026]. **El valor que falta está fuera del modelo**: en la epidemiología (P1), en la seguridad del mensaje (P1, P3, P7, P10), en la robustez al dominio argentino (P5) y en la validación local (sección 5).

---

### 5. Plan de validación clínica (Fase 6, Argentina)

`VALIDACION-CLINICA.md` ya tiene un diseño sólido: Etapa 0 sin serología, Etapa 1 con verificación estratificada y Begg-Greenes, LR+ con piso en 5, STARD y TRIPOD+AI. Propongo **agregados**, no un reemplazo.

#### 5.1 Qué pedir y a quién

| Prioridad | Fuente | Qué pedir | Para qué |
|---|---|---|---|
| 1 | **Hospital Eva Perón (San Martín), Sección Chagas + INP Fatala Chabén** (grupo Viotti / Álvarez / Lococo; Laucella, INP-CONICET) [lit-epi §5] | ECG **digitales** (preguntar primero si guardan señal o papel) de seropositivos confirmados con estadio de Kuschnir y eco, más ECG de seronegativos del mismo hospital | **Etapa 0,5 (nueva):** sensibilidad de la banda alta **por estadio** (0/I/II/III) en argentinos; validación de `hbai` en chagásicos (el pendiente nº 5); prevalencia de BRD y BRD+HBAI argentina. Es caso-control y tiene sesgo de espectro: se usa para mecanismo y no para el LR+ final |
| 2 | **Hospital Papa Francisco (Salta)** [lit-epi §5] | Los 110 seropositivos con ECG de 12 derivaciones (2013–2016), con señal si existe | Igual que arriba, en el NOA; población con más HBAI+BRD |
| 3 | **Hospital Privado Universitario de Córdoba** (Salomone) | Cohorte de progresión con ECG seriado | Si hay señal: progresión, y si el score sube antes de la anomalía mayor |
| 4 | **Un sitio de la Etapa 1** en provincia endémica (Chaco, Santiago del Estero o Salta) con ECG digital exportable y laboratorio con serología normatizada | ECG consecutivos con informe (Etapa 0) y luego el prospectivo (Etapa 1) | El diseño de VALIDACION-CLINICA |
| 5 | Mundo Sano / UNSa (tamizaje de ingresantes, NCT07033325); Robles (Santiago del Estero); Pampa del Indio (Gürtler / Cardinal) | Poblaciones con serología conocida y **sin ECG**: proponer agregar un ECG digital a sus campañas | Etapa 1 con prevalencia alta (Pampa del Indio, 24–29 %), lo que abarata las serologías [VALIDACION-CLINICA §5] y aporta **las preguntas epidemiológicas ya relevadas** |

**Preguntas operativas para cada sitio, en este orden:**
1. ¿El equipo exporta señal digital? ¿En qué formato?
2. ¿A qué frecuencia de muestreo y con qué filtros? (Es el insumo de P5.)
3. ¿Cuál es la seroprevalencia adulta del sitio?
4. ¿Quién recibe a los seropositivos?

#### 5.2 Diseño: agregados al protocolo existente

1. **Las preguntas epidemiológicas como prueba en paralelo, preregistrada.** Se toman las 3 del AI-ECG-EPI y los criterios de la guía (P1) antes de conocer la banda. Resultados secundarios:
   - LR de las preguntas solas;
   - LR de la banda alta **dentro** de quienes no tienen criterio epidemiológico (el uso donde DECA agrega más);
   - un modelo combinado ajustado en el sitio A y confirmado en el B, o con división temporal.

   Con esto Argentina tendría su versión del 0,645 → 0,776.
2. **Estadificación de los seropositivos encontrados.** Eco y ECG estándar según la guía, que ya es atención obligatoria por vinculación. Permite reportar la sensibilidad por estadio (cardiopatía frente a forma indeterminada), que es la arena "CCC" de Jidling (0,82) con verdad argentina. Además da el primer dataset argentino con **serología + señal + eco**: el dato que FASES dice que no existe.
3. **Estratos preregistrados:** sexo, edad (<45, 45–64, ≥65), presencia de BRD y criterio epidemiológico. Calidad de señal (P4) y marcapasos (P3) se reportan aparte.
4. **Registrar el equipo y la frecuencia de la red en cada ECG** para poder separar mezcla de casos de deriva de señal (la fórmula de VALIDACION-CLINICA §2).
5. **Comparadores obligatorios:**
   - "todo BRD → serología" (ya previsto);
   - **"criterio epidemiológico → serología"** (nuevo), porque es la práctica que la guía ya indica;
   - "criterio epidemiológico **o** banda alta" (la estrategia real del producto con P1).

   La pregunta a responder: **¿cuántos casos agrega DECA por cada 1.000 ECG sobre lo que ya encontraría la guía?**
6. **Economía.** No existe ningún estudio de costo-efectividad de "ECG/IA primero" para Chagas [lit-epi §2.4]. Registrar el costo por serología, la tasa de retorno al resultado y la vinculación. En TNMG completó la serología solo ~45 % de los invitados [lit-epi §4], así que el cuello de botella puede ser la conversión y no el AUC.
7. **Ética y regulación** (ya previsto en VALIDACION-CLINICA §8): comité según la Res. 1480/2011, consentimiento para la extracción, vinculación obligatoria de los seropositivos, exclusión de los ECG preocupacionales y consulta a ANMAT (Disp. 64/2025) antes de la Etapa 2.

#### 5.3 Criterio de lectura de los resultados

- **Si el LR+ de la banda alta tiene un límite inferior ≥ 5** y le gana a "todo BRD → serología": el priorizador sirve en Argentina.
- **Si el LR+ pasa el piso pero el valor incremental sobre el criterio epidemiológico es chico:** el producto se reubica como herramienta para quien **no tiene o no conoce** su historia epidemiológica (migrantes urbanos, consultas por otros motivos). Ese es, probablemente, el nicho real.
- **Si las cuatro tasas por estrato de BRD se caen:** el problema es de señal (equipo o red de 50 Hz). Se vuelve a P5 antes de cualquier otra cosa.

---

# Anexo A — Clínica del ECG en Chagas

## Electrocardiografía clínica de la cardiomiopatía chagásica crónica — revisión de literatura para DECA

Fecha de la búsqueda: 2026-10-14. Método: búsqueda web (PubMed/PMC, PLoS NTD, SciELO, sitios de sociedades). Cada número lleva su fuente. Uso estas marcas:

- **[V]** = verificado en el texto completo o en el abstract de la fuente.
- **[S]** = tomado de una fuente secundaria (una revisión que cita el original), sin haber leído el original.
- **[NV]** = no pude verificarlo; lo dejo marcado y no le asigno números.

Limitaciones de acceso: no pude abrir el texto completo de la declaración AHA 2018 (Circulation, 403), de Rassi 2006 (NEJM, 403), de Nunes 2021 (Circulation, 403) ni del Consenso SAC 2019 (falló la resolución DNS de sac.org.ar). Lo que digo de esos documentos sale de abstracts o de fuentes secundarias, y así lo indico.

---

### 0. Resumen para el equipo de IA

1. **El BRD (con o sin HBAI) es el hallazgo más característico, pero no explica a la mayoría de los seropositivos.** En población general seropositiva (REDS-II, donantes), alrededor del **49% tiene ECG normal** y el BRD aislado aparece en el **16%** [V]. Un detector de BRD, por construcción, no ve a la mitad de los infectados. Esa mitad no tiene cardiomiopatía, y la literatura no describe ningún marcador de ECG de 12 derivaciones en reposo que la identifique de forma reproducible.
2. **Fuera del BRD, lo que la literatura vincula a Chagas, con evidencia razonable, es lo siguiente:**
   - HBAI aislado: OR 1,60 en el metaanálisis; 15% vs 2% en REDS-II.
   - Extrasístoles ventriculares: OR 1,62; en Bambuí, 20,8% vs 9,6% para "extrasístoles frecuentes".
   - FA/aleteo: OR 2,11.
   - BAV de 1er grado: OR 1,71.
   - Ondas Q / zonas eléctricamente inactivas: 4,3% vs 1,7% en Bambuí; 10,6% en SaMi-Trop.
   - QT prolongado: 5,9% vs 2,1% en Bambuí; 15,1% en SaMi-Trop.
   - Duración del QRS como variable continua: mediana de 90 vs 88 ms en REDS-II, una diferencia estadísticamente significativa pero clínicamente mínima.
3. **Hallazgos que la literatura indica como NO discriminantes:**
   - Bajo voltaje del QRS: OR 0,84, no significativo en el metaanálisis.
   - BRD incompleto, BRI, HBPI, BAV de 2º y 3er grado: no significativos en el metaanálisis.
   - Alteraciones aisladas del ST-T en ancianos: 11,3% vs 13,0% en Bambuí.
   - Bradicardia sinusal ≥40/min, eje entre 0 y −30°, extrasístoles aisladas: consideradas inespecíficas por la revisión de Brito & Ribeiro (2018).
4. **Marcadores "sutiles" con ECG normal** (variabilidad de la FC, potenciales tardios, QRS fragmentado, variabilidad de la onda T, dispersión del QT): la evidencia es **preliminar o negativa** para tamizaje. Los estudios son chicos, mayormente sobre mortalidad en pacientes con cardiopatía y no sobre detección de la infección. Además, varios requieren Holter, ECG de alta resolución o señal promediada, no un ECG de reposo de 10 s.
5. **El mejor antecedente publicado coincide con el techo que midió DECA.** La red neuronal de Jidling et al. (2023) obtuvo AUC 0,59 en ELSA-Brasil (2% de seropositivos) y 0,68 en REDS-II. Cuando el positivo se define como *cardiomiopatía*, el AUC sube a 0,77 y 0,82 [V]. La literatura clínica explica por qué: el ECG refleja la cardiomiopatía, no la infección.
6. **Sexo.** El hombre tiene más anomalías mayores: OR ajustado 1,91 en SaMi-Trop, y más BRD+HBAI (15,3% vs 9,7%) [V/S]. Es coherente con la disparidad por sexo que midió DECA.

---

### 1. Anomalías del ECG en Chagas crónico y su frecuencia

#### 1.1 Definiciones de base

- **Forma indeterminada (FI):** serología positiva, sin síntomas, con ECG, radiografía de tórax y estudios contrastados de esófago y colon normales. El criterio se ratificó en la reunión de Araxá de 1985 [V] (Hasslocher-Moreno et al., *Rev Soc Bras Med Trop* 2021;54:e0254-2021, doi:10.1590/0037-8682-0254-2021).
- **Cardiomiopatía chagásica crónica (CCC):** se diagnostica por anomalías *características* del ECG en una persona seropositiva, tenga o no síntomas [V] (Brito & Ribeiro, *Rev Soc Bras Med Trop* 2018;51(5), doi:10.1590/0037-8682-0184-2018).
- **Estadificación (AHA 2018 / SAC 2019):**
  - A: ECG normal.
  - B1: anomalía del ECG con función global normal.
  - B2: FEVI reducida.
  - C: insuficiencia cardíaca.
  - D: insuficiencia cardíaca refractaria.

  Fuentes: [S] Nunes et al., *Lancet Reg Health Am* 2024; [S] definiciones del consenso SAC citadas por una revista argentina de reumatología. **Consecuencia para DECA: el estadio A es invisible en el ECG por definición.**

#### 1.2 Metaanálisis en población general — Rojas et al. 2018 [V]

Rojas LZ, Glisic M, …, Franco OH, Muka T. *PLoS Negl Trop Dis* 2018;12(6):e0006567. doi:10.1371/journal.pntd.0006567. https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0006567

- 49 estudios: 43 transversales y 6 de cohorte.
- 34.023 participantes: 12.276 con Chagas y 21.747 sin Chagas, de regiones endémicas.
- **Cualquier anomalía del ECG: 40,1% en Chagas vs 24,1% sin Chagas; OR 2,78 (IC95% 2,37–3,26).**

| Anomalía | OR (IC95%) | ¿Significativo? |
|---|---|---|
| BRD completo | **4,60 (2,97–7,11)** | Sí |
| BRD + HBAI | 3,34 (1,76–6,35) | Sí |
| FA / aleteo | 2,11 (1,40–3,19) | Sí |
| BAV 1er grado | 1,71 (1,25–2,33) | Sí |
| Extrasístoles ventriculares | 1,62 (1,14–2,30) | Sí |
| HBAI | 1,60 (1,21–2,13) | Sí |
| Bajo voltaje del QRS | 0,84 (0,47–1,50) | **No** |
| BRD incompleto | 0,85 (0,63–1,16) | No |
| BRI completo | 0,99 (0,62–1,58) | No |
| HBPI | 0,96 (0,35–2,62) | No |
| BAV 2º / 3er grado | 1,15 / 2,10 (IC que cruzan 1) | No |

El metaanálisis **no analiza** QT, ondas Q, ST-T ni bradicardia. Tampoco reporta prevalencias combinadas por anomalía, solo OR.

#### 1.3 REDS-II — donantes de sangre (São Paulo y Montes Claros) [V]

Ribeiro AL, Sabino EC, Marcolino MS, …, Busch MP, Murphy EL. *PLoS Negl Trop Dis* 2013;7(2):e2078. doi:10.1371/journal.pntd.0002078. https://pmc.ncbi.nlm.nih.gov/articles/PMC3585012/

- 499 seropositivos vs 488 seronegativos, emparejados por sitio, sexo, edad y año de donación.
- ECG y ecocardiograma leídos a ciegas por laboratorios centrales.

| Hallazgo | Seropositivos | Seronegativos |
|---|---|---|
| ECG anormal (cualquiera) | **51%** | 32% |
| ECG normal (calculado como complemento) | **≈49%** | ≈68% |
| Anomalías mayores | 26% | 9% |
| BRD aislado | 16% | 2% |
| HBAI aislado | 15% | 2% |
| BRD + HBAI | 4% | 0% |
| QRS, mediana (RIQ) | 90 ms (84–104) | 88 ms (82–96), p<0,001 |
| QTc, mediana | 429 ms | 427 ms, p=0,061 |
| FC, mediana | 62 lpm | 64 lpm, p=0,003 |
| FEVI <50% | 7,2% | 2%, p=0,0002 |

- En seropositivos con FEVI <50%, el ECG fue anormal en 69% vs 23% con FEVI normal.
- También fueron más frecuentes en ese grupo: BRD+HBAI (25% vs 2%), ST-T mayor (19% vs 3%) e infarto antiguo/ondas Q (11% vs 2%).
- Para detectar seropositivos con FEVI deprimida, el ECG tuvo **sensibilidad 92% y VPN 99%**. Las áreas bajo la curva para FEVI reducida fueron 0,778 para el QRS y 0,790 para el QTc.

Seguimiento del mismo grupo:
- Sabino EC et al., *Circulation* 2013;127:1105–15, doi:10.1161/CIRCULATIONAHA.112.123612 [S/abstract]. A ~10 años, el 24% de los seropositivos y el 5% de los seronegativos tenían cardiomiopatía definida. La diferencia de incidencia atribuible fue 1,85/100 personas-año. Se asoció con sexo masculino, ECG anormal previo y tercer ruido (S3).
- Nunes MCP et al., *Circulation* 2021;144:1553–66, doi:10.1161/CIRCULATIONAHA.121.055112 [abstract]. Incidencia de 13,8/1000 personas-año en seropositivos vs 4,6 en seronegativos (diferencia 9,2/1000 personas-año). El predictor destacado en el abstract es el **nivel de anticuerpos** (OR 1,4). No pude verificar qué variables del ECG resultaron predictoras [NV].

#### 1.4 SaMi-Trop — cohorte de pacientes con Chagas de Minas Gerais [V]

Rodrigues MGJ, Sabino EC, Ribeiro ALP, …, Ferreira AM. "Factors associated with major electrocardiographic abnormalities in Chagas disease patients: SaMi-Trop cohort." *Int J Cardiol Heart Vasc* 2026;63:101883. doi:10.1016/j.ijcha.2026.101883. https://pmc.ncbi.nlm.nih.gov/articles/PMC13040272/

- Línea de base 2013–14, n=1.938. Ojo: la cohorte se seleccionó a partir de ECG anormal en atención primaria (perfil de cohorte: Cardoso et al., *Int J Epidemiol* 2016), así que **no representa a la población seropositiva general**.
- 58,1% con al menos una anomalía mayor (Código de Minnesota).

| Anomalía | % |
|---|---|
| BRD completo o intermitente | 19,8 |
| QT prolongado (>115%) | 15,1 |
| ST-T mayor aislado | 11,9 |
| BRD + HBAI | 11,5 |
| Ondas Q mayores | 10,6 |
| FA | 4,4 |
| BRI | 4,0 |
| Marcapasos | 3,6 |
| Retraso de conducción intraventricular inespecífico | 2,4 |
| BAV de 2º/3er grado | 0,4 |

- Los autores advierten que las tablas 1 y 2 tienen inconsistencias internas.
- El artículo no reporta HBAI aislado, extrasístoles ventriculares, bajo voltaje ni bradicardia.
- Factores asociados (modelo ajustado):
  - Sexo masculino: OR 1,91 (1,41–2,58).
  - Hipertensión arterial: OR 2,00.
  - Sin tratamiento previo con benznidazol: OR 1,50.
  - Síntomas de insuficiencia cardíaca: OR 1,35.
- La edad no quedó en el modelo final.

#### 1.5 Bambuí — cohorte de adultos mayores (≥60 años) [V]

- Ribeiro AL, Marcolino MS, Prineas RJ, Lima-Costa MF. *J Am Heart Assoc* 2014;3(1):e000632. doi:10.1161/JAHA.113.000632. Datos tomados vía Brito 2023 y la revisión de Brito & Ribeiro 2018 [S].
- Brito BOF, Lima EM, Soliman EZ, Silva EF, Lima-Costa MF, Ribeiro ALP. *PLoS Negl Trop Dis* 2023;17(6):e0011419. doi:10.1371/journal.pntd.0011419. https://pmc.ncbi.nlm.nih.gov/articles/PMC10281574/ [V]

Muestra: 557 con Chagas vs 905 sin Chagas; línea de base en 1997.

| Anomalía | Chagas % | No Chagas % |
|---|---|---|
| Cualquier anomalía mayor | 56,0 | 31,8 |
| Extrasístoles supraventriculares/ventriculares frecuentes | **20,8** | 9,6 |
| BRD | 14,0 | 2,4 |
| BRD + HBAI | 9,1 | 0,8 |
| QT mayor (>115%) | 5,9 | 2,1 |
| FA | 5,7 | 1,8 |
| Ondas Q mayores | 4,3 | 1,7 |
| BRI | 3,2 | 2,0 |
| **ST-T mayor aislado** | **11,3** | **13,0** (no discrimina) |
| HVI | 1,9 | 4,1 |

- Incidencia de nuevas anomalías mayores en quienes partían de un ECG normal: 24,8 vs 15,5 por 1000 personas-año.
- La nueva anomalía con mayor exceso en Chagas fue BRD+HBAI (10,6 vs 3,7 por 1000 personas-año).
- HR ajustado de anomalía nueva por Chagas: 2,89 (2,28–3,67).
- Según JAHA 2014 [S]: cualquier anomalía mayor duplicó la mortalidad en Chagas (HR 2,18). La prevalencia de anomalías del ECG aumenta con la edad.

#### 1.6 Telesalud de Minas Gerais — "Marcolino" [V parcial]

Nota: no encontré un trabajo firmado por "Marcondes-Ribeiro". Supongo que se refiere a **Marcolino MS, Palhares DM, Ferreira LR, Ribeiro AL**, "Electrocardiogram and Chagas Disease: A Large Population Database of Primary Care Patients", *Global Heart* 2015;10(3):167–72, doi:10.1016/j.gheart.2015.07.001. Es el antecesor directo de la base CODE.

- 264.324 ECG de atención primaria; 7.590 pacientes (2,9%) con Chagas **autorreportado**.
- BRD: 22,7%; OR 10,73 (10,10–11,41).
- BRD+HBAI: 13,7%; OR 12,09 (11,20–13,04).
- Marcapasos: OR 13,29.
- FA: OR 3,15. BAV de 2º grado: OR 4,05 [S, vía Brito & Ribeiro 2018].
- ECG normal: **31,4% en Chagas vs 61,1% sin Chagas** [S].
- Sesgo: el grupo de comparación incluye infectados no declarados, y el autorreporte está enriquecido por pacientes ya diagnosticados (es el mismo problema de etiqueta que CODE-15%).

#### 1.7 ELSA-Brasil [V vía Jidling 2023]

No encontré un artículo que tabule anomalías del ECG por serología en ELSA-Brasil [NV]. Lo único verificado:
- 13.739 participantes; 2,0% seropositivos; 0,3% con CCC.
- Ver la sección 2.6.

#### 1.8 Datos argentinos y bolivianos

- **Salta** [V]: Nuñez Burgos A et al., *Insuf Card* 2016;11(4). https://www.scielo.org.ar/scielo.php?script=sci_arttext&pid=S1852-38622016000400003
  - 110 seropositivos: 60 sin patología demostrada y 50 con patología.
  - En los 50 con patología: HBAI+BRD 30%, bradicardia sinusal 20%, FA crónica 14%, HBAI aislado 12%, extrasístoles ventriculares 10%, BRD aislado 10%.
  - Serie hospitalaria chica y sin controles; los porcentajes publicados no cuadran con n=50.
- **Chaco boliviano** [V]: Fernandez AB, Nunes MCP, Clark EH et al., *Global Heart* 2015;10(3):159–66, doi:10.1016/j.gheart.2015.07.004.
  - 398 seropositivos de comunidades guaraníes: solo **13,8%** con alguna anomalía sugestiva.
  - HBAI aislado 4,3%, BRD+HBAI 3,3%, BRD aislado 2,8%, BAV 2,6%.
  - Por edad: 1,1% a los 10–19 años, 14,2% a los 20–39, 17,3% a los 40–59 y 26,4% a los ≥60.
  - No hubo controles seronegativos con ECG.
  - **Relevante:** en una población hiperendémica no seleccionada, ~86% de los seropositivos tiene ECG "normal" según la definición del estudio.
- **Consenso SAC (Mazza) 2011** [V]: el BRD, solo o con HBAI, es el trastorno de conducción más frecuente y tiene poco valor pronóstico. El BRI y el HBPI son raros y de mal pronóstico. Estiman que el 20–30% de los infectados desarrolla cardiopatía y ~10% la forma dilatada.

#### 1.9 Qué es típico y qué es inespecífico — Brito & Ribeiro 2018 [V]

- **Típico de CCC:**
  - BRD+HBAI, más prevalente en Chagas que en otras cardiomiopatías.
  - Zonas eléctricamente inactivas.
  - El BRI es raro pero se asocia fuertemente a insuficiencia cardíaca.
- **Típico pero no específico:** extrasístoles ventriculares (ocurren en sanos y, aisladas, no separan seropositivos de seronegativos).
- **Inespecífico:**
  - Bradicardia sinusal ≥40 lpm, BAV de 1er grado, eje entre 0 y −30°.
  - Bajo voltaje en derivaciones de miembros, extrasístoles aisladas, BRD incompleto, HBAI aislado.
  - Alteraciones de la repolarización inespecíficas.
  - FA (tasa similar a la de otras cardiomiopatías).
- En series clínicas antiguas, las zonas inactivas van del 0,5% al 30% y las alteraciones de repolarización del 0,2% al 40%. La heterogeneidad es enorme y depende de cómo se seleccionó la muestra.

---

### 2. Marcadores sutiles o tempranos con ECG normal — la pregunta crítica

**Conclusión general:** no hay ningún marcador del ECG de 12 derivaciones en reposo con evidencia sólida para detectar infección en seropositivos con ECG normal. Lo que existe es esto:

#### 2.1 Cambios "tempranos" descritos en series clásicas [S, evidencia débil]

Garzon SAC, Lorga AM, Nicolau JC. "Electrocardiography in Chagas' heart disease." *São Paulo Med J* 1995;113(2), doi:10.1590/S1516-31801995000200011. Es una revisión narrativa.

- Alteraciones aisladas tempranas que menciona: BAV de 1er grado, bajo voltaje, extrasístoles ventriculares raras, BRD incompleto y aplanamiento localizado de la onda T.
- En jóvenes de 10 a 20 años aparecerían primero el BRD incompleto, el HBAI, las T aplanadas y el BAV de 1er grado. Después vendrían el BRD y las extrasístoles ventriculares, y más tarde las zonas inactivas y el BAV completo.
- Describe **menos arritmia sinusal** en jóvenes seropositivos que en seronegativos, lo que sugiere disautonomía.
- Incidencia de nuevas anomalías en quienes parten de un ECG normal: ~2,6% por año.

**Advertencia:** el metaanálisis de 2018 encuentra que el BRD incompleto y el bajo voltaje **no** discriminan (OR 0,85 y 0,84), y Brito & Ribeiro los clasifican como inespecíficos. Son plausibles como marcadores "tempranos", pero no están validados como discriminantes.

#### 2.2 Duración del QRS como variable continua [V]

- REDS-II: mediana de 90 vs 88 ms (p<0,001). El efecto es real pero pequeño a nivel poblacional.
- Una revisión cita un QRS >100 ms correlacionado con FEVI baja y dilatación [S], y el QRS como la única variable aislada del ECG asociada a una caída de ≥5% de la FEVI en 8 años [S, Brito & Ribeiro 2018, ref. 33].
- En una cohorte de Santa Cruz (Bolivia, n=139), los que progresaron tenían un QRS más ancho (115,4 ± 25,1 ms) [S]. Es una cohorte chica y no pude leer el original.

#### 2.3 Variabilidad de la frecuencia cardíaca y disautonomía [S/V, preliminar]

- Ribeiro AL et al., "Parasympathetic dysautonomia precedes left ventricular systolic dysfunction in Chagas disease", *Am Heart J* 2001;141:260–5. Solo verifiqué la cita (vía AHA 2018); no leí el texto ni los números [NV].
- Correia D et al., *PACE* 2007, doi:10.1111/j.1540-8159.2007.00749.x [S]. En la forma indeterminada, las medianas de los índices de VFC **no difirieron** de los controles (p=0,17–0,87), aunque hubo gran dispersión individual (valores tanto deprimidos como exagerados).
- Una revisión de patogenia afirma que la disfunción autonómica aparece antes que la disfunción ventricular, incluso en la forma indeterminada. Agrega que **con cepas no neurotrópicas (Venezuela) aparece después**.
- La VFC requiere registros largos (Holter); en 10 s solo se puede medir FC y arritmia sinusal gruesa.
- Bambuí 2023 publicó "disautonomía vagal y mortalidad a 14 años" (PubMed 37979240). Solo vi el título [NV].

#### 2.4 ECG de señal promediada / potenciales tardíos [V, negativo]

- Koehler (*Scientia Medica*, PUCRS, ~2009), https://revistaseletronicas.pucrs.br/scientiamedica/article/view/3642. Con 59 pacientes en forma indeterminada vs 44 controles, el ECG de alta resolución fue positivo en el **22% vs 34,1% (p=0,23)**: no discrimina.
- El SEARCH-RIO (Benchimol-Barbosa et al., *Braz J Med Biol Res* 2013) [S] usa la señal promediada como **pronóstico** de taquicardia ventricular y muerte, no para tamizaje.
- El Consejo de Chagas de la Sociedad Sudamericana de Cardiología le asigna Clase IIIb al ECG de señal promediada.

#### 2.5 QRS fragmentado [V, sin datos en la forma indeterminada]

- FECHA (Baranchuk et al.; letra en *Indian Pacing Electrophysiol J* 2014;14(6):309–10): QRS fragmentado en **59,6%** de 98 pacientes con CCC y cardiodesfibrilador implantable. **No predijo terapias apropiadas** (p=0,87).
- La guía ASE 2018 de imágenes lo lista entre los marcadores de peor pronóstico.
- **No encontré ningún estudio de QRS fragmentado en forma indeterminada ni en seropositivos vs seronegativos.** Es una laguna real.
- El score QRS de Selvester se usó en Chagas agudo amazónico (BMC Infect Dis 2021). Según Brito & Ribeiro 2018 [S], un QRS-score >2 predijo realce tardío por resonancia y FEVI baja.

#### 2.6 Repolarización (onda T, QT) [V, mayormente pronóstico]

- Variabilidad de amplitud de la onda T >30 µV² predijo muerte con HR 5,76 (1,31–25,23); n=113, 14 muertes. Ribeiro AL et al., *J Cardiovasc Electrophysiol* 2011;22:799–805, doi:10.1111/j.1540-8167.2010.02000.x. Requiere análisis latido a latido.
- La desviación del eje de la onda T triplicó la mortalidad (Salles et al., *Am J Cardiol* 2004; n=738), sobre todo en quienes ya tenían ECG anormal.
- Dispersión del QT: HR 1,45 por cada 10 ms para mortalidad (Salles et al. 2003, PubMed 12835217) [S].
- Ejes P, QRS y T predijeron mortalidad a 14 años en Bambuí, con HR 1,48, 1,34 y 1,35 [S, Brito & Ribeiro 2018, ref. 49].
- **Todo esto es pronóstico en pacientes chagásicos, no detección de la infección.**

#### 2.7 Onda P

No encontré evidencia de que la onda P discrimine seropositivos con ECG normal [NV]. Solo aparece el eje P como predictor de mortalidad en Bambuí.

#### 2.8 Más allá del ECG: el sustrato existe aunque el ECG sea normal

- Fibrosis por resonancia en la forma indeterminada: 6/17 (41%) en un estudio chico (PMC5855905). Hay un metaanálisis de 19 estudios en *JACC Imaging* 2023 [S].
- El strain longitudinal global puede estar reducido en la forma indeterminada con ecocardiograma convencional normal (AHA 2018 [S]; PMC8826593).
- El Consejo de Chagas de la Sociedad Sudamericana de Cardiología (2023) sostiene que otras herramientas detectarían anomalías en **~20–30% de los asintomáticos con ECG normal** [V]. Es opinión de expertos, sin estudio de respaldo citado.
- En el Chaco boliviano, entre seropositivos con ECG normal y ecocardiograma (n=66), el 13,6% tuvo FEVI de 40–54% [V].
- **Lectura para DECA:** parte de los positivos "invisibles" tiene daño miocárdico subclínico que el ECG de superficie no expresa. No hay base para esperar que un ECG de 10 s los capture.

#### 2.9 IA sobre ECG (contexto directo para DECA) [V]

- **Jidling C, Gedon D, Schön TB, Oliveira CDL, Cardoso CS, …, Ribeiro ALP**. *PLoS Negl Trop Dis* 2023;17(7):e0011118. doi:10.1371/journal.pntd.0011118.
  - Validación interna: AUC 0,80.
  - **REDS-II: AUC 0,68.** Sensibilidad 0,52 y especificidad 0,77.
  - **ELSA-Brasil: AUC 0,59.** Sensibilidad 0,36 y especificidad 0,76.
  - Con solo CCC como positivo: AUC 0,82 en REDS-II y 0,77 en ELSA-Brasil.
  - Los autores atribuyen el bajo rendimiento en no-CCC a la calidad de etiqueta de CODE.
- **Brito BOF et al.** (IA-ECG para disfunción sistólica, SaMi-Trop). *PLoS Negl Trop Dis* 2021, doi:10.1371/journal.pntd.0009974.
  - n=1.304; 7,1% con FEVI ≤40%.
  - AUC 0,839; sensibilidad 73%, especificidad 83%, VPN 97%.
  - Datos en github.com/samitrop/AI-ECG-Chagas. Probablemente es el origen del `samitrop-lvsd/` que ya está en el SSD; conviene verificarlo.
  - Una versión posterior con ajuste fino reporta AUC 0,89 (PMC13400128).

---

### 3. Diferencias por sexo y por edad

**Sexo:**
- SaMi-Trop 2026 [V]: sexo masculino OR ajustado 1,91 para anomalía mayor.
  - BRD+HBAI: 15,3% en hombres vs 9,7% en mujeres (p<0,001).
  - BRD sin HBAI: 19,5% vs 19,9% (p=0,856).
  - Estos dos valores de sexo salen de un extracto de la tabla vía buscador [S]; los autores advierten inconsistencias de rótulos.
- Garzon 1995 [S]: razón de prevalencia de ECG anormal de 4,3 en hombres seropositivos vs seronegativos, y de 2,6 en mujeres. Brito & Ribeiro 2018 citan 26,1% vs 15,3% de ECG anormal en hombres vs mujeres seropositivos (ref. 10, no verificada en el original).
- Sabino 2013 [abstract]: el sexo masculino se asoció a cardiomiopatía incidente.
- Rassi 2006: el sexo masculino suma puntos en el score pronóstico.
- Bambuí: las mujeres predominan entre los infectados (67,5% de los casos de Chagas). El artículo no estratifica la incidencia por sexo.
- **Lectura para DECA:** la disparidad 18,5% vs 27,3% que midió el proyecto es coherente con la literatura. A igual infección, las mujeres expresan menos anomalías de conducción (en especial BRD+HBAI), así que un detector apoyado en BRD rinde menos en ellas. No encontré estudios que describan un patrón de ECG *alternativo* en mujeres [NV].

**Edad:**
- La prevalencia de anomalías sube con la edad, en Chagas más que en controles. En el Chaco boliviano va del 1,1% a los 10–19 años al 26,4% a los ≥60.
- El exceso relativo atribuible a Chagas es **máximo a los 25–44 años**: razón 9,6 en Garzon 1995 [S]. Se achica después de los 50, y por encima de 64 años no habría diferencia según esa fuente. Bambuí sí encuentra diferencias en ≥60 (56% vs 31,8% de anomalías mayores).
- **Implicancia:** en mayores, las anomalías por comorbilidad (HTA, isquemia) diluyen la especificidad. El ST-T no discrimina en ancianos (Bambuí) y la HVI es *menos* frecuente en Chagas.

---

### 4. Scores pronósticos y qué variables del ECG usan

| Score | Población | Variables del ECG | Otras variables | Fuente |
|---|---|---|---|---|
| **Rassi** (2006) | 424 pacientes de desarrollo + 153 de validación; mortalidad 31% en 7,9 años | **Bajo voltaje del QRS (2 pts)**, **TV no sostenida en Holter (3 pts)** | NYHA III/IV (5), cardiomegalia en la radiografía (5), disfunción del VI en el ecocardiograma (3), sexo masculino (2) | Rassi A Jr et al., *N Engl J Med* 2006;355:799–808, doi:10.1056/NEJMoa053457 |
| **SaMi-Trop** (2020) | 1.551 pacientes con CCC; mortalidad a 2 años | **FC ≥80 lpm (20 pts)**, **QRS ≥150 ms (15 pts)** | Edad (10 por década), NYHA >I (15), NT-proBNP anormal para la edad (55) | Di Lorenzo Oliveira C et al., *J Am Heart Assoc* 2020, https://pmc.ncbi.nlm.nih.gov/articles/PMC7335521 |
| SaMi-Trop + IA-ECG | Misma cohorte | La IA-ECG reemplaza al NT-proBNP | — | PMC13400128 (2026) |
| de Souza et al. 2015, *Int J Cardiol* | Muerte súbita | Dispersión del QT, extrasístoles ventriculares | Síncope, disfunción severa del VI | [S] Marin-Neto & Rassi, *Mem Inst Oswaldo Cruz* 2022;117:e210172 |
| SEARCH-RIO (Benchimol-Barbosa 2013) | TV / muerte | Señal promediada (QRS filtrado, transitorios), TV no sostenida | FEVI | [S] ídem |
| "Score integrado" Mitelman-Giménez (SSC 2023) | Asintomáticos con serología positiva | Dispersión del QT, VFC, señal promediada (IIIb) | Pruebas autonómicas, ecocardiograma, marcadores | [V] SSC, figshare 10.6084/m9.figshare.22303504 — expertos, sin validación publicada que haya encontrado |

Sobre Rassi:
- Riesgo bajo 0–6, intermedio 7–11, alto 12–20. Mortalidad a 10 años: 10%, 44% y 84%.
- Los puntos coinciden en dos fuentes secundarias y con lo que recuerdo del original, pero no pude abrir el NEJM. **Tratar los puntos como [S].**
- Sobre SaMi-Trop: [V] en el abstract.
- **Punto clave para DECA:** ningún score usa BRD ni HBAI. Rassi usa **bajo voltaje**, que es pronóstico en cardiopatía establecida (HR 1,87 [S]) pero **no discrimina la infección** (OR 0,84). Pronóstico y tamizaje usan señales distintas.

---

### 5. Guías argentinas y latinoamericanas

- **AHA/SIAC 2018:** Nunes MCP, Beaton A, Acquatella H, …, *Circulation* 2018;138:e169–e209, doi:10.1161/CIR.0000000000000599 [S; no pude acceder al texto completo].
  - Define la forma indeterminada con ECG normal y la estadificación A–D.
  - Reconoce el BRD y el HBAI como las alteraciones más comunes.
  - Cita el strain como marcador más sensible.
  - **No pude verificar** un intervalo concreto de ECG para el seguimiento [NV].
- **SAC — Consenso Enfermedad de Chagas 2019:** *Rev Argent Cardiol* 2020;88(Supl 8):1–74; versión resumida en *Rev Argent Cardiol* 2022;90(Supl 1):1–24.
  - No pude acceder (falla DNS de sac.org.ar).
  - Por fuentes secundarias: usa la categoría "crónico sin patología demostrada" y la estadificación A/B1/B2/C/D.
  - El ECG es el disparador del ecocardiograma (indicado en la línea de base y ante ECG nuevo anormal).
  - **Clase y nivel de las recomendaciones: [NV].**
- **SAC — Consenso Mazza (≈2011),** *Rev Argent Cardiol* 79(6) [V]:
  - Crónico sin patología demostrada: ECG de 12 derivaciones Clase I, Nivel A; Holter de 24 h Clase I, Nivel C.
  - **Control anual** (clínica, ECG, radiografía, ergometría, Holter, ecocardiograma): Clase I, Nivel C.
  - Con cardiopatía: ECG, Holter y ergometría Clase I, Nivel A.
- **Sociedad Sudamericana de Cardiología, Acuerdo Regional y Nueva Clasificación (2023)** [S/V]:
  - Seguimiento clínico mínimo anual con ECG de 12 derivaciones en la forma indeterminada.
  - Propone separar dos categorías: "asintomático sin patología demostrable (inactivo)" y "asintomático con patología subclínica (activo)". La segunda se detecta con herramientas distintas del ECG.
- **ASE/ECOSIAC/DIC-SBC 2018** (imágenes): ECG de seguimiento al menos cada 2–5 años en la forma indeterminada [S]. Lista como marcadores de peor pronóstico a las extrasístoles ventriculares, la dispersión del QT, el bajo voltaje, el QRS fragmentado y el QRS prolongado.
- **Sobre tamizaje:** ninguna guía que haya podido leer propone el ECG como herramienta de **tamizaje de la infección**. Todas lo usan para estadificar a quien ya es seropositivo. La indicación de serología se basa en criterio epidemiológico: origen o residencia en zona endémica, madre seropositiva, transfusión, etc. **DECA invierte el sentido habitual (del ECG a la serología), y no encontré respaldo de guías para eso**; tampoco una contraindicación explícita.

---

### 6. Implicancias concretas para el modelo

1. **El techo es esperable por la clínica.**
   - Entre el 49% (REDS-II) y el 86% (Chaco boliviano) de los seropositivos tienen ECG normal.
   - La progresión desde la forma indeterminada es de ~1,9% por año (metaanálisis Chadalawada 2020 [S]) o de ~0,9–1,85 por 100 personas-año de exceso (REDS-II).
   - Una sensibilidad baja en población general es una propiedad de la enfermedad, no del modelo.
2. **Señales no-BRD con respaldo, por orden de solidez:**
   1. HBAI aislado.
   2. Extrasístoles ventriculares (en 10 s, solo si aparecen en la tira).
   3. FA/aleteo.
   4. BAV de 1er grado.
   5. Ondas Q / zonas inactivas.
   6. QT/QTc prolongado.
   7. QRS ancho como variable continua.

   Todas tienen OR más bajos que el BRD (1,6–2,1 frente a 4,6) y son inespecíficas en mayores.
3. **No conviene invertir esfuerzo en** bajo voltaje, BRD incompleto, ST-T aislado en mayores, potenciales tardios o QRS fragmentado. Los primeros no discriminan; el QRS fragmentado no tiene evidencia en la forma indeterminada.
4. **Para validar con el objetivo correcto:** la literatura (Jidling 2023, Brito 2021) muestra que el ECG detecta **cardiomiopatía**, no serología. `samitrop-lvsd`, con FEVI ecocardiográfica, es coherente con eso.
5. **Sexo:** reportar el rendimiento estratificado por sexo es coherente con la literatura. La menor expresión de BRD+HBAI en mujeres está documentada.

---

### Referencias principales (verificadas salvo indicación)

1. Rojas LZ et al. *PLoS Negl Trop Dis* 2018;12:e0006567. doi:10.1371/journal.pntd.0006567
2. Ribeiro AL et al. *PLoS Negl Trop Dis* 2013;7:e2078. doi:10.1371/journal.pntd.0002078
3. Sabino EC et al. *Circulation* 2013;127:1105–15. doi:10.1161/CIRCULATIONAHA.112.123612 (abstract)
4. Nunes MCP et al. *Circulation* 2021;144:1553–66. doi:10.1161/CIRCULATIONAHA.121.055112 (abstract)
5. Rodrigues MGJ et al. *Int J Cardiol Heart Vasc* 2026;63:101883. doi:10.1016/j.ijcha.2026.101883
6. Ribeiro AL et al. *J Am Heart Assoc* 2014;3:e000632. doi:10.1161/JAHA.113.000632 [S]
7. Brito BOF et al. *PLoS Negl Trop Dis* 2023;17:e0011419. doi:10.1371/journal.pntd.0011419
8. Marcolino MS et al. *Global Heart* 2015;10:167–72. doi:10.1016/j.gheart.2015.07.001 (abstract y highlights)
9. Brito BOF, Ribeiro ALP. *Rev Soc Bras Med Trop* 2018;51(5). doi:10.1590/0037-8682-0184-2018
10. Fernandez AB et al. *Global Heart* 2015;10:159–66. doi:10.1016/j.gheart.2015.07.004
11. Nuñez Burgos A et al. *Insuf Card* 2016;11(4) (Salta)
12. Garzon SAC et al. *São Paulo Med J* 1995;113(2). doi:10.1590/S1516-31801995000200011
13. Hasslocher-Moreno AM et al. *Rev Soc Bras Med Trop* 2021;54:e0254-2021
14. Ribeiro AL et al. *J Cardiovasc Electrophysiol* 2011;22:799–805. doi:10.1111/j.1540-8167.2010.02000.x
15. Baranchuk A. *Indian Pacing Electrophysiol J* 2014;14:309–10 (FECHA)
16. Rassi A Jr et al. *N Engl J Med* 2006;355:799–808 [S]
17. Di Lorenzo Oliveira C et al. *J Am Heart Assoc* 2020 (SaMi-Trop score), PMC7335521
18. Jidling C et al. *PLoS Negl Trop Dis* 2023;17:e0011118. doi:10.1371/journal.pntd.0011118
19. Brito BOF et al. *PLoS Negl Trop Dis* 2021;15:e0009974. doi:10.1371/journal.pntd.0009974
20. Nunes MCP et al. *Circulation* 2018;138:e169–e209. doi:10.1161/CIR.0000000000000599 [S]
21. Nunes MCP et al. *Lancet Reg Health Am* 2024;37:100832. doi:10.1016/j.lana.2024.100832
22. Consenso de Enfermedad de Chagas-Mazza, *Rev Argent Cardiol* 79(6) (~2011)
23. Consenso Enfermedad de Chagas 2019 (SAC), *Rev Argent Cardiol* 2020;88(Supl 8) [NV, inaccesible]
24. Mitelman J, Giménez L et al. (SSC). Nueva clasificación clínica de la enfermedad de Chagas, 2023. doi:10.6084/m9.figshare.22303504
25. Marin-Neto JA, Rassi A Jr. *Mem Inst Oswaldo Cruz* 2022;117:e210172
26. Koehler (*Scientia Medica*, PUCRS) — potenciales tardíos en la forma indeterminada

---

# Anexo B — IA/ECG para Chagas

## IA/deep learning sobre ECG para detectar Chagas: revisión de literatura

Fecha de búsqueda: 2026-10-14. Fuentes: los PDFs de CinC 2025 (texto completo, cinc.org/archives/2025), arXiv, PubMed (E-utilities), PLOS NTD, PMC y el TSV oficial de resultados del Challenge.

**Cómo leer las cifras.** Cada número lleva una etiqueta:
- **[INT]**: validación interna. Es CV o un holdout sobre las mismas fuentes de entrenamiento (CODE-15% / SaMi-Trop / PTB-XL).
- **[EXT]**: validación externa. Son cohortes que el modelo no vio: REDS-II, SaMi-Trop 3, ELSA-Brasil, cohortes prospectivas.
- **[NV]**: no verificado en la fuente primaria.

"Score" o "TPR@5%" es la métrica del Moody Challenge 2025: la fracción de positivos que cae dentro del 5% de pacientes con score más alto.

---

### 0. Resumen para DECA

1. **Nadie reporta AUC > 0,85 con TPR@5% > 0,40 en un test externo con serología como etiqueta.**
   - El mejor sistema del Challenge 2025 (ViT con fundación ST-MEM + edad/sexo) obtuvo **TPR@5% medio 0,323** en el test oculto [EXT].
   - Por subconjunto: AUROC 0,777 en REDS-II, 0,767 en SaMi-Trop 3 y 0,566 en ELSA-Brasil.
   - Las cifras ≥ 0,85 / ≥ 0,40 aparecen solo en dos situaciones:
     - **Validación interna** sobre CODE-15%/SaMi-Trop/PTB-XL [INT].
     - **Cambiando la etiqueta** de "serología" a "cardiomiopatía" o "disfunción del VI por eco" [EXT]. Ejemplos: Jidling 2023 con CCC como positivo, AUC 0,82 en REDS-II; AI-ECG para FEVI ≤ 40% en SaMi-Trop, AUC 0,84 a 0,89.
2. **Nuestros números (test arena A: AUC 0,835 / AUPRC 0,157 / TPR@5% 0,372) están en la misma banda que el estado del arte en validación interna.**
   - Equipo AIMED (CinC 2025): CV con AUROC 0,842, AUPRC 0,167 y score 0,410 [INT].
   - Ahus AIM: CV con AUC 0,840 y score 0,439 [INT].
   - AIChagas sobre CODE-15% solo: score 0,416 [INT].
   - Biomed-Cardio: CV con AUROC 0,867 y score 0,490 [INT]. Incluye SaMi-Trop y PTB-XL en el pool, que inflan la métrica (ver §1.4).
   - Todos ellos cayeron a **0,20–0,32** en el test oculto.
3. **"Funciona como detector de BRD" es lo que la literatura espera.**
   - AIChagas midió en CODE-15% que el BRD es 8 veces más frecuente en positivos que en negativos (20% contra 2,4%).
   - Wen & Kang 2026 describen distribuciones bimodales: los modelos separan la cardiomiopatía establecida de la infección indeterminada, que es electrofisiológicamente indistinguible de los controles. Lo llaman un "plateau impuesto por la fisiopatología".
   - El trabajo prospectivo más grande (Ribeiro, *Circulation* 2026) da **AUC 0,645 para el AI-ECG solo**, en población real con serología.
4. **Palancas del top-5 que DECA podría no haber probado.** Hay que contrastarlas con FASES.md, que ya registra nueve palancas medidas.
   - Etiquetas suaves o *label smoothing* asimétrico, solo para CODE-15%. Lo usan los puestos 1º, 2º y 3º.
   - Aumentación de ruido de red (50/60 Hz) y resampleo, como antídoto contra atajos de fuente.
   - Preentrenamiento autosupervisado o con otra supervisión: MAE/ST-MEM, DINO, destilación de un clasificador de 28 patologías, biomarcadores de MIMIC.
   - Ensembles.
   - Pérdida de ranking contra negativos difíciles.
   - Ninguna de estas palancas movió el test externo más allá de ~0,32.
   - **Dato directo contra el multitarea con BRD:** el equipo chagas_detector (CinC 2025 #416) probó multitarea Chagas+BRD+BAV1 y bajó en su test interno, de 0,426 a 0,349.

---

### 1. George B. Moody PhysioNet Challenge 2025

#### 1.1 Diseño (paper oficial)
- **Referencias del paper oficial.**
  - Reyna MA, Koscova Z, Pavlus J, Saghafi S, Weigle J, Elola A, … Ribeiro AH, Ribeiro ALP, Sameni R, Clifford GD. "Detection of Chagas Disease from the ECG: The George B. Moody PhysioNet Challenge 2025". arXiv:2510.02202 (2025), https://arxiv.org/abs/2510.02202. También en CinC 2025: https://cinc.org/archives/2025/pdf/CinC2025-068.pdf.
  - Versión de revista: *Physiol Meas* 2026, doi:10.1088/1361-6579/aead44 (PMID 42805232).
- **Entrenamiento público.**
  - CODE-15%: 335.621 registros, 1,92% positivos, etiqueta autorreportada.
  - SaMi-Trop: 1.631 registros, 100% positivos, serología.
  - PTB-XL: 21.779 registros, 0% positivos, presunción geográfica.
- **Validación oculta.** REDS-II: 37.779 registros, 2,05% positivos. La prevalencia está ajustada por sobremuestreo.
- **Test oculto.**
  - REDS-II: 13.720 registros, 2,04%.
  - SaMi-Trop 3: 53.444 registros, 2,06%.
  - ELSA-Brasil: 13.739 registros, 2,04%. Es población mayormente asintomática, con serología.
- **Aumentación del lado de los organizadores.** Los datos ocultos recibieron ruido, filtros representativos de dispositivos y resampleo.
- **Participación.** 111 equipos, más de 1.300 envíos, 41 equipos rankeados.
- **Medianas entre equipos** [EXT]:
  - Validación REDS-II: 0,279.
  - Test REDS-II: ~0,275. El texto dice 0,0275, probablemente una errata.
  - Test SaMi-Trop 3: 0,236.
  - Test ELSA-Brasil: 0,100, una caída del 64%.
- **Ensemble de los organizadores** (abstract de *Physiol Meas*) [EXT]: TPR@5% de 0,389 en REDS-II y 0,146 en ELSA-Brasil. La cifra para SaMi-Trop 3 no figura en el abstract [NV].

#### 1.2 Top-15 oficial (TSV de resultados: https://moody-challenge.physionet.org/2025/results/challenge_2025_results.tsv)

Todas las columnas de test son [EXT]. "Val" es la validación oculta REDS-II.

| # | Equipo | Enfoque (ver §1.3) | Val | REDS-II test score / AUROC / AUPRC | SaMi-Trop 3 score / AUROC / AUPRC | ELSA score / AUROC / AUPRC | Media test |
|---|---|---|---|---|---|---|---|
| 1 | Biomed-Cardio (KU Leuven) | ViT preentrenado con ST-MEM, edad/sexo, etiquetas suaves | 0,445 | 0,468 / 0,777 / 0,298 | 0,376 / 0,767 / 0,205 | 0,125 / 0,566 / 0,063 | **0,323** |
| 2 | DlaskaLabMUI (Innsbruck) | Ensemble de MAE-Transformer + xLSTM (DINO), *label smoothing* en CODE-15% | 0,440 | 0,357 / 0,735 / 0,283 | 0,375 / 0,739 / 0,181 | 0,118 / 0,567 / 0,055 | 0,283 |
| 3 | AIChagas | ResNet1D, etiquetas suaves (BRD/BAV/FA), pérdida de ranking contra negativos difíciles | 0,360 | 0,382 / 0,704 / 0,125 | 0,329 / 0,749 / 0,159 | 0,129 / 0,626 / 0,055 | 0,280 |
| 4 | ISIBrno-AIMT | Destilación de un maestro U-Net de 28 patologías (más de 1M ECG), ensemble de 5 | 0,458 | 0,354 / 0,752 / 0,165 | 0,342 / 0,763 / 0,167 | 0,118 / 0,579 / 0,063 | 0,271 |
| 5 | Ahus AIM | Preentrenamiento con biomarcadores (MIMIC-IV), InceptionTime, ventanas de 2 s | 0,412 | 0,389 / 0,767 / 0,204 | 0,325 / 0,764 / 0,148 | 0,093 / 0,502 / 0,039 | 0,269 |
| 6 | cardio_Basel25 | InceptionTime con transfer, ventanas de 5 s, edad/sexo | 0,382 | 0,289 / 0,737 / 0,154 | 0,355 / 0,777 / 0,180 | 0,125 / 0,599 / 0,060 | 0,256 |
| 7 | CinCo Amigos (Leeds) | DANN + ELR + LMF/LDAM, edad/sexo, 2 etapas | 0,338 | 0,339 / 0,715 / 0,331 | 0,247 / 0,717 / 0,095 | **0,164** / 0,622 / 0,043 | 0,250 |
| 8 | chagas_detector | ECG-FM congelado + MLP con edad/sexo | 0,323 | 0,329 / 0,640 / 0,268 | 0,281 / 0,729 / 0,080 | 0,125 / 0,592 / 0,044 | 0,245 |
| 9 | AIMED (InCor-USP) | LiteVGG-11, 128 Hz, MC dropout | 0,372 | 0,321 / 0,712 / 0,207 | 0,295 / 0,713 / 0,123 | 0,104 / 0,558 / 0,046 | 0,240 |
| 10 | MIWEAR (Xiaomi) | FM de más de 10M ECG + curado por confianza | 0,379 | 0,282 / 0,682 / 0,211 | 0,308 / 0,722 / 0,128 | 0,125 / 0,564 / 0,051 | 0,238 |
| 12 | SwissBeatsNet (ETH) | MAE multiderivación, encoder congelado | 0,425 | 0,286 / 0,757 / 0,166 | 0,320 / 0,760 / 0,147 | 0,071 / 0,525 / 0,027 | 0,226 |

**Observaciones**
- **El máximo AUROC en un subconjunto de test es 0,777** (REDS-II, ganador). En ELSA-Brasil, el único conjunto externo y asintomático, **ningún equipo supera AUROC 0,63**.
- **El ranking en validación no predice el ranking en test.**
  - ISIBrno tuvo la mejor validación (0,458) y quedó 4º.
  - CinCo Amigos fue 1º en ELSA con una de las peores validaciones.

#### 1.3 Detalle técnico de los equipos punteros (leído del texto completo)

**1º Biomed-Cardio.**
- Referencia: Van Santvliet L, Nguyen PX, Vandenberk B, De Vos M. "Detecting Chagas Disease Using a Vision Transformer-based ECG Foundation Model". CinC 2025, doi:10.22489/CinC.2025.019.
- **Modelo.**
  - ViT-1D con 12 bloques de encoder, preentrenado con **ST-MEM** (75% de parches enmascarados, reconstrucción MSE) sobre 400.365 ECG.
  - La entrada es de 10 s a **100 Hz**, con parche de 50 muestras.
  - Agrega features de las 12 capas del encoder por *average pooling*.
- **Demografía.** Un MLP con sexo y edad en siglos produce shift/scale (tipo FiLM) sobre el vector de 768 dimensiones.
- **Ruido de etiqueta.** **Etiquetas suaves solo en CODE-15%** (positivo 0,8, negativo 0,2). PTB-XL y SaMi-Trop quedan en 0/1.
- **Desbalance.** BCE con peso 5 para positivos, más sobremuestreo ×5 de positivos.
- **Aumentación.**
  - **Ruido de red sintético de 50 o 60 Hz** con armónicos, SNR de 15 a 30 dB, con p = 0,5.
  - Recorte aleatorio a 5,65–10 s, *shift* y padding a 10 s.
- **Resultados.**
  - CV interna: score 0,490 ± 0,008, AUROC 0,867, AUPRC 0,252 [INT]. Con solo la cabeza entrenada y el backbone congelado: 0,381 [INT].
  - Validación REDS-II: 0,445. Test: 0,323 [EXT].
- **Lección.** Los ajustes contra confusores (ruido de red, resampleo) subieron la validación de 0,395 a 0,445 (+5 pts), mientras que en la CV interna solo movieron 2,7%. La métrica interna no ve las mejoras de robustez.

**2º DlaskaLabMUI.**
- Referencia: Nicolson A, Lunelli R, Pröll SM, Gruber N, Bauer A, Dlaska C. "Transformer–xLSTM Ensembles for ECG-Based Chagas Disease Detection". CinC 2025, doi:10.22489/CinC.2025.236.
- **Preentrenamiento.** Usaron **el CODE completo (8M ECG)**, que no es público y al que accedieron por los autores de CODE.
  - MAE-Transformer base (86M parámetros) y large (177M), con 80% de máscara.
  - xECG (xLSTM, 57M), entrenado con SimDINOv2.
- **Fine-tuning.** **Label smoothing 0,2 solo en CODE-15%.**
  - Sobremuestreo de positivos hasta una prevalencia efectiva del 5%, que coincide con el umbral de la métrica.
  - Ensemble de 5 modelos promediando logits.
  - Señal a 7,68 s, banda 0,5–60 Hz, z-score por derivación.
- **Resultados.**
  - Test interno: score 0,511, AUC 0,86 en la parte CODE-15% [INT]. Sobre SaMi-Trop+PTB-XL dio AUC 1,00 y AUPRC 0,92, lo que es un indicio de atajo de fuente.
  - Validación: 0,440. Test: 0,283 [EXT].
- **Lección.** Las mejoras en el test interno (que es casi todo CODE-15%) "no se tradujeron" a la validación oculta. El ensemble seleccionado ganaba 0,023 internamente y solo 0,004 en la validación.

**3º AIChagas.**
- Referencia: Hong C, Chou C-T, González S. "Managing Label Uncertainty in the Detection of Chagas Disease from the ECG". CinC 2025, doi:10.22489/CinC.2025.100.
- **Modelo.** ResNet1D desde cero, la arquitectura de Ribeiro/Jidling, con 12 × 4096 muestras (10 s).
- **Preprocesamiento.**
  - Revierte la escala de amplitud ×2 de SaMi-Trop y CODE-15%.
  - Banda 0,5–60 Hz.
  - Aumentación: máscaras simétricas de 0 a 1,5 s, deriva de línea de base, escala 0,8–1,2, *time-warp* 0,9–1,1, ruido y oclusión de hasta 1 s.
- **Etiquetas suaves en dos pasos.**
  1. Pacientes con autorreportes discordantes reciben la proporción de reportes positivos.
  2. **Negativos con BRD** reciben +0,15 de probabilidad, o +0,25 si además tienen BAV o FA. El tope es 0,49, para no reetiquetar.
  - Midieron en CODE-15% que el BRD aparece en 20% de los positivos contra 2,4% de los negativos (8×). FA: 5,8% contra 1,9%. BAV: 4% contra 1,6%.
- **Pérdida.** BCE (pos_weight 10) más **margin ranking loss** entre positivos y el 20% de negativos con mayor score del batch (*hard negatives*). El margen se escala por la diferencia de etiquetas suaves.
- **Ablación incremental** sobre su set interno "Chagas 5%" [INT]: 0,326 → 0,320 (aumentación) → 0,356 (etiquetas suaves) → 0,354 (ranking) → **0,388** (*hard negatives*).
- **Lo descartado.** DivideMix y la asignación dinámica de confianza "no convergieron o dieron mejoras menores".
- **Resultados.** Validación: 0,360. Test: 0,280 [EXT]. Fue el **mejor AUROC en ELSA (0,626)** del top-5.

**4º ISIBrno-AIMT.**
- Referencia: Nejedly P et al. "Knowledge Distillation from General Multi-Objective ECG Classification Model for Chagas Disease Detection". CinC 2025, doi:10.22489/CinC.2025.132.
- **Maestro.** U-Net multiobjetivo: segmentación P/QRS/T, 28 clases del Challenge 2021 y SNR. Lo entrenaron con más de 1M ECG, entre ellos MIMIC-IV-ECG con 800k.
- **Estudiantes.** ResNet que predicen Chagas más las 28 salidas del maestro (BCE de destilación). Ensemble de 5.
- **Preprocesamiento.** 500 Hz, banda 1–47 Hz, que elimina explícitamente 50/60 Hz porque Brasil usa 60 Hz y Alemania 50 Hz.
- **Hallazgo clave.** **Destilar del maestro superó a combinar explícitamente sus 28 hallazgos en una regresión logística**, que imitaba el razonamiento del cardiólogo.
- **Resultados.** Local: 0,558 [INT]. Validación: 0,458. Test: 0,271 [EXT].
- Reportan que en la fase no oficial hubo equipos cerca de 1,0, una señal de sesgo de datos.

**5º Ahus AIM.**
- Referencia: Stenhede E, Ranjbar A. "Biomarker-Based Pretraining for Chagas Disease Screening in Electrocardiograms". CinC 2025, doi:10.22489/CinC.2025.298; arXiv:2604.09782 (abril 2026).
- **Preentrenamiento.** Predice percentiles (100 bins) de 11 biomarcadores de MIMIC-IV (NT-proBNP, troponina, Hb, albúmina, …) a partir del ECG.
- **Modelo.** InceptionTime con ventanas de 2 s.
  - Usa solo 8 derivaciones: descarta I, II, III y aVR por ser combinaciones lineales.
  - En inferencia promedia 10 ventanas.
- **Etiquetas.** Promedio por paciente en los 1.825 pacientes con etiquetas discordantes.
- **Resultados.** CV: score 0,439, AUC 0,840 [INT]. Validación: 0,412. Test: 0,269 [EXT].
- **Sin ablación** con o sin el preentrenamiento [NV sobre su aporte]. Los modelos más grandes mejoraban la CV pero no la validación.

**7º CinCo Amigos.**
- Referencia: Wang X et al. "Two-Stage Domain Adversarial Learning…". CinC 2025, doi:10.22489/CinC.2025.158.
- **Etapa 1.**
  - DANN con dominios = datasets, incluyendo varios del Challenge 2021 sin sus etiquetas.
  - **ELR** (*Early Learning Regularization*) contra ruido de etiqueta.
  - Pérdida LMF (focal + LDAM).
  - Meta-red con edad y sexo.
- **Etapa 2.** Destilación de features sobre datos de alta calidad.
- **Resultado.** **1º en ELSA-Brasil (0,164)** [EXT]. Un SE-ResNet18 simple generalizó mejor que ECGNeXt, que "brillaba localmente".

**8º chagas_detector.**
- Referencia: Nguyen K et al. "Finetuning Foundational ECG Models to Detect Chagas Disease". CinC 2025, preprint https://cinc.org/2025/Program/accepted/416_Preprint.pdf.
- **Modelo.** ECG-FM congelado (90,4M parámetros, versión ajustada en MIMIC-IV-ECG) más un MLP con edad y sexo.
- **Resultados de interés para DECA** [INT salvo donde se indica]:
  - Entrenar solo con PTB-XL+SaMi-Trop **dio 0,081 en la validación oculta**, contra 0,323 con CODE-15%. Las etiquetas débiles son imprescindibles.
  - **Multitarea Chagas+BRD+BAV1 empeoró**: 0,349 contra 0,426 en su test interno.
  - Descongelar ECG-FM empeoró.
  - Una "Percentile Ranking Loss" orientada a TPR@5% fue peor que BCE.
  - Un modelo SE con multitarea dio 0,540 interno y **0,079 en la validación oculta**, un sobreajuste dramático a la fuente.

**Otros relevantes.**
- **CinC 2025 #192 "On Spurious Features When Predicting Chagas" (team Blue and Gold).**
  - Un solo feature, la razón de potencia 50/60 Hz, da score 0,088 con regresión logística. Un árbol sobre espectro de 0 a 170 Hz da 0,179, con 60 Hz en el segundo nivel.
  - Con el test KS rechazaron casi todo lo que está por encima de 44 Hz como espurio.
  - Es evidencia directa de que el atajo de fuente existe en estos datos.
- **#477 MIWEAR.** Usa CV con ResNet-18 para estimar la confianza de cada muestra y **retiene solo positivos y negativos de alta confianza** (estilo *confident learning*) antes del fine-tuning de un FM de más de 10M ECG.
  - Su CV mejora de 0,308 a 0,400 [INT]. Test: 0,238 [EXT].
- **#077 ECG_ChD_25.** CV 0,34 [INT], validación 0,372, test 0,200 [EXT]. Excluyó los positivos de CODE-15% y usó focal más hinge de ranking.
- **#197 HeartGoesOut2U.** HuBERT-ECG a 100 Hz con cabeza CNN. CV: AUROC 0,785, AUPRC 0,143, score 0,364 [INT]. Test: 0,204 [EXT].
- **#108 ChagAI.** LoRA sobre la red de ECG-age de Ribeiro. Test: 0,202 [EXT].
- **#368 AIMED.** LiteVGG-11 a 128 Hz.
  - CV: **AUROC 0,842 ± 0,001, AUPRC 0,167, score 0,410** [INT]. Es casi idéntico a nuestro test interno.
  - Test: 0,240 [EXT].
- **Im H et al. (CAUETUMN).** *Physiol Meas* 2026, doi:10.1088/1361-6579/ae8b72. Combina ResNet-1D, BiGRU y *cross-attention* condicionada por edad, sexo y descriptores de QRS/conducción, con BCE ponderada por fuente.
  - Validación: 0,347. Test: 0,218, puesto 17/41 [EXT].
- **Wen H, Kang J.** "Reliability-aware hierarchical learning for Chagas disease screening…". *Physiol Meas* 2026;47(9), doi:10.1088/1361-6579/ae98e0.
  - Compara una CNN específica contra un FM espaciotemporal y supervisa según la procedencia de la etiqueta.
  - Test: 0,163 [EXT].
  - **Dos conclusiones explícitas.**
    - El rendimiento llega a un plateau impuesto por la fisiopatología.
    - Las distribuciones bimodales indican que los modelos separan cardiomiopatía establecida de infección indeterminada, que es indistinguible de los controles.

#### 1.4 Lecciones transversales del Challenge
- **La caída de interno a externo es universal.** Típicamente va de 0,40–0,55 [INT] a 0,20–0,32 [EXT]. Ningún equipo cerró la brecha.
- **Los conjuntos internos que incluyen SaMi-Trop o PTB-XL inflan la métrica.** DlaskaLabMUI obtuvo AUC 1,00 sobre SaMi-Trop+PTB-XL. Esto coincide con el diseño de arenas de DECA.
- **Técnicas con evidencia de ayuda en datos ocultos.**
  - Aumentación contra confusores de adquisición: red eléctrica, resampleo, filtro de banda.
  - Etiquetas suaves en CODE-15%.
  - Preentrenamiento grande. Los dos primeros puestos usan transformers preentrenados por SSL.
  - Ensembles: +0,026 en validación para DlaskaLabMUI.
- **Técnicas sin evidencia de ayuda externa o con evidencia en contra.**
  - Multitarea con BRD/BAV1 (#416).
  - Pérdidas que optimizan TPR@5% directamente (#416 negativo; #100 positivo solo en interno).
  - Modelos más grandes (#298, #158).
  - Entrenar solo con etiquetas fuertes (#416).
- **Demografía.** La usan el 1º, 6º, 7º y 8º puesto, pero **ningún paper presenta una ablación de su aporte en datos ocultos** [NV]. DlaskaLabMUI lo señala como pregunta abierta.

---

### 2. Grupo Ribeiro/Lima/Brant/Sabino (CODE, SaMi-Trop, REDS-II, ELSA) y trabajos clínicos

#### 2.1 Jidling et al. 2023
- Referencia: Jidling C, Gedon D, Schön TB, Di Lorenzo Oliveira C, Cardoso CS, Ferreira AM, Giatti L, Barreto SM, Sabino EC, Ribeiro ALP, Ribeiro AH. "Screening for Chagas disease from the electrocardiogram using a deep neural network". *PLOS Negl Trop Dis* 17(7):e0011118, 2023. doi:10.1371/journal.pntd.0011118.
- **Datos.**
  - **CODE completo**: 2,3M ECG de 1,56M pacientes, con 2,0% de ECG autorreportados como Chagas.
  - SaMi-Trop: 2.054 pacientes, **93,4% positivos** en la cohorte original. La versión del Challenge es 100% positiva.
- **Modelo.** ResNet1D a 400 Hz con 4.096 muestras. Batches 50% CODE y 50% SaMi-Trop. Ensemble de 15 semillas.
- **AUC-ROC.**

  | Conjunto | Tipo | AUC (IC 95%) |
  |---|---|---|
  | Validación CODE+SaMi-Trop | [INT] | 0,80 (0,79–0,82) |
  | REDS-II (631 pacientes, 55,8% ChD) | [EXT] | 0,68 (0,63–0,71) |
  | ELSA-Brasil (13.739, 2,0% ChD) | [EXT] | 0,59 (0,56–0,63) |
  | **Solo CCC como positivo**, REDS-II | [EXT] | **0,82 (0,77–0,86)** |
  | **Solo CCC como positivo**, ELSA | [EXT] | **0,77 (0,68–0,85)** |

- **ELSA, sensibilidad y especificidad** (dos configuraciones): sensibilidad 0,52 y 0,36; especificidad 0,77 y 0,76. Qué par corresponde a qué configuración no quedó claro en el texto recuperado [NV].
- **Grad-CAM.** Resalta el QRS tardío en los bloqueos de rama.
- **Conclusión de los autores.** El modelo detecta cardiomiopatía; detecta peor la etapa previa a la CCC. Hay que curar datasets mejores.

#### 2.2 Ribeiro ALP, Cardoso CS, … Ribeiro AH, Sabino EC (prospectivo)
- Referencia: "Opportunistic Screening for Chagas Disease Using an Artificial Intelligence-Enabled ECG: Prospective Evaluation of Feasibility and Diagnostic Accuracy". *Circulation* 2026;154(10):879–891. doi:10.1161/CIRCULATIONAHA.125.078949 (PMID 42576808).
- **Diseño.**
  - Tele-ECG de rutina en 107 municipios endémicos de Minas Gerais, entre 2023 y 2024.
  - Modelo **AI-ECG-EPI**: AI-ECG más 3 preguntas epidemiológicas.
  - Serología a todos los positivos y a una muestra 3:1 de negativos, con ponderación por probabilidad inversa.
- **Resultados** [EXT, prospectivo]:
  - 59.154 individuos; 11,5% dio positivo en el cribado.
  - Prevalencia ponderada de 12,2%.
  - Sensibilidad 34,3%, especificidad 91,6%, DOR 5,69.
  - **AUC de AI-ECG-EPI 0,776 contra 0,700 de las preguntas solas y 0,645 del AI-ECG solo.**
  - Mejor rendimiento en mujeres y en cardiomiopatía.
- **Relevancia para DECA.**
  - Es la mejor estimación disponible del rendimiento real del ECG solo para serología: AUC ~0,65.
  - La epidemiología aporta más que el ECG.
  - Su hallazgo "mejor en mujeres" va en sentido contrario a la disparidad por sexo que observó DECA. Puede deberse a otro umbral o a la población; conviene revisar el texto completo [NV].

#### 2.3 Fonseca IMG, Nunes MCP, … Ribeiro ALP
- Referencia: "Artificial Intelligence-Enhanced Electrocardiography Analysis for Diagnosing Chagas Disease". *Arq Bras Cardiol* 2026;123(5):e20260032. doi:10.36660/abc.20260032.
- **Diseño.** Campaña de campo en Feira de Santana (BA) con 1.115 adultos. AI-ECG más 3 variables clínicas. Serología a los marcados y a controles 2:1.
- **Resultados** [EXT]:
  - 121 marcados.
  - 13 positivos, todos en el grupo marcado.
  - Sensibilidad 100% (IC 69–100), especificidad 40%, VPP 12%.
- Con n = 13 positivos, la estimación es muy imprecisa.

#### 2.4 ECG-age
- Referencia: Lima EM, Ribeiro AH, Paixão GMM, … Ribeiro ALP. "Deep neural network-estimated electrocardiographic age as a mortality predictor". *Nat Commun* 12:5117, 2021. doi:10.1038/s41467-021-25351-7. La lista de autores no se verificó en la fuente; es [NV] parcial.
- **Correlación** entre ECG-age y edad cronológica: R² 0,71 en CODE-15% [INT], 0,32 en ELSA y 0,35 en SaMi-Trop [EXT].
- **Mortalidad en SaMi-Trop.** Un ECG-age ≥ 8 años por encima de la edad real da HR 2,42 (1,53–3,83) [EXT].
- **Relación con la competencia.** La red de ECG-age se usó como backbone con LoRA en el Challenge (#108), con un score de solo 0,202.

---

### 3. Modelos fundacionales de ECG y su uso en Chagas

#### 3.1 Modelos

| Modelo | Referencia | Preentrenamiento | ¿Usado en Chagas? |
|---|---|---|---|
| **ST-MEM** | Na Y et al., ICLR 2024, "Guiding masked representation learning to capture spatio-temporal relationship of ECG" [cita de memoria, NV en esta búsqueda] | MAE espaciotemporal | **Sí.** Es la receta de preentrenamiento del 1º puesto (400k ECG) |
| **ECG-FM** | McKeen K, Masood S, Toma A, Rubin B, Wang B. arXiv:2408.05178 (2024) | 1,5M ECG, transformer, híbrido contrastivo y generativo. Reporta AUROC 0,929 para FEVI ≤ 40% | **Sí.** Congelado en #416 (test 0,245); alineado con RMN en Alvarez-Florez 2026 |
| **ECGFounder** | Li J, Aguirre A, Moura J, … Clifford G, Westover B, Hong S. arXiv:2410.04133 (2024); versión de revista en NEJM AI [NV] | Más de 10M ECG de Harvard-Emory, 150 etiquetas supervisadas | Probablemente es el FM de "más de 10M" de MIWEAR (#477) [NV]; base de ECGNeXt en CinCo Amigos |
| **HuBERT-ECG** | Coppola E et al., medRxiv 10.1101/2024.11.14.24317328 | SSL estilo HuBERT | **Sí.** #197, test 0,204 |
| **xECG** (xLSTM) | Citado por DlaskaLabMUI | SimDINOv2 | **Sí.** Ensemble del 2º puesto |

- **Benchmark general.** Al-Masud MA, Lopez Alcaraz JM, Strodthoff N. "Benchmarking ECG FMs: A Reality Check Across Clinical Tasks". arXiv:2509.25095; publicado en ICLR 2026.
  - Evalúan 8 FMs en 26 tareas.
  - Solo 3 superan consistentemente al supervisado en interpretación de adultos.
  - Un modelo compacto (ECG-CPC, *state-space*) domina 5 de 7 categorías.
  - Persisten brechas en estructura cardíaca y pronóstico.
  - No incluye Chagas.

#### 3.2 Alineación ECG–imagen (lo más cercano a "usar la ecocardiografía")
- Referencia: Alvarez-Florez L, Uyterlinde D, Ruipérez-Campillo S, Arts LPA, Asselbergs FW, Tjong FVY. "Leveraging Cardiac Imaging to Improve ECG-Based Detection of Chagas Disease in Resource-Constrained Settings". arXiv:2609.08582 (septiembre 2026). Es un preprint.
- **Método.**
  - ECG-FM (capa 9) alineado por InfoNCE asimétrico con embeddings de RMN cardíaca (CineMA).
  - La RMN pasa por un cuello de botella de fenotipos: FEVI, VDF, masa del VI, GLS, FA, IAM.
  - Usa 63.193 pares ECG-RMN de UK Biobank.
  - En el despliegue solo se usa el ECG, con una sonda lineal.
- **CV CODE-15%+SaMi-Trop** [INT]: TPR@5% 0,427 y AUROC 0,851, contra 0,377 y 0,827 de ECG-FM sin alinear (+5 pts).
- **Con PTB-XL** [INT]: 0,431 y 0,852.
- **Test oculto del Challenge** [EXT]:

  | Conjunto | TPR@5% | AUROC |
  |---|---|---|
  | REDS-II | 0,350 | 0,739 |
  | SaMi-Trop 3 | 0,326 | 0,773 |
  | ELSA | 0,132 | 0,556 |
  | Media | 0,269 | — |

  - No supera al 1º puesto, aunque tiene el mejor AUROC en SaMi-Trop 3.
  - La entrega combinaba la sonda con un modelo ajustado de punta a punta.
- **Lectura.** La supervisión estructural (la verdad de imagen) mejora el interno en ~5 pts, pero no rompe el techo externo.

#### 3.3 Conclusión sobre los FMs
- Los FMs **son el ingrediente común de los puestos 1º y 2º**.
- Su ventaja externa frente a una buena CNN es pequeña: 0,323 / 0,283 contra 0,280 del 3º, una ResNet1D desde cero con buen manejo de etiquetas.
- **No hay estudio con ablación limpia "mismo pipeline, FM contra CNN" en datos externos de Chagas** [NV].

---

### 4. Etiquetas ruidosas o débiles, PU learning y serología frente a cardiomiopatía

#### 4.1 Lo aplicado específicamente a Chagas/ECG
- **Suavizado o etiquetas suaves solo en la fuente débil.** Lo usan el 1º puesto (0,8/0,2), el 2º (ε = 0,2) y el 3º (por discordancia y por BRD/BAV/FA). Es la técnica más consistentemente asociada al top-3.
- **Agregación por paciente de etiquetas discordantes.** AIChagas y Ahus AIM: 1.825 pacientes de CODE-15% tienen ECG etiquetados como positivo y como negativo.
- **ELR** (Liu S et al., NeurIPS 2020, "Early-Learning Regularization Prevents Memorization of Noisy Labels" [cita de memoria]). Lo usa CinCo Amigos, que fue el mejor en ELSA.
- **Curado por confianza** (estilo *confident learning*; Northcutt C et al., JAIR 2021 [cita de memoria]). Lo usa MIWEAR.
- **DivideMix** (Li J et al., ICLR 2020 [cita de memoria]). AIChagas lo probó y "no convergió o dio mejoras menores".
- **Supervisión jerárquica por confiabilidad de fuente.** Wen & Kang 2026 y Im et al. 2026.

#### 4.2 PU learning
- **No encontré ningún trabajo que formule Chagas-ECG explícitamente como PU learning** (con estimación de prior de clase, nnPU, etc.) [búsqueda en PubMed y web sin resultados].
- La formulación encaja bien: los negativos de CODE-15% son en realidad "no reportados", y la infección indeterminada no deja huella en el ECG.
- **Referencias base** [citas de memoria, NV en esta búsqueda]:
  - Elkan & Noto, KDD 2008.
  - Kiryo et al., "Positive-Unlabeled Learning with Non-Negative Risk Estimator", NeurIPS 2017.
  - Bekker & Davis, "Learning from positive and unlabeled data: a survey", *Mach Learn* 2020.
- **Advertencia.** PU learning corrige el sesgo de los negativos contaminados. No crea señal donde el ECG no la tiene: el límite fisiopatológico de §2.1 y §4.3 sigue en pie.

#### 4.3 Serología contra cardiomiopatía (el ruido "de constructo")
- **Evidencia convergente de que la etiqueta serológica es el cuello de botella**, no la capacidad del modelo:
  - Jidling: AUC 0,68 → 0,82 en REDS-II al pasar de "cualquier ChD" a "CCC".
  - Wen & Kang: bimodalidad y plateau.
  - Circulation 2026: el AI-ECG solo da AUC 0,645 con serología prospectiva.
  - AIChagas: "los seropositivos pueden no tener signos obvios en el ECG".
- Solo ~20–30% de los infectados desarrolla cardiomiopatía (citado por ISIBrno). Por eso el techo de sensibilidad de un detector ECG para serología está acotado por diseño.

---

### 5. AI-ECG para disfunción sistólica del VI aplicado a Chagas

**Base: Attia ZI et al. "Screening for cardiac contractile dysfunction using an artificial intelligence–enabled electrocardiogram".** *Nat Med* 25:70–74, 2019. doi:10.1038/s41591-018-0240-2.
- Una CNN entrenada con 44.959 pares ECG–eco de Mayo, para FE ≤ 35%.
- AUC 0,93 [INT, misma institución]. La sensibilidad de 86,3% y la especificidad de 85,7% son cita de memoria [NV].

**Brito BOF, Attia ZI, Martins LNA, Perel P, Nunes MCP, Sabino EC, Cardoso CS, Ferreira AM, Gomes PR, Ribeiro ALP, Lopez-Jimenez F. "Left ventricular systolic dysfunction predicted by artificial intelligence using the electrocardiogram in Chagas disease patients–The SaMi-Trop cohort".** *PLOS Negl Trop Dis* 15(12):e0009974, 2021. doi:10.1371/journal.pntd.0009974 (corrección doi:10.1371/journal.pntd.0010488).
- **Diseño.** Aplicaron el algoritmo de Attia **tal cual, sin reentrenar**, a 1.304 pacientes de SaMi-Trop.
  - El 7,1% tenía FEVI < 40% por eco (Simpson). El abstract dice ≤ 40%.
  - Los ECG se resamplearon a 500 Hz y se puntuaron en ventanas de 2 s.
- **Resultados** [EXT, población y país distintos al entrenamiento]:

  | Modelo | AUC | Sensibilidad | Especificidad | VPN |
  |---|---|---|---|---|
  | AI solo | **0,84** (0,79–0,88) | 0,73 | 0,83 | 0,97 |
  | + sexo masculino + QRS ≥ 120 ms | 0,85 | — | — | — |
  | + sexo + NT-proBNP | 0,87 | — | — | — |

  - El abstract reporta 0,839, 0,859 y 0,874, con discrepancias menores respecto del texto.
- Los datos figuran en github.com/samitrop/AI-ECG-Chagas [NV: no comprobé el contenido].

**Cardoso CS, … Attia ZI (coinventor), … Ribeiro ALP. "AI-ECG for Detecting Left Ventricular Systolic Dysfunction in Chagas Disease: Diagnostic and Prognostic Value".** *JACC: Advances* 5(7):102878, 2026. doi:10.1016/j.jacadv.2026.102878 (PMC13400128).
- **Diseño.** AI-ECG-2019 (población general) **fine-tuneado en REDS-II** produce AI-ECG-ChD, que se evalúa en SaMi-Trop.
- **Detección de LVSD** (SaMi-Trop, n = 1.304) [EXT respecto del fine-tuning]:

  | Modelo | AUC (IC 95%) |
  |---|---|
  | AI-ECG-ChD | **0,89** (0,85–0,93) |
  | AI-ECG-2019 | 0,86 (0,82–0,90) |
  | NT-proBNP | 0,90 (0,87–0,93), p = 0,52 |

  - AI-ECG-ChD positivo: sensibilidad 76%, especificidad 90%, VPP 37%, VPN 98%.
- **Mortalidad.** C-index a 9 años de 0,78 para AI-ECG contra 0,77 para NT-proBNP. HR univariado 5,3–5,5.
- **LVSD incidente a 7 años.** RR ajustado de 3,81 (2,31–6,27).
- **Inconsistencia interna.** La discusión cita 0,839 para AI-ECG-2019 y los resultados dicen 0,86.

**Otros.**
- Validación externa de AI-ECG de LVSD en Chagas con ECG en papel digitalizado por smartphone: *J Card Fail Intersections* 2025, https://www.onlinejcfi.org/article/S3050-6611(25)00038-3/fulltext. **No pude acceder** (HTTP 403), así que los números y autores son [NV].
- Abstract en *Eur Heart J* 46(Suppl 1) ehaf784.4411 (2025), "AI-ECG performance in detecting LVSD and predicting death in Chagas disease patients: SaMi-Trop project". No lo leí [NV].

**Sobre "SaMi-Trop LVSD".**
- No encontré un dataset público con ese nombre en Zenodo.
- La fuente más probable del set que DECA tiene en `samitrop-lvsd/x/` es el repositorio de Brito 2021 (github.com/samitrop/AI-ECG-Chagas) [NV].
- Las trampas que DECA ya documentó (300 Hz, orden de derivaciones distinto) son consistentes con un export distinto del de Zenodo 4905618.

**Lectura para DECA.**
- Con **etiqueta ecocardiográfica**, el AI-ECG logra AUC 0,84–0,89 **externo**, incluso sin entrenar en Chagas (0,84 a 0,86).
- Es la única familia de resultados que supera el umbral AUC 0,85 fuera de muestra.
- Pero responde otra pregunta: disfunción del VI en un chagásico conocido, con prevalencia del 7%. No responde "quién es seropositivo en la población".

---

### 6. ¿Hay resultados por encima de AUC 0,85 / TPR@5% 0,40 en test externo?

| Resultado | Etiqueta | Tipo | AUC | TPR@5% | ¿Supera ambos? |
|---|---|---|---|---|---|
| Biomed-Cardio, REDS-II test | serología | EXT | 0,777 | **0,468** | Solo TPR, en un subconjunto |
| Biomed-Cardio, REDS-II val | serología | EXT | 0,793 | 0,445 | Solo TPR |
| ISIBrno, REDS-II val | serología | EXT | 0,774 | 0,458 | Solo TPR |
| Ensemble de organizadores, REDS-II | serología | EXT | NV | 0,389 | No |
| Mejor ELSA (CinCo Amigos / AIChagas) | serología | EXT | 0,62–0,63 | 0,16 | No |
| Jidling, CCC en REDS-II | cardiomiopatía | EXT | 0,82 | — | No (cerca) |
| Brito 2021, LVSD en SaMi-Trop | eco FEVI < 40% | EXT | 0,84 | — | No (cerca) |
| Cardoso 2026, LVSD en SaMi-Trop | eco FEVI ≤ 40% | EXT* | **0,89** | — | Sí en AUC; otra pregunta clínica |
| Alvarez-Florez 2026 (CMR-aligned) | serología | INT | 0,852 | 0,431 | Solo interno |
| Biomed-Cardio CV | mixta | INT | 0,867 | 0,490 | Solo interno, pool inflado |
| Circulation 2026, AI-ECG solo | serología, prospectivo | EXT | 0,645 | — | No |

\*Externo respecto de la cohorte de fine-tuning (REDS-II). El modelo base es de Mayo.

**Conclusión.**
- Con **serología** como verdad y **validación externa**, ningún trabajo publicado llega a AUC 0,85.
- El TPR@5% ≥ 0,40 solo aparece en REDS-II, una cohorte de donantes de sangre seropositivos con alta carga de cardiomiopatía, y con AUROC < 0,80.
- Las cifras altas se obtienen de dos maneras:
  - **Cambiando el constructo** a cardiomiopatía o disfunción del VI, o
  - **Midiendo internamente**, sobre un pool donde SaMi-Trop y PTB-XL infla la separabilidad.
- Esto es coherente con lo que DECA ya concluyó en Fase 5. El "techo" de DECA es el techo del campo.

---

### 7. Fuentes no verificadas o pendientes
- El texto completo de *Physiol Meas* 2026 (paper oficial del Challenge), por el desempeño del ensemble en SaMi-Trop 3.
- El artículo de smartphone en *J Card Fail Intersections* 2025 (HTTP 403).
- La sensibilidad y especificidad de Attia 2019, y la versión de revista de ECGFounder: citadas de memoria.
- ST-MEM, ELR, DivideMix, confident learning y las referencias de PU learning: citadas de memoria, sin reabrirlas en esta sesión.
- El contenido de github.com/samitrop/AI-ECG-Chagas y su relación con `samitrop-lvsd/x/`.
- Si "mejor en mujeres" (Circulation 2026) se refiere al mismo umbral y métrica que la disparidad que midió DECA.

---

# Anexo C — Epidemiología y tamizaje en Argentina

## Chagas en Argentina: epidemiología, tamizaje y lugar de un priorizador ECG-IA

*Revisión de literatura para el módulo IA de DECA. Fecha de búsqueda: 2026-10-14.*

**Cómo leer este informe.** Cada cifra lleva su fuente. Las cuentas propias se marcan como **[cálculo propio]**. Cuando algo no se encontró, se dice. Las fuentes secundarias (prensa, ONG) se señalan como tales y no reemplazan a una fuente primaria. Varias fuentes en PMC no se pudieron abrir completas, porque PMC devolvía un captcha. En esos casos se usó el resumen indexado en Europe PMC. Por lo tanto, algunas cifras salen del *abstract* y no del texto completo.

**Referencia DECA.** El checkpoint congelado `patrones-lr8` deriva al 1,4% de la población con un VPP de 29,5%. La prevalencia de la arena A es de alrededor del 1,9%, según CLAUDE.md.

---

### 1. Prevalencia, subdiagnóstico y progresión

#### 1.1 Magnitud nacional

| Indicador | Valor | Fuente |
|---|---|---|
| Infectados en Argentina | ~1.500.000 (3,7% de la población). Más de 370.000 con cardiopatía chagásica | Klein K, Burrone MS, Alonso JP, …, Sosa-Estani S. *Rev Panam Salud Pública* 2017;41:e20. https://www.scielosp.org/article/rpsp/2017.v41/e20/ |
| Estimación OMS 2010 | 1.505.235 infectados. Argentina es el país con más infectados estimados, por delante de Brasil (1.156.821) y México (876.458) | OMS, *Weekly Epidemiological Record* 2015;90(6):33-44 (citado por Mundo Sano y reviews; la tabla original no se verificó) |
| Población en riesgo | ~7,3 millones viven en áreas con vector, según una estimación OMS citada por Mundo Sano | Mundo Sano, 2021. https://mundosano.org/en/chagas-the-neglected-disease-affecting-1-5-million-people-in-argentina-20-of-the-total-of-infected-people-in-the-world/ (secundaria) |
| Recién nacidos infectados por año | ~1.300-1.500 | Danesi E, Codebó MO, Sosa-Estani S. *Medicina (B Aires)* 2019;79(2):81-89. https://www.medicinabuenosaires.com/indices-de-2010-a-2019/volumen-79-ano-2019-no-2-indice/transmision-congenita-de-trypanosoma-cruzi/ |

Las cifras de 1,4 a 1,6 millones que circulan derivan todas de estimaciones de 2006-2010 (OPS/OMS). **No encontré una estimación nacional de prevalencia posterior a 2015 basada en datos nuevos.** Hay además una cifra de "2,33 millones (7,2%)" atribuida a la SAC en una fuente secundaria que no pude verificar, por lo que no la uso.

#### 1.2 Por grupo poblacional y provincia

**Personas gestantes.** Es el grupo con mejor dato nacional, porque el tamizaje es obligatorio por la Ley 26.281.

- Serie histórica del subsector público: la tasa de infección materna bajó de **9,0% (1997) a 2,6% (2014)**. Por estrato de riesgo vectorial (trienio 2010-2012) fue de 6,0% en provincias de alto riesgo, 3,1% en las de riesgo medio y 1,5% en las de bajo riesgo. Chaco tuvo la mayor tasa en 2014, con **10,4%**. La cobertura de tamizaje en el sector público subió de 36,6% (2002) a 60,3% (2014). (Danesi, Codebó, Sosa-Estani 2019, ref. arriba)
- La Auditoría General de la Nación (2015) informó 309.248 gestantes tamizadas, de las cuales 7.794 eran positivas (**2,5%**). Citado en Danesi et al. 2019.
- **Boletín Epidemiológico Nacional N°743 (SE 6, 2025)**, informe "Chagas en embarazadas y Chagas congénito 2019-2024". https://www.argentina.gob.ar/sites/default/files/2025/01/ben_743_se_6.pdf
  - El NEA tiene la positividad más alta en embarazadas estudiadas: **6,4%**, que baja a **2,3% en 2023**. NOA, Cuyo, Centro y Sur tienen menor positividad. En zonas sin vector, la positividad se atribuye a la migración.
  - Se notificaron 15.478 casos confirmados entre 2019 y 2024 (promedio de 2.571 por año). La razón nacional fue de 5,6 casos por 1.000 nacidos vivos en 2019 y de 3,6 en 2024.
  - Razón por 1.000 nacidos vivos, por provincia (2019 → 2024): **Chaco 46,4 → 20,9; Formosa 29,1 → 5,3; Santiago del Estero 7,2 → 17,6 (pico de 45,3 en 2022); Salta 14,5 → 5,0; San Juan 15,1 → 6,8; Mendoza 8,8 → 3,4**. CABA estuvo en 2,6-5,0, Buenos Aires en 1,3-2,1 y las provincias del Sur en 0-9,5, con números pequeños. Esta razón **no es una prevalencia**: depende de la notificación nominal, que el propio BEN reconoce como incompleta.
  - En 2024, la razón por edad sube con la edad materna: 2,9 en el grupo de 15-19 años y 8,5 en el de 45-49.
- Positividad reciente según la gacetilla del BEN SE 38 de 2025: **0,98%** hasta la SE 36 de 2025 y **1,05%** en el mismo período de 2024. https://www.argentina.gob.ar/noticias/actualizacion-del-boletin-epidemiologico-nacional-de-la-se-ndeg-38 (resultado de búsqueda; el PDF no se verificó y el texto tiene una inconsistencia entre 1.180 y 1.536 positivos)
- La proporción de embarazadas sin estudio serológico de Chagas bajó de 60% (2019) a 49% (2024) y subió a 52% en 2025. Fuente: resumen de los BEN de 2025-2026 recuperado por búsqueda, sin verificar en el PDF.
- Córdoba: la seroprevalencia en embarazadas bajó de 4,27% (2008) a 0,86% (2022), según el Gobierno de Córdoba. https://www.cba.gov.ar/dia-mundial-de-chagas-abordaje-integral-para-interrumpir-la-transmision-vectorial-y-controlar-la-perinatal/ (fuente oficial provincial)
- Neuquén: 0,50% en gestantes (2016). Guía provincial 2023. https://salud.neuquen.gob.ar/wp-content/uploads/2024/09/Chagas-Guia-2023.pdf

**Transmisión congénita.** La tasa nacional media de la serie 2002-2014 fue de 4,2% (rango anual 1,9-8,2%). El subregistro se estima en 1 caso detectado por cada 6 a 13 casos esperados (Danesi et al. 2019). Entre 2019 y 2024 se notificaron 8.987 casos sospechosos de Chagas congénito y se confirmó solo un 3-6% por año, con alta pérdida de seguimiento (BEN 743).

**Donantes de sangre.**
- Buenos Aires, 1995-1999: 64.887 donaciones, con 2,66% reactivas al tamizaje y **1,46% confirmadas**. Blejer JL, Saguier MC, Salamone HJ. *Int J Infect Dis* 2001;5(2):89-93. doi:10.1016/S1201-9712(01)90032-3
- Panorama regional 2012-2023 en 21 países endémicos: la mediana de prevalencia en donantes bajó de 0,45% (2012) a 0,16% (2023). En Argentina, la prevalencia en donantes de órganos y tejidos fue de 0,9%. Beltrán M, Remesar M, Menjivar A, Barco G. *Rev Panam Salud Pública* 2026;50:e5. doi:10.26633/RPSP.2026.5
- **No encontré una serie reciente de prevalencia en donantes de sangre por provincia argentina.** Probablemente existe en los informes del Plan Nacional de Sangre, pero no está publicada de forma accesible.

**Conscriptos (tendencia histórica).** En jóvenes de 18 años, la seroprevalencia fue de 10,1% (1965), 5,8% (1981) y 1,9% (1993) (Danesi et al. 2019).

**Adultos y jóvenes de áreas endémicas rurales.** En los focos rurales, la prevalencia sigue siendo un orden de magnitud mayor que la nacional.
- **Pampa del Indio (Chaco), población qom y criolla.** Antes de la intervención (2008) la seroprevalencia era de **29,0%** (n=1.373). Fernández MDP, Gaspe MS, Sartor P, Gürtler RE. *PLoS NTD* 2019;13(12):e0007430. doi:10.1371/journal.pntd.0007430. Seis a nueve años después del control vectorial era de **24,8%** (n=1.337). Factores asociados: tiempo de exposición a triatominos, número de convivientes seropositivos, madre seropositiva, residencia reciente en la vivienda y paredes aptas para la colonización. Macchiaverna NP, Enriquez GF, Gaspe MS, …, Cardinal MV. *Parasit Vectors* 2024;17:41. doi:10.1186/s13071-024-06125-8
- **Pampa del Indio, menores de 18 años:** 24,1% (n=395). Sartor P, Colaianni I, Cardinal MV, …, Gürtler RE. *PLoS NTD* 2017;11(2):e0005336. doi:10.1371/journal.pntd.0005336
- **Área rural del NEA, jóvenes de 10 a 20 años:** **9,33%** (n=460). Ningún infectado conocía su condición. La presencia de *T. infestans* en el domicilio se asoció con OR 2,08. **No encontraron asociación entre bloqueo de rama derecha (BRD) y Chagas en esta población joven.** Baeck IM, Mando P, Virasoro BM, …, Lopez Saubidet I. *Am J Trop Med Hyg* 2022;107(1):146-150. doi:10.4269/ajtmh.21-0646
- **Robles (Santiago del Estero), algoritmo de tamizaje en terreno:** 48 confirmados entre 1.380 testeados, es decir **~3,5%** [cálculo propio]. Marchiol A, Mendicino D, Thelin A, …, Seu SDV. *Rev Fac Cienc Med Córdoba* 2026;83(2). doi:10.31053/1853.0605.v83.n2.50102
- **Añatuya (Santiago del Estero).** Intervención ecosanitaria de 14 años. La infestación intradomiciliaria bajó de 17,9% a 0,2%. Weinberg D, Casale MF, Cejas RG, …, Abril MC. *PLoS NTD* 2023;17(6):e0011410. doi:10.1371/journal.pntd.0011410

**Migrantes y población urbana.**
- **Hospital San Juan de Dios, La Plata, 2006-2008.** Seroprevalencia de 11,4% en 2.921 muestras de personas a quienes se les *solicitó* serología, lo que implica un fuerte sesgo de selección. Por país de nacimiento, los seropositivos eran 47,8% argentinos, 43,5% bolivianos y 8,3% paraguayos. El 60,4% tenía necesidades básicas insatisfechas y el 49,7% consultó por síntomas. Corallini JC, Fernández O, et al. *Acta Bioquím Clín Latinoam* 2011;45(3). https://www.scielo.org.ar/scielo.php?script=sci_arttext&pid=S0325-29572011000300004
- **La Matanza (AMBA), 2023-2024.** Consultorio de Chagas reactivado, con 61 pacientes. Edad media alta y mucha miocardiopatía al diagnóstico, por lo que la mayoría ya no era candidata a tratamiento tripanocida. 35% se perdió en el seguimiento. No hay datos epidemiológicos oficiales del distrito. Vilariño A, Castañeda B, Warley E. *Medicina (B Aires)* 2025;85(6):1307-1311.
- **No encontré un estudio poblacional reciente de seroprevalencia en adultos migrantes del AMBA.** Lo que existe son series hospitalarias de personas derivadas o que consultaron.

#### 1.3 Proporción que no sabe que está infectada

- **No hay una cifra específica para Argentina medida con denominador propio.**
- OPS (2021): **el 70% de las personas con Chagas no sabe que está infectada**. https://www.paho.org/en/news/13-4-2021-70-people-chagas-dont-know-theyre-infected
- OPS (2022-2023): **menos del 10%** de los infectados recibe diagnóstico y tratamiento oportunos. https://www.paho.org/en/news/13-4-2023-less-10-people-chagas-receive-diagnosis. Esta cifra mide diagnóstico y tratamiento a tiempo, que no es lo mismo que desconocer la infección. Trabajos argentinos (Marchiol 2026; artículo sobre La Matanza) repiten "10% diagnosticados, 1% tratados", con origen en Sanmartino et al. 2015 y OPS.
- Evidencia local puntual: en el NEA, **0%** de los jóvenes infectados conocía su condición (Baeck 2022). En Feira de Santana (Brasil), el 23% de los seropositivos no lo sabía (Jesus et al. 2026, ver §3).
- En el sistema de salud, solo el 45,4% de 652 pacientes con miocardiopatía chagásica atendidos en varias instituciones argentinas (2018-2021) tenía hechos el ECG, la radiografía de tórax y el ecocardiograma. Chuit R, Antonietti L, Agüero RN, …, Pizzi R. *Front Parasitol* 2023;2:1195646. doi:10.3389/fpara.2023.1195646

#### 1.4 Progresión a cardiomiopatía

| Fuente | Cifra |
|---|---|
| Guía nacional (Programa Nacional de Chagas / INP Fatala Chabén, 2012, RM 1337/14) | ~**30%** de los infectados crónicos desarrolla lesión de órgano, sobre todo cardíaca, en 10-20 años. **20-30%** evoluciona a cardiopatía y **~10%** a la forma dilatada. https://www.argentina.gob.ar/sites/default/files/2021/07/chagas.-guia-para-equipos-de-salud.pdf |
| Rassi A Jr, Rassi A, Marin-Neto JA. *Lancet* 2010;375:1388-1402. doi:10.1016/S0140-6736(10)60061-X | **30-40%** tiene o tendrá cardiomiopatía, megavísceras o ambas |
| Chadalawada S, et al. *JAMA Netw Open* 2020 (metaanálisis de 32 estudios). https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2770045 | Desde la forma indeterminada: **1,9% por año** (IC95% 1,3-3,0; I²=98%). Desde la fase aguda: 4,6% por año |
| Nunes MCP, Buss LF, Silva JLP, …, Sabino EC. *Circulation* 2021;144:1553-1566. doi:10.1161/CIRCULATIONAHA.121.055112 (cohorte de donantes REDS-II) | Incidencia de cardiomiopatía de **13,8/1.000 persona-años** en seropositivos contra 4,6 en seronegativos (diferencia de 9,2/1.000 persona-años). Mortalidad de seropositivos sin cardiomiopatía basal igual a la de los seronegativos (3,7 contra 3,6/1.000) |
| Basquiera AL, Sembaj A, Aguerri AM, …, Salomone OA. *Heart* 2003;89:1186-1190. doi:10.1136/heart.89.10.1186 (Córdoba, n=56) | 21,4% progresó en ~2,6 años. **Sexo masculino: RR 5,0** (IC95% 1,65-15,7). PCR positiva: RR 4,1 |
| Viotti R, et al. (Hospital Eva Perón). *Ann Intern Med* 2006;144:724-734. doi:10.7326/0003-4819-144-10-200605160-00006 | 566 pacientes (283 tratados con benznidazol y 283 sin tratar), seguimiento de ~9,8 años. Predictores de progresión: edad, síntomas y **ECG anormal**. Empeoramiento del ECG en 4,2% de los tratados contra 14,1% de los no tratados (cifras de fuente secundaria) |
| Sosa-Estani S, Viotti R, Segura EL. *Mem Inst Oswaldo Cruz* 2009;104(Supl 1). doi:10.1590/S0074-02762009000900023 | En la cohorte argentina, **BRD o HBAI** duplica aproximadamente la probabilidad de progresión. La TV sostenida previa la triplica |

**Para DECA.** Con 1,5 millones de infectados y ~30% de progresión, hay del orden de 370.000 a 450.000 personas con cardiopatía chagásica en Argentina [cálculo propio, coherente con Klein 2017]. La progresión es lenta (≈2% anual) y es más frecuente en hombres. Esto concuerda con la disparidad por sexo que DECA midió: se detecta menos a las mujeres, por la menor prevalencia de BRD.

---

### 2. Programas de tamizaje, costos y costo-efectividad

#### 2.1 Marco legal y programa

**Ley 26.281 (2007)**, de prevención y control de todas las formas de transmisión del Chagas. http://leg.msal.gov.ar/atlas/dja/LEY_C_026281.pdf
- La serología es obligatoria en **embarazadas, recién nacidos, hijos de madres infectadas** hasta el año de vida y el resto de los hijos menores de 14 años de esas madres. También en niños al cumplir **6 y 12 años**, según lo establezca la autoridad de aplicación.
- Son obligatorios los controles en **donantes y receptores de sangre, órganos y tejidos**. Si un donante es reactivo, hay que informarle y orientarlo al tratamiento.
- **El diagnóstico y el tratamiento son gratuitos en todos los establecimientos públicos.**
- **Está prohibido hacer serología de Chagas a aspirantes a empleo.** Esto es importante para DECA: un ECG preocupacional con IA que infiera Chagas sería, como mínimo, éticamente y legalmente delicado.
- Los resultados solo se usan con fines preventivos y de tratamiento, bajo la Ley 25.326 de datos personales.

**Algoritmo diagnóstico.** La infección crónica se confirma con **dos técnicas serológicas distintas** (ELISA, HAI o IFI), y una tercera si son discordantes. Guía nacional de 2012 y Forsyth et al. 2022 (ver §4).

**Búsqueda activa en adultos: prácticamente no existe.**
- "A excepción de tamizajes en mujeres embarazadas, no se realiza búsqueda activa en la población de adultos o niños" (Klein et al. 2017, ref. arriba).
- Un estudio cualitativo en el primer nivel de una zona endémica encontró que en los CAPS solo se indica serología a embarazadas. SciELO Brasil, *Interface* (Botucatu). http://www.scielo.br/j/icse/a/psz7xc94wKSQmjk5S3Dn7cN/?lang=es
- Iniciativas puntuales:
  - Mundo Sano, UNSa y el Ministerio de Salud de Salta: tamizaje masivo de ingresantes universitarios (NCT07033325, 2024-2025). No encontré resultados publicados.
  - Campaña "Saber si tenés Chagas es tu derecho" (2024).
  - Pampa del Indio: un programa de atención primaria con participación social logró 70% de cobertura serodiagnóstica y **82% de cobertura de tratamiento** (Sartor 2017).
- Gobernanza: según La Nación (abril de 2024), el Programa Nacional de Chagas quedó sin conducción a partir de diciembre de 2023 (fuente periodística).

#### 2.2 Costo de la serología

- **No encontré un arancel oficial vigente** en pesos ni en dólares para la serología de Chagas en Argentina. El nomenclador bioquímico de PAMI trabaja en Unidades Bioquímicas, y no pude recuperar las UB asignadas al código anti-*T. cruzi*. https://www.pami.org.ar/pdf/manual_nomenclador.pdf
- Única referencia de precio de insumo: un kit ELISA de lisado de Wiener Lab de 96 determinaciones, publicado en un marketplace a ~$1.135.131, es decir **~$11.800 de reactivo por muestra** [cálculo propio]. La fecha de la publicación no está clara y el monto no incluye personal, equipamiento ni la segunda técnica obligatoria. **No es citable como costo.**
- En el sector público, el costo para el paciente es cero por ley. El costo relevante para el sistema es el de las dos técnicas más la extracción, el traslado de la muestra y el seguimiento. Que la persona vuelva a buscar el resultado es probablemente un cuello de botella mayor que el reactivo: Danesi 2019 estima una pérdida de 55-80% de los niños expuestos antes de los 10 meses.

#### 2.3 Costo-efectividad del tamizaje

- **No encontré ningún estudio de costo-efectividad del tamizaje serológico de adultos en Argentina.**
- El estudio argentino de costo-efectividad más citado es de **control vectorial**, no de tamizaje: US$132, 82 y 45 por caso humano evitado para las estrategias vertical, mixta y horizontal en el noroeste argentino. Vazquez-Prokopec GM, Spillmann C, Zaidenberg M, Kitron U, Gürtler RE. *PLoS NTD* 2009;3(1):e363. doi:10.1371/journal.pntd.0000363
- **Requena-Méndez A, Bussion S, Aldasoro E, …, Sicuri E. *Lancet Glob Health* 2017;5(4):e439-e447.** doi:10.1016/S2214-109X(17)30073-6. Modelo de Markov del tamizaje serológico de migrantes latinoamericanos en atención primaria en Europa, con prevalencia base de 4,2%. ICER determinístico de **€5.442 por AVAC**. Seguía siendo costo-efectivo (<€30.000 por AVAC) **aun con una prevalencia de 0,05%**.
- Tamizaje congénito en zona no endémica: tamizar a gestantes latinoamericanas y a sus recién nacidos fue más efectivo y menos costoso que no tamizar. *Nat Commun* 2025. https://www.nature.com/articles/s41467-025-63760-0 (no se verificó el detalle)

#### 2.4 Costo-efectividad de un tamizaje basado en ECG

- **No encontré ningún modelo económico publicado que evalúe una estrategia "ECG primero" (convencional o con IA) para Chagas.** Lo más cercano:
  - Spinicci M, Fumagalli C, Maurizi N, …, Bartoloni A. *Microorganisms* 2021;9(9):1889. doi:10.3390/microorganisms9091889. ECG móvil de una derivación más test rápido en el Chaco boliviano (n=140, 70% positivos). Costo de implementación **menor a US$10 por persona**. Es un estudio de factibilidad, no de costo-efectividad.
  - La Sociedad Americana de Ecocardiografía (2018) recomienda el ECG como estudio costo-efectivo **después** de la serología, no antes.
- **Esto es un hueco real en la literatura y una oportunidad para DECA.**

---

### 3. Factores de riesgo clínico-epidemiológicos y scores

#### 3.1 Criterios oficiales de sospecha (Argentina)

La guía nacional de 2012 indica sospechar infección crónica en quien:
1. resida o haya residido en una zona endémica, con o sin contacto conocido con el vector;
2. tenga **madre biológica infectada**;
3. haya recibido **transfusiones** o hemoderivados;
4. sea o haya sido usuario de drogas inyectables;
5. tenga síntomas o signos compatibles;
6. tenga un **familiar cercano con cardiopatía o muerte súbita** a edad temprana, o con serología reactiva.

#### 3.2 Evidencia cuantitativa por factor

| Factor | Evidencia |
|---|---|
| Vinchuca en la vivienda | OR 2,08 en el NEA (Baeck 2022). RP 4,38 (IC95% 1,44-13,3) en Bahía (Jesus 2026) |
| Madre seropositiva / convivientes seropositivos | Asociados en Pampa del Indio (Macchiaverna 2024). Los índices por hogar identificaron 97-99% de los hogares con niños seropositivos |
| Etnia y vivienda | En qom, la seroprevalencia era aproximadamente el doble que en criollos (29,7% contra 18,7%) (Fernández 2019) |
| Origen del migrante | En La Plata, el 43,5% de los seropositivos era de origen boliviano (Corallini 2011) |
| Edad | La positividad en gestantes sube con la edad materna (BEN 743). La infección crónica se acumula con la edad |
| Sexo | Mayor progresión a cardiomiopatía en hombres (Basquiera 2003, RR 5,0). La AHA incluye el sexo masculino como factor de progresión |
| Transfusión | Hoy el riesgo es residual, por el tamizaje universal de donantes. Es relevante en adultos transfundidos antes de los años 90 |

#### 3.3 Scores y cuestionarios validados

- **Brasil PEAA, Xavier SS, Holanda MT, Hasslocher-Moreno AM, Braga JU. *Rev Soc Bras Med Trop* 2016;49(3).** doi:10.1590/0037-8682-0196-2016. Score diagnóstico de Chagas crónico desarrollado en INI-Fiocruz (n=602, prevalencia de 19,9%) con validación temporal (n=138, prevalencia de 17,4%).
  - Predictores: sexo, edad, derivación desde banco de sangre, vida rural, **reconocer la vinchuca en fotos**, hipertensión, número de hermanos con Chagas, familiares con ACV, **bajo voltaje, HBAI, onda Q patológica, BRD y extrasístoles**.
  - **AUC de 0,904 en desarrollo y 0,912 en validación.**
  - Umbrales: **riesgo <7% descarta** (sensibilidad ≥95%) y **riesgo >43% confirma** (especificidad ≥95%). Hay una calculadora en línea.
  - Limitación: es una población de centro de referencia con prevalencia de ~20%, no de atención primaria. Incluye además los mismos patrones de ECG que DECA aprende.
- **AI-ECG-EPI (TNMG, Brasil).** Tres preguntas epidemiológicas: residencia en zona con triatominos, vivienda precaria y antecedente familiar de Chagas. Las preguntas solas tuvieron **AUROC de 70,0%**, el ECG solo 64,5% y la combinación con IA-ECG 77,6% (Ribeiro 2026, §6).
- **No encontré un cuestionario de riesgo validado contra serología en atención primaria argentina.**

---

### 4. Umbrales de probabilidad pretest y de derivación

No hay un umbral "oficial" de probabilidad pretest para indicar serología de Chagas. Las referencias que sirven para anclar el umbral son estas:

| Referencia | Probabilidad pretest con la que ya se recomienda testear o se considera razonable |
|---|---|
| Ley 26.281: tamizaje universal en embarazadas | Positividad nacional actual de **~1%** (0,98-1,05% en 2024-2025) |
| EE.UU. (Forsyth C, et al. *J Infect Dis* 2022;225(9):1601-1610, doi:10.1093/infdis/jiab513) | Testear a toda persona nacida en Latinoamérica continental o que haya vivido allí más de 6 meses, con una prevalencia comunitaria de **1-3,8%**. El antecedente familiar o un ECG anormal elevan el riesgo |
| Europa (Requena-Méndez 2017) | Costo-efectivo hasta una prevalencia de **0,05%** |
| Score de Brasil 2016 | <7%: descartar. 7-43%: zona gris en la que se recomienda serología. >43%: alta probabilidad |
| Moody Challenge 2025 (Reyna MA, …, Ribeiro ALP, Sameni R, Clifford GD. arXiv:2510.02202) | Métrica: TPR en el **5% superior**, que es la cota estimada de capacidad de serología en Brasil. Datos de prueba ajustados a una prevalencia de ~2% (ELSA-Brasil 2,04%) |

**[Cálculo propio] Comparación con DECA:**

| Estrategia | % derivado | VPP | Serologías por caso hallado | Lift sobre la prevalencia |
|---|---|---|---|---|
| Serología universal (prevalencia ~1,9%) | 100% | 1,9% | ~53 | 1× |
| Serología universal en gestantes (~1%) | 100% | ~1% | ~100 | — |
| **DECA `patrones-lr8`** | **1,4%** | **29,5%** | **~3,4** | **~15×** |
| TNMG AI-ECG-EPI (Ribeiro 2026) | 11,5% | 36,8% entre los derivados testeados (prevalencia ponderada de 12,2%) | ~2,7 | ~3× |
| Feira de Santana (Fonseca 2026) | 10,9% | 12% | ~8 | — |

Si la prevalencia es de 1,9%, el corte de DECA en 1,4% captura aproximadamente **0,014 × 0,295 / 0,019 ≈ 22%** de los infectados [cálculo propio, aproximado]. Esto es coherente con el TPR@5% de 37,2% medido.

**Interpretación.**
1. Un VPP de 29,5% está **muy por encima** de cualquier umbral al que hoy ya se indica serología. Quien cae en la banda alta de DECA tiene una probabilidad pretest comparable a la zona gris alta del score de Brasil y **30 veces mayor** que la de una gestante a la que la ley ya obliga a testear. Para quien entra en la banda, la derivación está plenamente justificada.
2. Lo contrario **no vale**. Quedar fuera de la banda no baja la probabilidad lo suficiente para no testear. Con una sensibilidad de ~22%, cerca de 4 de cada 5 infectados quedan afuera. Las personas con criterios epidemiológicos de la guía nacional (§3.1) deben recibir serología **independientemente** del resultado del ECG. DECA es un "rule-in" oportunista, no un "rule-out".
3. En Argentina la serología es gratuita y relativamente barata, así que la restricción operativa no es el costo del test sino la **captación y el seguimiento**. En TNMG completó la serología solo el ~45% de los invitados (3.509/7.804) [cálculo propio]. El valor de DECA está en encontrar a quien **nadie iba a testear**, porque llegó al ECG por otro motivo, y en hacerlo con un VPP alto que justifica el esfuerzo de llamarlo.

---

### 5. Fuentes argentinas con ECG y serología que podrían solicitarse

**No encontré ningún dataset argentino público con ECG digital crudo (señal) y serología.** Lo que existe son cohortes clínicas cuyos registros podrían o no conservar el ECG digital. Hay que preguntar en cada caso si guardan **señal** o solo el informe o el papel.

| Fuente | Qué tiene (según la literatura) | Contacto institucional / referentes |
|---|---|---|
| **Hospital Interzonal General de Agudos Eva Perón (San Martín, Pcia. de Bs. As.), Sección Chagas** | La cohorte longitudinal más larga del país: cientos de adultos con serología confirmada, clasificación de Kuschnir, ECG, eco y ergometría seriados desde los años 80 (Viotti 2006: 566 pacientes, ~10 años). Trabaja con el INP Fatala Chabén y CONICET (Laucella) | Grupo Viotti / Álvarez MG / Lococo B; Laucella SA (INP-CONICET) |
| **Instituto Nacional de Parasitología "Dr. Mario Fatala Chabén" (ANLIS-Malbrán)** | Centro nacional de referencia diagnóstica y Centro Colaborador OPS/OMS. Cohorte de Chagas congénito (499 niños, 2014-2018) y pacientes adultos | Dirección del INP. Proceso vía ANLIS |
| **Hospital Privado Universitario de Córdoba** | Cohorte de Basquiera/Salomone (progresión, PCR y sexo) | Salomone OA |
| **Hospital Papa Francisco (Salta)** | 400 evaluados y 110 con Chagas confirmado por dos serologías, todos con ECG de 12 derivaciones (2013-2016). El patrón más frecuente fue HBAI+BRD. Nuñez Burgos A, et al. *Insuf Card* 2016;11(4). https://ri.conicet.gov.ar/handle/11336/106637 | Servicio de Cardiología |
| **Pampa del Indio (Chaco), UBA-CONICET (Laboratorio de Eco-Epidemiología, Gürtler / Cardinal)** | Serología poblacional (>1.300 personas, qom y criollos) con datos de vivienda y hogar. No consta ECG | Gürtler RE, Cardinal MV |
| **Fundación Mundo Sano** (Añatuya, Santiago del Estero; Salta) | Intervenciones territoriales con diagnóstico y tratamiento; tamizaje de ingresantes de la UNSa (2024-2025) | Abril MC, Weinberg D |
| **Robles (Santiago del Estero)** | 1.380 testeados con algoritmo en terreno (Marchiol 2026) | Grupo de la Universidad Nacional de Santiago del Estero / Mendicino D |
| **Registro CONAREC XXII (2024)** | Registro multicéntrico transversal de adultos internados con serología positiva. Solo publicado como protocolo. Arri ML, …, Zaidel E. *Rev CONAREC* 2024;173 | CONAREC / SAC |
| **Chuit et al. 2023** | 652 pacientes con miocardiopatía chagásica en varias instituciones (2018-2021), revisión de historias | Chuit R (Academia Nacional de Medicina) |
| **Hospital de Niños Ricardo Gutiérrez (Servicio de Parasitología y Chagas)** | 234 niños con seguimiento de 6 años o más después del tratamiento, con Holter y eco. Es pediátrico: sirve poco para la población adulta de DECA | — |
| **CEMIC** | **No encontré publicaciones de CEMIC sobre Chagas con ECG y serología.** | — |
| **Bancos de sangre / Plan Nacional de Sangre** | Tienen la serología de millones de donantes, pero **no tienen ECG**. Una cohorte tipo REDS-II argentina (donantes seropositivos y controles con ECG) **no existe**, y sería el diseño más limpio para replicar | — |

**Recomendación práctica.** Para un primer pedido, lo más realista es el **Hospital Eva Perón junto con el INP**: tienen serología confirmada, ECG seriado y fenotipo cardíaco por Kuschnir y eco. Eso permitiría validar externamente sobre "cardiomiopatía confirmada" y no solo sobre "serología", que es justamente la etiqueta que DECA identificó como limitante. Haría falta un convenio institucional, aprobación de comité de ética y saber si el ECG está digitalizado como señal.

---

### 6. IA-ECG en atención primaria en Latinoamérica: TNMG y lecciones

#### 6.1 La infraestructura: Telehealth Network of Minas Gerais (TNMG)

- Oliveira CRA, Paixão GMM, Carvalho VT, …, Marcolino MS, Ribeiro ALP. *BMJ Glob Health* 2025;10(1):e016692. doi:10.1136/bmjgh-2024-016692
  - **~9,88 millones de ECG informados** hasta octubre de 2024, con ~150.000 por mes. Cubre 1.320 municipios de 14 estados y el 72% de ellos tiene menos de 20.000 habitantes. Comenzó en 2005 con 82 municipios.
  - Lecciones:
    - centros regionales capacitados a distancia;
    - operación 24/7;
    - software que bloquea diagnósticos inconsistentes con las mediciones;
    - auditoría de 400 ECG por mes;
    - **el financiamiento continuo es crítico**, porque hubo crisis en 2015-2017 y una interrupción de contrato en 2018;
    - interoperabilidad limitada entre plataformas;
    - la IA requiere validación, monitoreo y supervisión humana.
  - Los datos solo se comparten a pedido, por restricciones de privacidad.
- Alkmim MB, et al. *Bull WHO* 2012;90(5):373-378 (lecciones de campo): equipos de bajo costo, simplicidad y estándares abiertos para que se pueda replicar. https://www.scielosp.org/article/bwho/2012.v90n5/373-378/
- De esta red sale **CODE / CODE-15%**, uno de los datasets de entrenamiento de DECA.

#### 6.2 IA-ECG para Chagas: evidencia

| Estudio | Diseño | Resultado |
|---|---|---|
| Jidling C, Gedon D, Schön TB, …, Ribeiro AH, Ribeiro ALP. *PLoS NTD* 2023;17(7):e0011118. doi:10.1371/journal.pntd.0011118 | Retrospectivo, CODE + SaMi-Trop | AUC de 0,80 en validación, **0,68 en REDS-II y 0,59 en ELSA-Brasil**. Contando como positivo solo la cardiomiopatía: 0,82 en REDS-II y 0,77 en ELSA |
| Reyna MA, et al. Moody Challenge 2025. arXiv:2510.02202 | Competencia, métrica TPR@5% | Mejor equipo: 0,468 en REDS-II, 0,376 en SaMi-Trop 3 y **0,125 en ELSA-Brasil** (media 0,323). La mediana cae 64% en ELSA, la cohorte más parecida a una población de tamizaje. Aun así, los mejores modelos encuentran ~3 veces más positivos que testear al azar |
| **Ribeiro ALP, Cardoso CS, Taconeli CA, …, Sabino EC. *Circulation* 2026;154(10):879-891.** doi:10.1161/CIRCULATIONAHA.125.078949 | **Prospectivo, tamizaje oportunista dentro de TNMG**, 107 municipios endémicos, oct 2023 a sep 2024 | 59.154 adultos con modelo IA-ECG más 3 preguntas. 11,5% positivos. Se invitó a 7.804 personas y **completaron 3.509 (~45%)**. Chagas confirmado en 36,8% de los positivos y 9,3% de los negativos. Prevalencia ponderada de 12,2%. **Sensibilidad 34,3%, especificidad 91,6%**, AUROC 77,6% (preguntas solas 70,0%, ECG solo 64,5%) |
| Fonseca IMG, Nunes MCP, …, Ribeiro ALP. *Arq Bras Cardiol* 2026;123(5):e20260032. doi:10.36660/abc.20260032 | Campaña comunitaria, Feira de Santana (n=1.115) | 10,9% IA-positivos. 13 seropositivos, todos en el grupo IA-positivo. Sensibilidad 100% (IC 69-100), especificidad 40%, **VPP 12%** |
| Jesus FSS, et al. *PLoS NTD* 2026;20(1):e0013892. doi:10.1371/journal.pntd.0013892 | Mismo campo de Feira | 23% de los seropositivos no conocía su infección. La IA-ECG generó 22 falsos positivos. Hay inconsistencias entre el abstract y los resultados |
| Brito BOF, et al. *PLoS NTD* 2021;15(12):e0009974. doi:10.1371/journal.pntd.0009974 | IA-ECG de disfunción ventricular izquierda en SaMi-Trop | AUC de 0,839, sensibilidad 73%, especificidad 83%, **VPN 97%**. Datos públicos en github.com/samitrop/AI-ECG-Chagas |

#### 6.3 Lecciones para DECA

1. **El resultado prospectivo más importante (Ribeiro 2026) se parece mucho al de DECA.** La sensibilidad es baja (~34%) y el VPP en derivados es útil. Las preguntas epidemiológicas solas rinden más que el ECG solo, y la combinación es la mejor. Que DECA no incluya variables epidemiológicas es la brecha más clara frente al estado del arte prospectivo. No es una palanca de modelado sobre la señal: es una variable de entrada que hay que *recolectar*.
2. **La caída fuera de muestra es la norma.** ELSA-Brasil, que es población general, rinde mucho peor que las cohortes enriquecidas. Antes de desplegar en Argentina hay que validar en población argentina, con otros equipos de ECG y otra mezcla de etnias y cepas.
3. **El cuello de botella es la conversión a serología.** Menos de la mitad de los invitados se testeó en TNMG. El diseño del circuito importa tanto como el AUC: extracción en el mismo acto, test rápido en el punto de atención y llamado activo.
4. **La infraestructura precede a la IA.** TNMG funcionó porque ya existía un flujo masivo de ECG digitales centralizados. En Argentina no encontré una red de tele-ECG pública comparable. DECA necesita un flujo de ECG digitales (señal, no papel) para tener dónde correr.
5. **Sostenibilidad y gobernanza.** El financiamiento estable fue la lección explícita de TNMG. En Argentina, el Programa Nacional de Chagas tuvo discontinuidades de conducción en 2023-2024.

---

### 7. Síntesis para la toma de decisiones en DECA

1. **Hay población objetivo.** Unas 1,5 millones de personas infectadas, en su mayoría sin diagnóstico, de las cuales ~30% desarrollará lesión de órgano. Fuera del embarazo prácticamente no hay búsqueda activa de adultos.
2. **La prevalencia es muy heterogénea.** Va de ~1% en gestantes a nivel nacional y <1% en el Sur hasta 20-30% en focos rurales del Chaco. Un mismo VPP no se traslada entre contextos: el 29,5% medido corresponde a una prevalencia de ~1,9%.
3. **El VPP de 29,5% supera con holgura cualquier umbral razonable de derivación**, pero solo como *rule-in*. Con una sensibilidad estimada de ~22%, DECA no puede reemplazar los criterios epidemiológicos ni usarse para negar serología.
4. **Hay un hueco en la literatura.** No existe ningún análisis de costo-efectividad de una estrategia "ECG/IA primero" para Chagas, ni en Argentina ni en otro país. Tampoco existe un cuestionario de riesgo validado en atención primaria argentina.
5. **La fuente argentina más prometedora** para validación externa con fenotipo cardíaco es Eva Perón junto con el INP. Una cohorte tipo REDS-II argentina no existe.
6. **Restricción legal a considerar:** la Ley 26.281 prohíbe la serología de Chagas en exámenes preocupacionales. Usar DECA sobre ECG laborales requiere un análisis legal y ético explícito.

---

### Fuentes que no se pudieron verificar en texto completo

- Tabla por país del WER 2015 de la OMS: solo se usó la cifra de 1.505.235 citada por terceros.
- Positividad en gestantes de 2024-2025 (0,98% y 1,05%): proviene de la gacetilla del BEN SE 38, que tiene una inconsistencia interna.
- Detalles del texto completo de Ribeiro 2026 (*Circulation*), como VPN, costos y barreras: el sitio devolvió 403 y solo se usó el abstract.
- Viotti 2006 (*Ann Intern Med*): las cifras de empeoramiento del ECG vienen de una fuente secundaria.
