# Validación clínica de DECA — estrategia

Cierra la tarea pendiente de la Fase 6 ("definir estrategia de validación clínica/estadística
antes de considerar el modelo apto para uso real"). Es la **estrategia**, no todavía el
protocolo que se presenta a un comité de ética: dice qué se mide, sobre quién, con cuántos,
contra qué se compara y qué resultado cuenta como éxito. Todo eso se fija **antes** de ver un
solo dato argentino, que es la misma regla que cuidó el test de la Fase 5.

Las decisiones que no son técnicas y que requieren firma de Axel están marcadas **[firma]**.
El resto son propuestas argumentadas: se pueden cambiar, pero antes de empezar, no después.

---

## 0. En una página

**La pregunta.** ¿La banda alta del modelo concentra casos de Chagas en población argentina
como lo hizo en el test brasileño (1 de cada 3,4 derivados positivo, contra 1 de cada 52 en la
población general)?

**Por qué hace falta.** Todo lo medido hasta hoy es de Brasil (CODE-15%, SaMi-Trop) y Europa
(PTB-XL). Otra población cambia la mezcla de casos, y otro equipo de ECG cambia la señal. Se
sabe que las dos cosas pueden romper un modelo sin que la salida lo muestre: al modelo se le
dio ruido blanco y devolvió percentil 92 con total aplomo (FASES.md, 2026-09-22).

**El diseño, en tres etapas:**

| etapa | qué | serología | decide |
|---|---|---|---|
| **0** | 1.500–2.000 ECG del archivo de un hospital, con su informe | **no** | ¿la señal local se parece a la de entrenamiento? |
| **1** | estudio de exactitud prospectivo, modelo en modo silencioso | sí: toda la banda alta + 10 % al azar del resto | ¿la banda alta concentra casos acá? |
| 2 | uso real con el resultado visible al médico | la de rutina | ¿cambia a quién se testea? (sólo se esboza) |

**Medida principal: el LR+ de la banda alta** (cociente de verosimilitud positivo). En test es
**21,5 [18,0 – 25,6]**. **Criterio de éxito: límite inferior del IC95 ≥ 5.** **[firma]**

**Tamaño:** depende sobre todo de la prevalencia del sitio. Con 5 %: **~5.000 ECG y ~600
serologías**. Ver sección 5.

**Lo que queda congelado:** el checkpoint (`sha256 b0e2ecfc…`), el umbral de la banda alta
(0,9301) y el preprocesamiento. Lo único que se puede ajustar después de la Etapa 1 son los **textos**
de las bandas (sección 6).

---

## 1. Qué se valida y qué no

**Se valida:** que la banda alta sirve para **priorizar** a quién se le ofrece la serología.
Es lo único del producto que está medido fuera de muestra y funciona (FASES.md, "SÍNTESIS").

**No se valida, porque ya se sabe que no es cierto o no es el uso:**

- **Tamizaje poblacional.** Para encontrar el 95 % de los casos habría que derivar al 62 % de
  la gente. Eso quedó medido en la Fase 5 y un estudio argentino no lo va a cambiar.
- **Descarte.** El producto no tiene un resultado negativo: `no_alta` es el 98,6 % de la
  población y ahí cae el 78 % de los casos. Se mide igual (como resultado secundario), pero
  ningún resultado del estudio habilita a decir "sin Chagas".
- **Diagnóstico o severidad.** El diagnóstico es la serología. Y contra fracción de eyección o
  mortalidad dentro de chagásicos el modelo no discrimina (FASES.md, 2026-09-10, sección 7).

### Por qué la medida principal no es el VPP

El VPP ("de cada 100 derivados, cuántos son positivos") es lo que le importa al médico, pero
**depende de la prevalencia del lugar**, no sólo del modelo. El mismo modelo, con el mismo LR+
de 21,5, da:

| prevalencia del lugar | 2 % | 5 % | 10 % | 20 % |
|---|---|---|---|---|
| VPP de la banda alta | 30,5 % | 53,1 % | 70,5 % | 84,3 % |

Consecuencia directa para el producto: **el texto que muestra hoy el servicio ("de cada 100
personas priorizadas así, cerca de 30 resultaron positivas") sólo es cierto con la prevalencia
de CODE-15% (1,9 %)**. En un hospital de zona endémica con 10 % se queda corto por más de la
mitad. En un centro urbano con 1 % promete de más. El LR+ no tiene ese problema: es una
propiedad del modelo que se traslada entre poblaciones, siempre que la mezcla de casos se
parezca (y eso se chequea, ver sección 4). El VPP local sale del LR+ y de la prevalencia local
con una multiplicación.

**Por qué el piso es 5.** Es el umbral convencional de un LR que "cambia la probabilidad de
forma moderada" (Jaeschke, Guyatt y Sackett, *JAMA* 1994). Con prevalencia de 5 %, un LR+ de 5
lleva el VPP a 21 %: sigue siendo cuatro veces mejor que testear al azar. Por debajo de eso, la
banda alta deja de justificar su lugar en la pantalla.

---

## 2. De dónde se parte: los números de test

Todo sale de la medición única del 2026-09-14 y del artefacto `test_scores_20260916-152548.parquet`,
con los mismos umbrales calibrados en validación. **No es una segunda medición de test**:
mismos scores, mismos umbrales, nada se retoca. Arena A (CODE-15%), a nivel paciente: 34.772
pacientes, 666 positivos, prevalencia 1,92 %.

### Los dos resultados como cocientes de verosimilitud

El servicio devuelve dos resultados: `alta`, que manda a serología, y `no_alta`. Es decisión
de Axel del 2026-09-23 (hasta ese día había también `media` y `baja`, que no derivaban a
nadie; ver FASES.md).

| resultado | % de la población | P(resultado \| positivo) | P(resultado \| negativo) | VPP | **LR** [IC95] |
|---|---|---|---|---|---|
| **alta** (≥ 0,9301) | 1,41 % | 21,8 % (145) | 1,01 % (346) | 29,5 % | **21,5** [18,0 – 25,6] |
| no_alta | 98,59 % | 78,2 % (521) | 99,0 % (33.760) | 1,5 % | **0,79** [0,76 – 0,82] |

Se lee así: **estar en banda alta multiplica las odds de tener Chagas por veinte; no estar
casi no las cambia** (×0,79: de 1,9 % a 1,5 %). Por eso `no_alta` nunca puede comunicarse
como un resultado tranquilizador.

### El comparador: lo que haría un clínico sin el modelo

La pregunta "¿comparado contra qué?" de la SÍNTESIS tiene una respuesta más exigente que "contra
nada". En zona endémica, un médico que ve un **bloqueo de rama derecha** en un ECG ya tiene
motivo para pedir serología: es el primer hallazgo de la tabla de la Guía nacional. Y el modelo,
según el análisis de falsos negativos del 2026-09-16, es operativamente un detector de BRD. Así
que la comparación honesta es contra esa regla:

| regla | deriva | sensibilidad | FPR | VPP | LR+ |
|---|---|---|---|---|---|
| **banda alta del modelo** | 1,41 % | 21,8 % | 1,01 % | **29,5 %** | **21,5** [18,0 – 25,6] |
| "todo BRD informado → serología" | 2,45 % | 18,0 % | 2,15 % | 14,1 % | 8,4 [7,0 – 10,0] |

**El modelo encuentra un poco más de casos derivando a casi la mitad de gente (1,41 % contra
2,45 %), con el doble de VPP.**
No es sólo un detector de BRD: dentro de los que tienen BRD elige a los chagásicos (deriva al
73 % de los BRD chagásicos y al 27 % de los no chagásicos). Y fuera del BRD, cuando se enciende
(pocas veces), acierta tanto como adentro. Además la regla necesita que alguien lea el ECG, y
la falta de cardiólogos en el norte es uno de los tres motivos del ROADMAP.

**Esta comparación entra en la Etapa 1 como resultado secundario obligatorio.** Si en Argentina
el modelo no le gana a "todo BRD → serología", lo que aporta es sólo ahorrar la lectura, y eso
es otro producto.

### El mecanismo, que permite predecir el resultado antes de medirlo

Estratificando por la anotación real de BRD:

| | sensibilidad de la banda alta | FPR de la banda alta |
|---|---|---|
| con BRD | 73,3 % | 27,4 % |
| sin BRD | 10,4 % | 0,43 % |

Si esas cuatro tasas se mantienen en Argentina, el punto de operación local depende sólo de
cuánto BRD hay entre los positivos (r₁) y entre los negativos (r₀):

```
sensibilidad_local ≈ 0,733·r₁ + 0,104·(1 − r₁)
FPR_local          ≈ 0,274·r₀ + 0,0043·(1 − r₀)
```

Con los valores de CODE-15% (r₁ = 18,0 %, r₀ = 2,15 %) la fórmula devuelve 21,8 % y 1,01 %:
reproduce el test exacto. Esto sirve para dos cosas:

1. **Predecir antes de sacar sangre.** r₀ se mide en la Etapa 0 con los informes, sin
   serología. Una población hospitalaria más vieja tiene más BRD no chagásico. Con r₀ = 5 %, el
   FPR sube a 1,8 % y el LR+ esperado baja a ~12. Eso no sería un fallo del modelo sino la
   mezcla de casos, y conviene saberlo de antemano.
2. **Distinguir por qué falla, si falla.** Si en la Etapa 1 el LR+ sale bajo pero las cuatro
   tasas por estrato se sostienen, la causa es la población. Si las tasas por estrato se caen,
   la causa es la señal (equipo, filtros) y la respuesta es otra.

**Por sexo el LR+ es el mismo** (mujeres 22,1 [17,2 – 28,4], varones 21,4 [16,7 – 27,6]). La
banda alta es igual de informativa para una mujer cuando se enciende; se enciende menos, porque
las mujeres tienen menos BRD. Es la versión en LR de lo que ya midió el 2026-09-16.

---

## 3. Etapa 0 — ¿la señal local se parece a la de entrenamiento? (sin serología)

**Para qué.** Es la etapa barata que evita la cara. Si el equipo de ECG del hospital produce una
señal que el modelo lee distinto, eso aparece acá, contra el informe, sin pinchar a nadie.

**Qué se pide al sitio.** 1.500 a 2.000 ECG consecutivos de adultos de su archivo, **exportados
en digital** (no papel ni PDF, ver API.md sección 5), cada uno con **el informe** que ya escribió
un médico. Anonimizados. Sin serología ni contacto con pacientes. Es uso secundario de datos
existentes, pero **igual necesita el aval del comité de ética del sitio**, que decide si hace
falta consentimiento.

**Qué se mide** (todo con el servicio congelado, tal cual lo recibiría el backend):

| # | medida | por qué | criterio **[firma]** |
|---|---|---|---|
| 0.1 | % de archivos que el servicio rechaza (422), por código | factibilidad: si el equipo no exporta algo legible, no hay estudio | ≤ 10 % |
| 0.2 | **AUC de la cabeza de BRD contra el BRD del informe** | test 0,9838, SaMi-Trop 0,9797: si acá cae, la señal es distinta | **IC95 inferior ≥ 0,93** |
| 0.3 | AUC de la cabeza de HBAI contra el HBAI del informe | test 0,96; nunca medida fuera de la población de entrenamiento | descriptivo |
| 0.4 | % en banda alta **entre los ECG sin BRD informado** | alarma de señal: ver abajo | ≤ 4 % |
| 0.5 | r₀ = prevalencia de BRD en los informes | alimenta la predicción de la sección 2 | descriptivo |

**Por qué 0.4 es una alarma que no depende de la prevalencia.** Entre gente sin BRD, la banda
alta se enciende en 0,43 % de los negativos y 10,4 % de los positivos. Aunque el sitio tuviera
20 % de prevalencia (muy alto para un hospital), la fracción esperada sería 2,4 %. Más de 4 %
no se explica con ninguna prevalencia plausible: es la firma de un modelo que puntúa alto algo
que no entiende, como el ruido blanco.

**Cuántos ECG.** Con ~4 % de BRD en una población hospitalaria, 1.500 ECG traen ~60 BRD. Si el
AUC real es 0,98, el IC95 inferior queda en ~0,955: pasa el criterio con margen. Si el AUC real
cayera a 0,93, el inferior sería ~0,885 y el criterio lo detecta.

**Qué cierra de paso.** La medida 0.3 es la primera validación de la cabeza de HBAI fuera de
los datos de entrenamiento: el pendiente nº 5 de la SÍNTESIS. No la cierra entera (eso requiere
HBAI en chagásicos, que llega con la Etapa 1), pero es el primer número.

**Si la Etapa 0 falla**, no se pasa a la Etapa 1. Se diagnostica con los datos que ya hay
(filtros del equipo, frecuencia de muestreo, derivaciones) sin que haya costado una serología.

---

## 4. Etapa 1 — estudio de exactitud diagnóstica

### Población

**Adultos (≥ 18 años) a los que se les hizo un ECG de 12 derivaciones por cualquier motivo
clínico**, en un sitio de zona endémica, reclutados de forma consecutiva. Es el uso previsto en
el ROADMAP: un ECG que ya se hizo por otra razón.

**Exclusiones, y por qué:**

- **Diagnóstico de Chagas ya conocido o tratamiento tripanocida previo.** El producto busca a
  quien no sabe. Además, después del tratamiento la serología puede seguir reactiva años (Guía
  2018, sec. 3.2.2), así que el patrón de referencia no significaría lo mismo.
- **ECG preocupacional.** La Ley 26.281, art. 5, **prohíbe hacer serología de Chagas a
  aspirantes a un empleo**. No es una elección de diseño.
- **Ritmo de marcapasos.** El QRS estimulado no es la conducción del paciente, y quien tiene
  marcapasos ya tiene una cardiopatía estudiada.
- **Un ECG por persona**: el primero elegible. Evita la agregación por máximo que usa la Fase 5
  y deja una fila por paciente.

**No se filtra por riesgo epidemiológico** (haber vivido en zona endémica, madre con Chagas),
pero se registra. Filtrar subiría la prevalencia y bajaría el costo, pero la conclusión
valdría sólo para esa subpoblación. Se analiza como subgrupo.

### Prueba índice

El servicio de inferencia **congelado**: checkpoint `patrones-lr8/mejor.pt` (sha256
`b0e2ecfc838e169c…`), `calibracion.json` del 2026-09-23 (dos resultados) y umbral de la
banda alta 0,9301. Se registra
el `modelo.sha256` de cada respuesta. Un análisis hecho con otro hash queda afuera.

**Modo silencioso:** el resultado no se le muestra al médico ni al paciente y no cambia la
atención. Sólo lo ve el sistema del estudio, que lo usa para decidir a quién se le ofrece la
serología (ver "verificación").

### Patrón de referencia

Serología según la **Guía nacional (2018, sec. 3.2.2)**: dos reacciones normatizadas de
principios distintos sobre la misma muestra, al menos una de alta sensibilidad (ELISA o IFI). Si
coinciden, el resultado es definitivo. Si discrepan, una tercera prueba o derivación a un centro
de referencia. **El laboratorio no conoce la banda.**

### Verificación estratificada

Acá está el ahorro. Los ECG ya están hechos y no cuestan nada; lo caro es cada serología
(consentimiento, extracción, dos técnicas, devolución del resultado). Entonces:

- **A toda la banda alta** se le ofrece serología.
- **A una fracción fija al azar (10 %) de los `no_alta`**, también.

La sensibilidad y la FPR se estiman ponderando cada serología por la inversa de su probabilidad
de haber sido elegida (Begg y Greenes, *Biometrics* 1983). Es válido porque la elección depende
sólo de la banda y la fracción es conocida. Dos reglas que lo sostienen:

1. **Sólo cuentan las serologías del sorteo.** Si alguien no sorteado se hace la serología por
   su cuenta (tiene derecho: es gratuita por ley), ese resultado no entra al análisis principal.
   Si entrara, rompería las ponderaciones.
2. **Los que no aceptan** la extracción se cuentan y se reportan por banda. Se hace un análisis
   de sensibilidad con los dos extremos (todos positivos / todos negativos).

**Si el sitio puede testear a todos**, mejor: el análisis es más simple y no depende de
ponderaciones. Cuesta unas tres veces más serologías para la misma potencia (sección 5).

### Resultados

**Primario:** LR+ de la banda alta. **Éxito si el límite inferior del IC95 ≥ 5.** **[firma]**

**Secundarios, fijados de antemano** (se reportan todos, salgan como salgan):

| medida | para qué |
|---|---|
| LR− de `no_alta` (0,79 en test) | confirmar con números locales que no estar en banda alta casi no baja la probabilidad, y que el texto de `no_alta` dice la verdad |
| **LR+ de "todo BRD informado → serología"**, sobre los mismos pacientes | el comparador; ¿le gana el modelo a la regla clínica? |
| sensibilidad y FPR de la banda alta **dentro de cada estrato de BRD** | ¿se sostiene el mecanismo? (sección 2) |
| sensibilidad y LR+ por sexo | la disparidad ya conocida, con su mecanismo |
| **AUC de la cabeza de HBAI entre seropositivos** | cierra el pendiente nº 5 de la SÍNTESIS |
| AUC del score de Chagas | para comparar con la literatura, no para decidir |
| prevalencia local | convierte los LR en VPP locales (sección 6) |

**No se reporta ninguna métrica que mezcle sitios** si hay más de uno, por la misma razón por
la que el proyecto no mezcla arenas: la prevalencia distinta de cada sitio contamina el
promedio.

### Análisis

- A nivel persona, una fila por participante.
- IC95 del LR por el método logarítmico con varianza por método delta, y bootstrap como
  control. Con ponderaciones, bootstrap estratificado por banda.
- **El plan de análisis se congela con un commit en este repo antes del primer resultado de
  serología**, con fecha, igual que el test de la Fase 5. Lo que se agregue después se reporta
  como exploratorio.
- Se reporta según **STARD 2015** (exactitud diagnóstica) y **TRIPOD+AI** (validación externa
  de un modelo de aprendizaje automático).

---

## 5. Tamaño de muestra

Simulado (4.000 repeticiones por celda, semilla 42) bajo tres escenarios de cómo se comporta el
modelo en Argentina:

| escenario | sensibilidad | FPR | LR+ real |
|---|---|---|---|
| **A** — transporta igual que en test | 21,8 % | 1,01 % | 21,5 |
| **B** — más BRD no chagásico (r₀ = 5 %, población más vieja) | 21,8 % | 1,78 % | 12,2 |
| **C** — la señal se degrada | 15 % | 2 % | 7,5 |

**El estudio se dimensiona para B**, que es la degradación plausible por mezcla de casos. En
el escenario C, con un LR real apenas por encima del piso de 5, lo más probable es un
resultado **no concluyente**. Es lo correcto: un modelo que apenas pasa el piso no debería
declararse validado con un solo estudio.

**Potencia para "IC95 inferior del LR+ ≥ 5", verificación estratificada (alta + 10 % del
resto):**

| prevalencia del sitio | ECG | serologías | potencia A | **potencia B** | potencia C |
|---|---|---|---|---|---|
| 2 % | 8.000 | ~900 | 100 % | **91 %** | 28 % |
| **5 %** | **5.000** | **~590** | 100 % | **97 %** | 37 % |
| 10 % | 3.000 | ~380 | 100 % | **97 %** | 36 % |

**La misma potencia verificando a todos:**

| prevalencia del sitio | ECG = serologías | potencia A | potencia B | potencia C |
|---|---|---|---|---|
| 2 % | 5.000 | 100 % | 95 % | 35 % |
| 5 % | 2.000 | 100 % | 91 % | 29 % |
| 10 % | 1.000 | 98 % | 82 % | 23 % |

**Lectura:** a potencia comparable, la verificación estratificada usa entre 3 y 5 veces menos
serologías, a cambio de 1,6 a 3 veces más ECG, que ya están hechos. **Lo que más mueve el costo es
la prevalencia del sitio**, así que el primer dato a conseguir de un sitio candidato es su
seroprevalencia en adultos, de sus propios registros o de datos provinciales.

**Salvedad.** Los resultados secundarios por subgrupo (sexo, estratos de BRD) van a tener
intervalos anchos con estos tamaños. El estudio está dimensionado para el primario. Los
subgrupos son descriptivos y se dice así.

---

## 6. Qué se puede cambiar después y qué no

| | ¿se puede tocar después de la Etapa 1? |
|---|---|
| checkpoint, pesos | **no**: reentrenar invalida el estudio entero |
| umbral de la banda alta 0,9301 | **no**: es la prueba índice |
| preprocesamiento (`ventana.py`, `lectura_ecg.py`) | **no**, salvo agregar un lector de formato que no cambie la señal. Se verifica igual que el 2026-09-22 |
| **textos de las bandas** (VPP que se muestra) | **sí**, y debería: VPP local = f(LR medido, prevalencia local) |
| población de referencia del percentil | sí: se puede recalcular con ECG locales sin serología (la de la Etapa 0 sirve) |

Si se quisiera recalibrar umbrales con datos argentinos, hace falta **otra** muestra para
confirmar, igual que val/test en el proyecto. Recalibrar y confirmar sobre los mismos datos es
exactamente el error que la Fase 5 evitó.

---

## 7. Etapa 2 — uso real (sólo se esboza)

Si la Etapa 1 pasa, lo siguiente es mostrar el resultado al médico y medir qué cambia. La guía
de referencia es **DECIDE-AI** (evaluación clínica temprana de sistemas de apoyo a la decisión
con IA). Preguntas: ¿a cuántos de la banda alta se les termina haciendo la serología? ¿Cuántos
casos nuevos por cada 1.000 ECG, contra un período previo o sitios sin la herramienta? ¿Algún
médico lee `no_alta` como descarte? Es la pregunta de "¿comparado contra qué?" de la
SÍNTESIS, medida en el mundo. **No se diseña acá**: depende de lo que muestre la Etapa 1 y del
sitio.

---

## 8. Ética y regulación

- **Comité de ética en investigación** del sitio, según la Res. 1480/2011 del Ministerio de
  Salud (Guía para investigaciones con seres humanos). La Etapa 1 necesita **consentimiento
  informado** para la extracción.
- **Todo seropositivo se vincula a atención** según la Guía nacional (evaluación cardiológica y
  evaluación del tratamiento tripanocida). No es opcional: un estudio que encuentra casos y no
  los deriva no se puede aprobar ni se debería. Hay que acordar antes quién recibe esos
  pacientes.
- **Ley 26.281, arts. 5 y 6.** El art. 5 prohíbe la serología de Chagas a aspirantes a un
  empleo, y el art. 6 considera discriminatorio usar esa información en perjuicio de la
  persona. Para el estudio, eso excluye los ECG preocupacionales. **Para el producto es más
  grave**: el ROADMAP menciona como caso de uso el "ECG que ya se realiza mucha gente por
  trabajo". Un resultado de DECA sobre un ECG preocupacional es información sobre infección
  chagásica generada en un contexto laboral. **Ese caso de uso necesita revisión legal antes de
  ofrecerse**, y como mínimo el resultado tendría que ir a la persona y nunca al empleador.
- **Ley 25.326 de protección de datos personales.** Los ECG y resultados se anonimizan. El
  servicio ya no guarda la señal (API.md), pero el estudio sí va a guardar datos, y eso lo
  regula el protocolo.
- **ANMAT, Disposición 64/2025** (software como producto médico). En las Etapas 0 y 1 el
  resultado no llega a la atención, así que no se usa como producto médico. **La Etapa 2 sí**:
  un software cuyo uso previsto es la prevención o el diagnóstico de una patología entra en la
  definición. Hay que consultarlo antes de mostrarle el resultado a un médico real, no después.

---

## 9. Lo que no se resuelve desde este repo

Por orden, y ninguno es código:

1. **Un sitio**: hospital de provincia endémica con ECG digital exportable, laboratorio con
   serología normatizada y una estimación de su seroprevalencia en adultos.
2. **Un investigador clínico responsable** en ese sitio. El comité de ética lo exige, y el
   proyecto necesita a alguien que firme el protocolo clínico.
3. **El formato de exportación del equipo del sitio.** Es la primera pregunta a hacerle: si no
   es CSV, JSON o WFDB, hay que escribir un lector (API.md, sección 5) y verificarlo antes de la
   Etapa 0.
4. **Financiamiento de ~600 a 900 serologías** con su logística (extracción, dos técnicas,
   devolución de resultados y derivación), según la prevalencia del sitio.
5. **La firma de Axel** sobre los criterios marcados **[firma]**: el piso de LR+ ≥ 5 y los tres
   criterios de la Etapa 0.

---

## Referencias

- Ministerio de Salud de la Nación. [*Guía para la atención al paciente infectado con
  Trypanosoma cruzi*](https://www.argentina.gob.ar/sites/default/files/bancos/2020-01/chagas-atencion-paciente-infectado-2018.pdf),
  3ª ed., 2018 (Res. 461/2019). Sec. 3.2.2 (diagnóstico en fase crónica), Tabla 1 (hallazgos de
  ECG) y estadificación de Kuschnir: el 70 % de los infectados está en estadio 0, con ECG normal.
- [Ley 26.281](https://servicios.infoleg.gob.ar/infolegInternet/anexos/130000-134999/131904/norma.htm)
  — prevención y control de la enfermedad de Chagas. Arts. 4, 5 y 6.
- [Res. 1480/2011](https://www.argentina.gob.ar/normativa/nacional/resoluci%C3%B3n-1480-2011-187206/texto)
  — Guía para investigaciones con seres humanos.
- [ANMAT, Disposición 64/2025](https://www.argentina.gob.ar/noticias/anmat-actualiza-los-criterios-regulatorios-para-software-como-dispositivo-medico-samd)
  — software como producto médico.
- Collins GS et al. [TRIPOD+AI statement](https://pubmed.ncbi.nlm.nih.gov/38626948/). *BMJ*
  2024;385:e078378.
- Vasey B et al. [DECIDE-AI](https://www.nature.com/articles/s41591-022-01772-9). *Nat Med*
  2022;28:924–933.
- Bossuyt PM et al. STARD 2015. *BMJ* 2015;351:h5527.
- Begg CB, Greenes RA. Assessment of diagnostic tests when disease verification is subject to
  selection bias. *Biometrics* 1983;39:207–215.
- Jaeschke R, Guyatt GH, Sackett DL. Users' guides to the medical literature III. How to use an
  article about a diagnostic test. B. *JAMA* 1994;271:703–707.
