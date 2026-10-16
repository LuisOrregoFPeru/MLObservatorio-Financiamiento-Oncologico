# 💊 Observatorio del Financiamiento Oncológico INEN: Impacto Normativo en el Tiempo
### INEN Oncology Financing Observatory: Regulatory Impact Over Time

Dashboard interactivo en **Streamlit**, construido con la misma arquitectura
que el dashboard hermano [`inen-cancer-dashboard`](../inen-cancer-dashboard)
(casos de cáncer por departamento, estilo *Cancer Over Time* de GLOBOCAN),
pero para la línea de investigación sobre el **impacto de la normativa
oncológica peruana 2019-2025 en el gasto que el INEN reporta al SIS y a
FISSAL**.

> **Cita sugerida / Suggested citation**
> Orrego Ferreyros LA, Cervera Salazar LR, Toledo Morote YG, Sarria Bardales G, Payet Meza E (2026).
> *Observatorio del Financiamiento Oncológico INEN: Impacto Normativo en el Tiempo* / *INEN Oncology Financing
> Observatory: Regulatory Impact Over Time*. Lima, Perú/Peru: Instituto Nacional de Enfermedades Neoplásicas.
> Disponible en / Available from: https://impactonormativocancer.streamlit.app

A diferencia del dashboard de casos (datos anuales, por departamento), este
usa **datos mensuales** (2021-2025) desagregados por:

- **Diagnóstico oncológico**: los 7 que FISSAL financia explícitamente
  (cuello uterino, mama, colon, estómago, próstata, leucemias, linfomas) +
  "Otro" (resto de oncología atendida por el INEN) + total agregado.
- **IAFAS financiador**: SIS / FISSAL.
- **Tipo de consumo**: medicamento / insumo (dispositivo médico) /
  procedimiento.

## ✨ Funcionalidades

- **Panel de control** con selección libre de diagnóstico(s), IAFAS,
  tipo(s) de consumo y rango de meses — todos los sub-análisis se recalculan
  sobre la selección activa, no hay combinaciones "hardcodeadas".
- **Gráficos múltiples, no uno solo que cambia**: al seleccionar más de un
  diagnóstico, la pestaña "Serie temporal" genera **un gráfico independiente
  por diagnóstico** (no una única línea agregada), cada uno desagregable por
  IAFAS o por tipo de consumo — reproduce exactamente las hojas
  `Graficos_IAFAS` / `Graficos_TipoConsumo` del workbook Excel de referencia
  de este análisis.
- Tipo de gráfico línea/barras, marcadores, etiquetas de datos, escala
  logarítmica y suavizado LOWESS, todo configurable desde la barra lateral.
- **Análisis de punto de corte normativo**: regresión segmentada de
  **interrupciones múltiples** (ITS) con errores Newey-West sobre los 4
  puntos de corte de `data/norms.csv`, con semáforo de significancia
  (α configurable) y comparación lado a lado entre los 7 diagnósticos.
- **Detección automática de punto de quiebre** (sin fecha impuesta):
  regresión segmentada de dos tramos + test de Chow, busca el mes que mejor
  explica un cambio de tendencia — igual metodología que `detect_breakpoint()`
  del dashboard de casos, adaptada a meses.
- **Quiebre en una fecha específica (evento)**: ancla el test de Chow en
  cualquier fecha que elijas, con ventana de "implementación" opcional a
  excluir justo después.
- **Proyección de gasto a futuro** (1-24 meses) con análisis de sensibilidad:
  3 escenarios (conservador, recomendado con Holt amortiguado, lineal-Sen) +
  banda de incertidumbre al 90% por bootstrap.
- **Ranking por período**: qué diagnóstico/IAFAS/tipo de consumo concentra
  más gasto en un año o en el rango de meses seleccionado, con barras
  horizontales y matices de color por puesto (estilo GLOBOCAN).
- **Tendencia**: test de **Mann-Kendall** (Hamed-Rao, respaldo Yue-Wang) +
  pendiente de Sen sobre la serie mensual, y regresión lineal sobre la serie
  **anual** (excluyendo el último año si está incompleto).
- **Clasificación ABC (Pareto 80/15/5)** de medicamentos, insumos y
  procedimientos, filtrable por IAFAS y diagnóstico (ya no restringido a los
  7 diagnósticos FISSAL), con curva de Pareto interactiva.
- **Medicamentos de alto costo**: se identifican por **código SISMED** contra
  el listado institucional «NO PNUME - Alto costo» (`data/listado_alto_costo.csv`),
  no por un umbral de precio calculado — el umbral DIGEMID de 9 UIT se
  descartó porque esta base no tiene gasto acumulado por paciente-año (ver
  metodología más abajo).
- **Tabla dinámica y descarga en CSV**.
- **Diferencia en Diferencias (DiD)**: nueva pestaña dedicada. Compara el grupo tratamiento (oncológico total o
  un diagnóstico específico) contra el comparador no oncológico, antes/después de cada punto de corte, con un
  modelo `log(gasto) ~ tendencia + tratamiento + post + tratamiento×tendencia + tratamiento×post`. El coeficiente
  de la interacción tratamiento×post es el efecto porcentual estimado, neto de la tendencia general del INEN;
  incluye el chequeo del supuesto de tendencias paralelas (tratamiento×tendencia antes del quiebre) y avisa
  cuando es cuestionable para un punto de corte específico.
- **Comparador no oncológico (control)**: en "Punto de corte normativo", un panel dedicado compara el gasto
  oncológico contra el gasto en diagnósticos NO oncológicos del INEN (comorbilidades: infecciones, enfermedades
  respiratorias/renales/digestivas/endocrinas, trastornos de sangre no neoplásicos) en los mismos puntos de
  corte — un test de falsificación tipo "control": si el gasto no oncológico muestra los mismos quiebres, sugiere
  un efecto presupuestal general de la institución más que un efecto específico de la normativa oncológica.
- **Cronología normativa editable** sin tocar código (`data/norms.csv`).


## 📁 Estructura del proyecto

```
inen-sisfissal-dashboard/
├── app.py                     # Interfaz Streamlit (todo el dashboard)
├── data_processing.py         # Carga y filtrado de datos
├── stats_analysis.py          # ITS multi-quiebre, Mann-Kendall, ABC, alto costo
├── requirements.txt
├── .streamlit/
│   └── config.toml            # Tema de colores de la app
├── data/
│   ├── monthly_diag_iafas_tipo.csv   # Serie mensual (todos los diagnósticos)
│   ├── product_annual_onco.csv       # Detalle anual por producto, con código SISMED (todos los diagnósticos)
│   ├── listado_alto_costo.csv        # Listado institucional "NO PNUME - Alto costo" (cod. SIGA/SISMED)
│   └── norms.csv                     # Cronología normativa y puntos de corte
└── README.md
```

### Sobre los archivos de datos

Los CSV de series en `data/` son extractos **ya agregados** de la base de
consumo línea-de-gasto del INEN (`2021_2025_CONSUMO.xlsx`, ~4,15 millones de
filas), para mantener el repositorio liviano y evitar subir datos
línea-de-gasto completos (potencialmente sensibles) a un repositorio público:

- `monthly_diag_iafas_tipo.csv`: `tiempo, diag_group, iafas, tipo_consumo,
  TOTAL_NETO` — una fila por combinación y mes.
- `product_annual_onco.csv`: `ANNIO, DESC_CONSUMO, cod_sismed, diag_group,
  iafas, tipo_consumo, total_neto, total_cantidad` — una fila por
  producto/ítem, código SISMED, año y combinación. Incluye **todos** los
  diagnósticos (los 7 FISSAL + "Otro" + el comparador no oncológico) — ya no
  está restringido a los 7 FISSAL; es la única fuente con detalle de
  producto, usada por ABC y alto costo.
- `listado_alto_costo.csv`: `cod_siga, cod_sismed, descripcion,
  forma_farmaceutica, programacion, clasificacion_petitorio, tipo,
  precio_referencial, obs_adicional` — el listado institucional INEN "NO
  PNUME - Alto costo" (programación 2026, 42 ítems). Fuente oficial para
  identificar medicamentos de alto costo por código; reemplaza con una
  versión actualizada del listado (mismas columnas) cuando corresponda.

Si actualizas la base fuente, regenera `monthly_diag_iafas_tipo.csv` y
`product_annual_onco.csv` con el mismo agrupamiento (`groupby` sobre las
columnas indicadas, sumando `TOTAL_NETO`/`TOTAL_CANTIDAD`; para
`product_annual_onco.csv`, agrupa también por `CODIGO SISMED` y no filtres
por diagnóstico) y reemplázalos aquí; no hace falta tocar el código.

## 🚀 Ejecutar en local

```bash
cd inen-sisfissal-dashboard
python -m venv venv && source venv/bin/activate   # opcional pero recomendado
pip install -r requirements.txt
streamlit run app.py
```

Se abrirá en `http://localhost:8501`.

## ☁️ Desplegar en Streamlit Community Cloud

Igual que el dashboard hermano: sube esta carpeta completa (con `data/`) a
un repositorio de GitHub, entra a https://share.streamlit.io, "New app",
selecciona el repo/rama y `app.py` como archivo principal.

## 📐 Metodología — punto de corte normativo (ITS multi-quiebre)

`stats_analysis.segmented_its()` ajusta:

```
Y_t = β0 + β1·t + Σ_k (δ_k·D_k,t + γ_k·S_k,t)
```

donde `D_k,t` es 1 desde la fecha efectiva del punto de corte `k` en
adelante, y `S_k,t` es un contador que empieza en 1 ese mismo mes. `δ_k` es
el cambio de nivel (salto) atribuible a ese punto de corte; `γ_k` es el
cambio de pendiente respecto del segmento anterior; la pendiente de cada
segmento se obtiene acumulando los `γ` de los puntos de corte ya vigentes.
Los errores estándar son robustos a heterocedasticidad y autocorrelación
(Newey-West, 3 rezagos), porque el gasto mensual institucional suele estar
autocorrelacionado (ejecución presupuestal por trimestres) — con errores
OLS clásicos se subestimaría el error de los coeficientes.

Los puntos de corte de `data/norms.csv` (v36) se ubican en el **mes de publicación** de cada norma (mes siguiente si se
publicó después del día 20); el rezago de implementación se evalúa como sensibilidad, no como supuesto. Las normas se
agrupan en 6 cortes (`grupo_id`): B1 Ley 31336 (ago-21), B2 Reglamento (abr-22), B3 RM 945 y 964-2022 (dic-22), B3c DS 001-2023 y
RM 191-2023 (feb-mar 2023), B3b Ley 31870 (sep-23), B4 DS 018, RM 820 y RM 885 (dic-24) y B5 Ley 32357 (jun-25). Con el modo
«Agrupar» activo se usa la vigencia más tardía de cada grupo. La app avisa cuando dos cortes quedan a menos de 3 meses,
porque el segmento corto produce coeficientes inestables.

## 🆕 Novedades v37
- Se agrega la **RM 862-2019/MINSA** (publicada 17/09/2019, numeral 8.7 de la Directiva Administrativa 249-MINSA/2018/DIGEMID). Habilita a los establecimientos nivel III-2 a usar medicamentos NO PNUME previa aprobación del CFT. Es anterior a la serie (ene-2021), por eso se muestra como **contexto** en la barra lateral y no como punto de corte.
- El resto del paquete es el de la v36 (fechas corregidas, pestaña de transferencias vs valorizado, comparador con pre-tendencia).

## 🆕 Novedades v36

- **Fechas de publicación corregidas** contra El Peruano: RM 964-2022 = 29/11/2022 (no 26), DS 001-2023-SA = 24/01/2023,
  RM 191-2023 = 22/02/2023, RM 820-2024 = 27/11/2024, RM 885-2024 = 19/12/2024 (el 18 es la firma).
- **Normas agregadas:** RM 945-2022 (lista complementaria oncológica), Ley 31870, DS 018-2024-SA, Ley 32357.
- **DiD:** la base del índice es ahora el inicio de la serie y no el primer quiebre; se muestra el p-valor de pre-tendencia, el
  cambio del propio comparador en ±6 meses y una advertencia cuando el comparador no es válido (cae cerca de 65 % en abr-2022).
- **Pestaña «Transferencias vs valorizado»:** compara Donaciones y Transferencias (MEF) con el valorizado, para descartar que
  un cambio se deba a presupuesto. Datos en `data/ejecucion_presupuestal.csv`.

## 🧪 Mann-Kendall vs. regresión lineal

Igual razón que en el dashboard de casos: la regresión lineal simple asume
residuos independientes, lo que en gasto mensual autocorrelacionado infla
la significancia artificialmente. `mann_kendall_trend()` usa la
modificación de Hamed & Rao (1998), con respaldo automático en Yue-Wang
(2004) si la corrección de varianza es numéricamente inestable, más la
pendiente de Sen (mediana de pendientes entre pares de puntos, robusta a
outliers).

## 💰 Clasificación ABC y alto costo

`abc_classification()` implementa el criterio de Pareto 80/15/5 estándar
en gestión de inventarios hospitalarios: se ordenan los ítems de mayor a
menor gasto acumulado; Zona A = los ítems que juntos explican hasta 80% del
gasto total, Zona B = el tramo 80-95%, Zona C = el tramo 95-100%. Ya no está
restringida a los 7 diagnósticos FISSAL — cubre todos los diagnósticos con
detalle de producto (los 7 FISSAL + "Otro" + el comparador no oncológico),
filtrable desde la interfaz.

`high_cost_by_reference()` reemplazó el umbral DIGEMID (RM 964-2022/MINSA)
de 9 UIT/paciente-año: esa base no tiene trazabilidad por paciente, por lo
que el costo unitario (gasto/cantidad dispensada) nunca superaba el umbral
y el resultado siempre era 0 candidatos. En su lugar, un medicamento se
considera de alto costo si su **código SISMED** figura en el listado
institucional "NO PNUME - Alto costo" (`data/listado_alto_costo.csv`,
programación 2026, 42 ítems) — un criterio administrativo por identidad de
producto. El cruce se hace por `cod_sismed`, no por nombre ni por precio;
de los 42 códigos del listado, los que tienen gasto real registrado se
muestran con su detalle (año, diagnóstico, IAFAS, cantidad, gasto, precio
de adjudicación referencial), y los que no registran gasto en el ámbito
filtrado se listan aparte (el listado es una programación para 2026, no
necesariamente un histórico de compras ya ejecutadas).

## 🛠️ Stack técnico

Python · [Streamlit](https://streamlit.io) · [Plotly](https://plotly.com/python/) ·
Pandas · [statsmodels](https://www.statsmodels.org/) (HAC/Newey-West) · SciPy.
Mann-Kendall (clásico, Hamed-Rao 1998, Yue-Wang 2004) y la pendiente de Sen están
implementados directamente en `stats_analysis.py` con numpy/scipy — sin depender
de `pymannkendall` como librería externa (las fórmulas se verificaron contra esa
librería antes de reemplazarla, ver el docstring al inicio de esa sección).
