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
  procedimientos, filtrable por IAFAS y diagnóstico, con curva de Pareto
  interactiva.
- **Medicamentos de alto costo** según el umbral DIGEMID de 9 UIT/paciente-
  año (con la limitación metodológica declarada explícitamente en la app).
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
│   ├── product_annual_onco.csv       # Detalle anual por producto (7 dx FISSAL)
│   └── norms.csv                     # Cronología normativa y puntos de corte
└── README.md
```

### Sobre los archivos de datos

Los CSV en `data/` son extractos **ya agregados** de la base de consumo
línea-de-gasto del INEN (`2021_2025_CONSUMO.xlsx`, ~4,15 millones de filas),
para mantener el repositorio liviano y evitar subir datos línea-de-gasto
completos (potencialmente sensibles) a un repositorio público:

- `monthly_diag_iafas_tipo.csv`: `tiempo, diag_group, iafas, tipo_consumo,
  TOTAL_NETO` — una fila por combinación y mes.
- `product_annual_onco.csv`: `ANNIO, DESC_CONSUMO, diag_group, iafas,
  tipo_consumo, total_neto, total_cantidad` — una fila por producto/ítem,
  año y combinación, restringido a los 7 diagnósticos FISSAL (es la única
  fuente con detalle de producto, usada por ABC y alto costo).

Si actualizas la base fuente, regenera estos dos CSV con el mismo agrupamiento
(`groupby` sobre las columnas indicadas, sumando `TOTAL_NETO`/`TOTAL_CANTIDAD`)
y reemplázalos aquí; no hace falta tocar el código.

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

Los puntos de corte en `data/norms.csv` evalúan **cada norma de forma independiente** (una fila = un
punto de corte), sin agrupar. Dos pares quedan a menos de 3 meses entre sí (RM 964-2022 → DS 001-2023-SA,
y RM 820-2024 → RM 885-2024): la app lo detecta automáticamente y muestra una advertencia en la pestaña
"Punto de corte normativo", porque un segmento mensual tan corto produce coeficientes estadísticamente
inestables (matriz de diseño casi singular) — interpreta esos casos con cautela. RM 862-2019/MINSA (2019)
y la RD 003-2025-CETS/INS (vigencia estimada oct-2025) quedan fuera de la ventana de datos fiable
(ene-2021 a ago-2025) y no están en `norms.csv`; si amplías la serie fuente, agrégalas ahí.

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
en gestión de inventarios hospitalarios: Zona A = ítems que acumulan hasta
80% del gasto, Zona B = 80-95%, Zona C = 95-100%. `high_cost_medications()`
aplica el umbral DIGEMID (RM 964-2022/MINSA) de 9 UIT, con la limitación
declarada en la propia app: al no haber trazabilidad por paciente en esta
base, el costo unitario se aproxima como gasto/cantidad dispensada, una
cota inferior del costo real por paciente-año.

## 🛠️ Stack técnico

Python · [Streamlit](https://streamlit.io) · [Plotly](https://plotly.com/python/) ·
Pandas · [statsmodels](https://www.statsmodels.org/) (HAC/Newey-West) · SciPy.
Mann-Kendall (clásico, Hamed-Rao 1998, Yue-Wang 2004) y la pendiente de Sen están
implementados directamente en `stats_analysis.py` con numpy/scipy — sin depender
de `pymannkendall` como librería externa (las fórmulas se verificaron contra esa
librería antes de reemplazarla, ver el docstring al inicio de esa sección).
