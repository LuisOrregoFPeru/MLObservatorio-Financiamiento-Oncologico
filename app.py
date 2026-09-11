"""
Dashboard "Gasto Oncológico INEN — SIS/FISSAL"
================================================
Construido sobre la misma arquitectura que el dashboard hermano
"Cáncer en el Tiempo — Perú (INEN)" (casos por departamento, estilo
GLOBOCAN), con paridad de análisis estadístico (tendencia lineal,
detección automática de quiebre, quiebre por evento, Mann-Kendall,
proyección con escenarios, ranking) más lo propio de esta línea de
investigación: ITS de interrupciones múltiples sobre los 4 puntos de
corte normativos, ABC y alto costo.

A diferencia del dashboard de casos, aquí la serie es MENSUAL
(2021-2025) y hay tres ejes de desagregación en vez de uno: diagnóstico
oncológico, IAFAS financiador (SIS/FISSAL) y tipo de consumo
(medicamento/insumo/procedimiento). La pestaña "Serie temporal" puede
generar UN GRÁFICO POR DIAGNÓSTICO seleccionado (en vez de una sola
serie agregada), igual que las hojas Graficos_IAFAS/Graficos_TipoConsumo
del workbook de referencia de este análisis.

Ejecutar localmente:
    streamlit run app.py
"""

from __future__ import annotations

import colorsys
import string

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_processing import (
    ALL_DIAGS_LABEL,
    IAFAS_LIST,
    MONTHLY_FILE,
    ONCO_DIAGS,
    OTRO_LABEL,
    NO_ONCO_LABEL,
    TIPO_CONSUMO_LIST,
    UIT_BY_YEAR,
    breaks_dict,
    breakpoint_view,
    diag_options,
    filter_monthly,
    load_monthly,
    load_norms,
    load_product_annual,
    merge_monthly_data,
    monthly_series,
    process_raw_upload,
)
from stats_analysis import (
    abc_classification,
    cagr,
    chow_test_event,
    detect_breakpoint_auto,
    diff_in_diff,
    high_cost_medications,
    linear_trend,
    mann_kendall_trend,
    project_series,
    ranking_table,
    segmented_its,
    significance_label,
    smooth_series,
)

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------

from pathlib import Path
from PIL import Image

_ASSETS_DIR = Path(__file__).parent / "assets"
_favicon_path = _ASSETS_DIR / "ofo_logo.png"
_page_icon = Image.open(_favicon_path) if _favicon_path.exists() else "🔭"

st.set_page_config(
    page_title="INEN Oncology Financing Observatory | Observatorio del Financiamiento Oncológico INEN",
    page_icon=_page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY_COLOR = "#2C5F8A"
SECONDARY_COLOR = "#D9542F"
ACCENT_COLOR = "#1F4E78"
OK_COLOR = "#2F9E44"
WARN_COLOR = "#E8A33D"
BAD_COLOR = "#C94141"
DEFAULT_PALETTE = [PRIMARY_COLOR, SECONDARY_COLOR, OK_COLOR, "#8850C4", "#C9A13B", "#5C6AC4", "#7D7D7D", "#4AA8C9"]

CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700;800;900&display=swap');
    .main .block-container {{ padding-top: 1.3rem; max-width: 1400px; }}
    h1, h2, h3 {{ color: {ACCENT_COLOR}; }}
    .app-header {{
        background: linear-gradient(90deg, {ACCENT_COLOR} 0%, {PRIMARY_COLOR} 100%);
        padding: 1.1rem 1.6rem; border-radius: 10px; color: white; margin-bottom: 1.2rem;
        display: flex; align-items: center; gap: 1rem;
    }}
    .app-header .ofo-logo {{ flex-shrink: 0; width: 64px; height: 64px; }}
    .app-header .ofo-logo * {{ fill: white; stroke: white; }}
    .app-header .ofo-logo .ofo-ring {{ fill: none; }}
    .app-header h1 {{
        color: white; margin: 0; font-size: 1.7rem; font-family: 'Poppins', sans-serif;
        font-weight: 800; text-transform: uppercase; letter-spacing: 0.01em; line-height: 1.15;
    }}
    .app-header p {{ color: #e8eef5; margin: 0.2rem 0 0 0; font-size: 0.95rem; }}
    div[data-testid="stMetric"] {{ background: #f7f8fb; border: 1px solid #e2e6ee; border-radius: 10px; padding: 0.6rem 0.9rem; }}
    .source-note {{ font-size: 0.8rem; color: #666; margin-top: 0.4rem; }}
    .interp-box {{ background: #f4f7fb; border-left: 4px solid {PRIMARY_COLOR}; border-radius: 6px;
                   padding: 0.7rem 1rem; margin: 0.6rem 0 1rem 0; font-size: 0.92rem; color: #333; }}
    .mini-title {{ font-size: 0.95rem; font-weight: 600; color: {ACCENT_COLOR}; margin-bottom: 0.2rem; }}
    .citation-box {{ background: #fafbfc; border: 1px solid #e2e6ee; border-radius: 8px;
                      padding: 1rem 1.2rem; margin-top: 1.2rem; font-size: 0.85rem; color: #333; }}
    .citation-box h4 {{ margin: 0 0 0.5rem 0; font-size: 0.8rem; letter-spacing: 0.04em; color: {ACCENT_COLOR}; }}
    .citation-box p {{ margin: 0 0 0.6rem 0; line-height: 1.5; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

_ofo_logo_svg = (_ASSETS_DIR / "ofo_logo.svg").read_text() if (_ASSETS_DIR / "ofo_logo.svg").exists() else ""
_ofo_logo_svg = _ofo_logo_svg.replace('<svg ', '<svg class="ofo-logo" ', 1)


def fmt_df(df: pd.DataFrame, money_cols=(), int_cols=(), pct_cols=(), dec_cols=()):
    """Devuelve un pandas Styler con separadores de miles: dinero sin
    decimales ('S/ 1,234,567'), enteros con separador de miles,
    porcentajes con un decimal, y decimales genéricos con 2 decimales.
    Columnas que no existen en `df` se ignoran silenciosamente (evita
    KeyError si una tabla no tiene alguna de las columnas típicas)."""
    fmts = {}
    for c in money_cols:
        if c in df.columns:
            fmts[c] = "S/ {:,.0f}"
    for c in int_cols:
        if c in df.columns:
            fmts[c] = "{:,.0f}"
    for c in pct_cols:
        if c in df.columns:
            fmts[c] = "{:,.1f}%"
    for c in dec_cols:
        if c in df.columns:
            fmts[c] = "{:,.2f}"
    return df.style.format(fmts)


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"



def shades_of(hex_color: str, n: int, l_min: float = 0.30, l_max: float = 0.80) -> list[str]:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))
    h, _l, s = colorsys.rgb_to_hls(r, g, b)
    lights = [l_min] if n <= 1 else np.linspace(l_min, l_max, n)
    out = []
    for lig in lights:
        rr, gg, bb = colorsys.hls_to_rgb(h, float(lig), s)
        out.append("#%02x%02x%02x" % (int(round(rr * 255)), int(round(gg * 255)), int(round(bb * 255))))
    return out


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

monthly_df = load_monthly()
product_df = load_product_annual()
norms_df = load_norms()

MIN_MONTH, MAX_MONTH = monthly_df["tiempo"].min(), monthly_df["tiempo"].max()

st.markdown(
    f"""<div class="app-header">{_ofo_logo_svg}<div>
<h1>Observatorio del Financiamiento Oncológico INEN: Impacto Normativo en el Tiempo</h1>
<p style="font-style: italic;">INEN Oncology Financing Observatory: Regulatory Impact Over Time</p>
<p>Gasto oncológico del INEN reportado al SIS/FISSAL · serie mensual {MIN_MONTH:%b-%Y} a {MAX_MONTH:%b-%Y}</p>
</div></div>""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Panel de control")

    # ------------------------------------------------------------------
    # Base de datos adicional (serie temporal externa)
    # ------------------------------------------------------------------
    with st.expander("📊 Base de datos adicional (serie temporal)", expanded=False):
        st.caption(
            "Sube cualquier serie temporal adicional (CSV o Excel) con al menos una columna de "
            "fecha y una columna numérica. Se analizará de forma exploratoria en la pestaña "
            "«Exploración (base adicional)» y se comparará con el gasto oncológico en «Análisis cruzado»."
        )
        _sec_upload = st.file_uploader(
            "Archivo de serie adicional", type=["xlsx", "xls", "csv"], key="upload_sec"
        )
        if _sec_upload is not None:
            try:
                _is_csv = _sec_upload.name.lower().endswith(".csv")
                if _is_csv:
                    _sec_raw = pd.read_csv(_sec_upload)
                else:
                    # Leer hojas disponibles antes de cargar datos
                    _xls = pd.ExcelFile(_sec_upload)
                    _sheet_names = _xls.sheet_names
                    if len(_sheet_names) > 1:
                        _sec_sheet = st.selectbox(
                            "Hoja a utilizar", _sheet_names, key="sec_sheet",
                            help="El archivo tiene varias hojas — elige cuál contiene la serie temporal.",
                        )
                    else:
                        _sec_sheet = _sheet_names[0]
                        st.caption(f"Hoja: **{_sec_sheet}**")
                    _sec_raw = _xls.parse(_sec_sheet)
                _sec_cols = _sec_raw.columns.tolist()
                st.caption(f"Columnas detectadas: {', '.join(_sec_cols)}")
                _sec_date_col = st.selectbox("Columna de fecha", _sec_cols, key="sec_date_col")
                _num_cols = [c for c in _sec_cols if c != _sec_date_col]
                _sec_val_cols = st.multiselect(
                    "Columna(s) de valor (numérico)",
                    _num_cols,
                    default=[_num_cols[0]] if _num_cols else [],
                    key="sec_val_cols",
                    help="Selecciona una o más columnas. Si eliges varias, se combinarán (suma o promedio) en una sola serie.",
                )
                _sec_label = st.text_input("Etiqueta para esta serie", "Serie adicional", key="sec_label")
                _sec_units = st.text_input(
                    "Unidades del eje Y", "Valor", key="sec_units",
                    help="Ej: 'Soles (S/)', 'Pacientes', 'Casos', 'USD'",
                )
                # Combinar solo si el usuario lo pide (y solo cuando hay varias columnas)
                _multi_cols = len(_sec_val_cols) > 1
                if _multi_cols:
                    _sec_combine = st.checkbox(
                        "Combinar columnas en una sola serie",
                        value=False,
                        key="sec_combine",
                        help="Si está marcado, las columnas se suman o promedian fila a fila antes de agregar al período. Sin marcar, cada columna conserva su propia serie.",
                    )
                else:
                    _sec_combine = True  # con una sola columna no hay nada que combinar
                if _sec_combine:
                    _sec_agg = st.radio(
                        "Agregar por", ["Suma", "Promedio"], horizontal=True, key="sec_agg",
                        help="Cómo combinar las columnas fila a fila antes de agrupar al período.",
                    )
                else:
                    _sec_agg = "Suma"  # valor por defecto (no se usará para la serie combinada)
                if not _sec_val_cols:
                    st.warning("Selecciona al menos una columna de valor.")
                elif st.button("Cargar serie adicional", key="btn_load_sec"):
                    try:
                        _df = _sec_raw[[_sec_date_col] + _sec_val_cols].copy()
                        _df = _df.rename(columns={_sec_date_col: "tiempo"})
                        # Combinar columnas numéricas en una sola (suma o promedio fila a fila)
                        if not _sec_combine:
                            # Sin combinar: la serie principal = primera columna seleccionada
                            _df["valor"] = pd.to_numeric(_df[_sec_val_cols[0]], errors="coerce").fillna(0.0)
                        elif len(_sec_val_cols) == 1:
                            _df["valor"] = pd.to_numeric(_df[_sec_val_cols[0]], errors="coerce").fillna(0.0)
                        else:
                            _num_df = _df[_sec_val_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
                            _df["valor"] = _num_df.sum(axis=1) if _sec_agg == "Suma" else _num_df.mean(axis=1)
                        _df = _df[["tiempo", "valor"]]
                        # Detectar si la columna contiene solo años (enteros 1900-2100)
                        _as_num = pd.to_numeric(_df["tiempo"], errors="coerce")
                        if _as_num.notna().all() and _as_num.between(1900, 2100).all():
                            # Año entero → 1 de enero de ese año
                            _df["tiempo"] = pd.to_datetime(
                                _as_num.astype(int).astype(str), format="%Y"
                            )
                        else:
                            _df["tiempo"] = pd.to_datetime(
                                _df["tiempo"], dayfirst=True, errors="coerce"
                            )
                        _df = _df.dropna(subset=["tiempo"])
                        _df["valor"] = pd.to_numeric(_df["valor"], errors="coerce").fillna(0.0)
                        _agg_fn = "sum" if _sec_agg == "Suma" else "mean"
                        # Frecuencia: anual si todos los valores eran años enteros,
                        # mensual en cualquier otro caso.
                        _sec_freq = "YS" if (_as_num.notna().all() and _as_num.between(1900, 2100).all()) else "MS"
                        _s = _df.groupby(pd.Grouper(key="tiempo", freq=_sec_freq))["valor"].agg(_agg_fn)
                        _s = _s[_s.index.notna()]
                        # Descartar fechas epoch (parseo fallido → 1970-01-01)
                        _s = _s[_s.index >= pd.Timestamp("1990-01-01")]
                        if _s.empty:
                            st.error(
                                "No se encontraron fechas válidas (>1990) en la columna seleccionada. "
                                "Verifica que la columna de fecha tenga un formato reconocible "
                                "(ej. 2021-01-01, 01/01/2021, enero 2021, o solo 2021)."
                            )
                        else:
                            full_idx_s = pd.date_range(_s.index.min(), _s.index.max(), freq=_sec_freq)
                            _s = _s.reindex(full_idx_s, fill_value=0.0)
                            # Construir también serie individual por columna (para grafico multi-línea)
                            _df_time = _df[["tiempo"]].copy()
                            _sec_cols_dict: dict[str, pd.Series] = {}
                            _raw_num = _sec_raw[[_sec_date_col] + _sec_val_cols].copy()
                            _raw_num = _raw_num.rename(columns={_sec_date_col: "tiempo"})
                            _raw_num["tiempo"] = _df["tiempo"].values  # ya parseado
                            for _col in _sec_val_cols:
                                _cv = pd.to_numeric(_raw_num[_col], errors="coerce").fillna(0.0)
                                _raw_num["_v"] = _cv
                                _sc = _raw_num.groupby(pd.Grouper(key="tiempo", freq=_sec_freq))["_v"].agg(_agg_fn)
                                _sc = _sc[_sc.index.notna() & (_sc.index >= pd.Timestamp("1990-01-01"))]
                                _sc = _sc.reindex(full_idx_s, fill_value=0.0)
                                _sec_cols_dict[_col] = _sc
                            st.session_state["secondary_serie"] = _s
                            st.session_state["secondary_label"] = _sec_label
                            st.session_state["secondary_units"] = _sec_units
                            st.session_state["secondary_freq"] = _sec_freq
                            st.session_state["secondary_cols_dict"] = _sec_cols_dict
                            st.session_state["secondary_val_cols"] = list(_sec_val_cols)
                            if _sec_freq == "YS":
                                st.success(
                                    f"✓ Serie anual cargada: {len(_s)} año(s) "
                                    f"({_s.index.min():%Y} a {_s.index.max():%Y})"
                                )
                            else:
                                st.success(
                                    f"✓ Serie mensual cargada: {len(_s)} meses "
                                    f"({_s.index.min():%b-%Y} a {_s.index.max():%b-%Y})"
                                )
                    except Exception as _e:
                        st.error(f"Error al procesar el archivo: {_e}")
            except Exception as _e:
                st.error(f"No se pudo leer el archivo: {_e}")

        if "secondary_serie" in st.session_state:
            _lbl = st.session_state.get("secondary_label", "Serie adicional")
            _ss = st.session_state["secondary_serie"]
            st.info(
                f"**'{_lbl}'** cargada — {len(_ss)} meses "
                f"({_ss.index.min():%b-%Y} a {_ss.index.max():%b-%Y})"
            )
            if st.button("🗑️ Quitar serie adicional", key="btn_rm_sec"):
                for _k in ["secondary_serie", "secondary_label", "secondary_units"]:
                    st.session_state.pop(_k, None)
                st.rerun()

    with st.expander("📤 Incorporar datos recientes"):
        st.caption(
            "Sube un archivo .xlsx o .csv con el mismo formato crudo que la base original "
            "(2021_2025_CONSUMO.xlsx): columnas ANNIO, MES, ESFISSAL (S/N), TIPO_CONSUMO, COD_DIAG y "
            "TOTAL_NETO. Se clasifica igual que el histórico y se combina con él; si un mes/diagnóstico/"
            "IAFAS/tipo de consumo ya existía, el valor nuevo lo reemplaza."
        )
        _upload = st.file_uploader("Archivo con datos nuevos", type=["xlsx", "xls", "csv"], key="upload_new_data")
        if _upload is not None:
            try:
                _raw_df = pd.read_csv(_upload) if _upload.name.lower().endswith(".csv") else pd.read_excel(_upload)
                _new_df = process_raw_upload(_raw_df)
                _existing_df = load_monthly()
                _merged_df = merge_monthly_data(_existing_df, _new_df)
                _merged_df.to_csv(MONTHLY_FILE, index=False)
                load_monthly.clear()
                n_dropped = _new_df.attrs.get("n_dropped", 0)
                st.success(
                    f"Datos incorporados: {len(_new_df)} combinación(es) mes/diagnóstico/IAFAS/tipo procesadas "
                    f"desde el archivo."
                    + (f" Se descartaron {n_dropped} fila(s) con IAFAS o diagnóstico no reconocible." if n_dropped else "")
                )
                if st.button("Recargar dashboard con los datos actualizados"):
                    st.rerun()
            except ValueError as e:
                st.error(f"No se pudo procesar el archivo: {e}")
            except Exception as e:
                st.error(f"Error inesperado al procesar el archivo: {e}")
        st.caption(
            "⚠️ En Streamlit Community Cloud, este archivo se guarda en el sistema de archivos de la app y "
            "persiste mientras la app siga corriendo, pero se pierde si la app se reinicia o se vuelve a "
            "desplegar desde GitHub — para que el cambio sea permanente, actualiza también "
            "data/monthly_diag_iafas_tipo.csv en el repositorio."
        )

    st.divider()
    st.markdown("**IAFAS financiador**")
    if "iafas_checks" not in st.session_state:
        st.session_state["iafas_checks"] = {opt: True for opt in IAFAS_LIST}
    iafas_sel = []
    for opt in IAFAS_LIST:
        checked = st.checkbox(opt, value=st.session_state["iafas_checks"].get(opt, True), key=f"iafaschk_{opt}")
        st.session_state["iafas_checks"][opt] = checked
        if checked:
            iafas_sel.append(opt)

    st.divider()
    st.markdown("**Tipo de consumo**")
    if "tipo_checks" not in st.session_state:
        st.session_state["tipo_checks"] = {opt: True for opt in TIPO_CONSUMO_LIST}
    tipo_sel = []
    for opt in TIPO_CONSUMO_LIST:
        checked = st.checkbox(opt.capitalize(), value=st.session_state["tipo_checks"].get(opt, True), key=f"tipochk_{opt}")
        st.session_state["tipo_checks"][opt] = checked
        if checked:
            tipo_sel.append(opt)

    st.divider()
    st.markdown("**Diagnóstico oncológico**")
    st.caption("Marca uno o más — cada uno genera su propio gráfico debajo, en la pestaña Serie temporal.")
    _diag_all_opts = diag_options()
    if "diag_checks" not in st.session_state:
        st.session_state["diag_checks"] = {opt: (opt == ALL_DIAGS_LABEL) for opt in _diag_all_opts}
    diag_sel = []
    for opt in _diag_all_opts:
        checked = st.checkbox(opt, value=st.session_state["diag_checks"].get(opt, False), key=f"diagchk_{opt}")
        st.session_state["diag_checks"][opt] = checked
        if checked:
            diag_sel.append(opt)

    st.divider()
    date_range = st.slider(
        "Rango de meses", min_value=MIN_MONTH.to_pydatetime(), max_value=MAX_MONTH.to_pydatetime(),
        value=(MIN_MONTH.to_pydatetime(), MAX_MONTH.to_pydatetime()), format="MMM YYYY",
    )
    alpha = st.select_slider("Umbral de significancia (α)", options=[0.01, 0.05, 0.10], value=0.05)

    st.divider()
    st.markdown("**Visualización**")
    chart_type = st.radio("Tipo de gráfico", ["Línea", "Barras"], horizontal=True)
    show_markers = st.checkbox("Mostrar marcadores", value=True)
    show_labels = st.checkbox("Mostrar etiquetas de datos", value=False)
    show_lowess = st.checkbox("Mostrar línea suavizada (LOWESS)", value=False)
    log_scale = st.checkbox("Escala logarítmica", value=False)

    st.divider()
    st.markdown("**Puntos de corte normativos**")
    group_mode = st.radio(
        "Modo", ["Evaluar cada norma por separado", "Agrupar normas cercanas (paquetes)"],
        help="Independiente: 7 puntos de corte, uno por norma. Agrupado: las normas cuya vigencia efectiva "
             "cae a menos de 3 meses de otra se combinan en un solo punto de corte (más estable "
             "estadísticamente, pero no distingue el efecto de cada norma dentro del paquete).",
    )
    norms_view = breakpoint_view(norms_df, grouped=group_mode.startswith("Agrupar"))

    st.caption("Marca cuáles mostrar como líneas verticales en los gráficos.")
    _norm_checks_key = "norm_checks_grouped" if group_mode.startswith("Agrupar") else "norm_checks_individual"
    if _norm_checks_key not in st.session_state:
        st.session_state[_norm_checks_key] = {nid: True for nid in norms_view["id"]}
    norm_sel = []
    for _, _row in norms_view.iterrows():
        _nid = _row["id"]
        checked = st.checkbox(_row["label"], value=st.session_state[_norm_checks_key].get(_nid, True), key=f"normchk_{_nid}_{_norm_checks_key}")
        st.session_state[_norm_checks_key][_nid] = checked
        if checked:
            norm_sel.append(_nid)

    with st.expander("📜 Detalle de cada punto de corte (editable en data/norms.csv)"):
        st.dataframe(
            norms_view[["label", "norma", "fecha_publicacion", "fecha_efectiva", "nota"]],
            hide_index=True, use_container_width=True,
        )

BREAKS = breaks_dict(norms_view)
BREAK_LABELS = dict(zip(norms_view["id"], norms_view["label"]))

if not diag_sel:
    diag_sel = [ALL_DIAGS_LABEL]
if not iafas_sel:
    iafas_sel = list(IAFAS_LIST)
if not tipo_sel:
    tipo_sel = list(TIPO_CONSUMO_LIST)

date_lo, date_hi = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
filtered = filter_monthly(monthly_df, diag_sel, iafas_sel, tipo_sel, date_lo, date_hi)
serie = monthly_series(filtered)
active_breaks = {k: v for k, v in BREAKS.items() if k in norm_sel and date_lo < v <= date_hi}
_sorted_break_ids = [k for k, _ in sorted(active_breaks.items(), key=lambda x: x[1])]
BREAK_CODES = {k: letter for k, letter in zip(_sorted_break_ids, string.ascii_uppercase)}


def selection_caption() -> str:
    d = ", ".join(diag_sel) if len(diag_sel) <= 3 else f"{len(diag_sel)} diagnósticos"
    return f"**Selección actual:** {d} · {'/'.join(iafas_sel)} · {len(tipo_sel)} tipo(s) de consumo · {date_lo:%b-%Y} a {date_hi:%b-%Y}"


st.caption(selection_caption())

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------

k1, k2, k3, k4, k5 = st.columns(5)
total_periodo = serie.sum()
k1.metric("Gasto total del período", f"S/ {total_periodo:,.0f}")

if len(serie) >= 2:
    var_pct = (serie.iloc[-1] / serie.iloc[0] - 1) * 100 if serie.iloc[0] else np.nan
    k2.metric(f"Variación {serie.index[0]:%b-%y} → {serie.index[-1]:%b-%y}", f"{var_pct:+.1f}%" if not np.isnan(var_pct) else "n/d")
    n_years = len(serie) / 12
    c = cagr(serie.iloc[0], serie.iloc[-1], n_years) if n_years > 0 else None
    k3.metric("CAGR (crec. anual compuesto)", f"{c*100:+.1f}%/año" if c is not None else "n/d")
else:
    k2.metric("Variación", "n/d")
    k3.metric("CAGR", "n/d")

peak_month = serie.idxmax() if len(serie) else None
k4.metric("Mes pico", f"{peak_month:%b-%Y}" if peak_month is not None else "n/d", f"S/ {serie.max():,.0f}" if len(serie) else "")

mk_quick = mann_kendall_trend(serie.values) if len(serie) >= 8 else None
k5.metric("Tendencia (Mann-Kendall)", mk_quick.trend if mk_quick else "n/d",
          f"p={mk_quick.p_value:.3f}" if mk_quick else "")

its_quick = segmented_its(serie, BREAKS, BREAK_LABELS, alpha=alpha) if len(serie) >= 12 else None

st.divider()

tab_series, tab_break, tab_proj, tab_rank, tab_trend, tab_did, tab_abc, tab_table, tab_sec_exp, tab_cross = st.tabs([
    "📈 Serie temporal", "📐 Punto de corte normativo", "🔮 Proyección", "🏆 Ranking",
    "🧪 Tendencia (Mann-Kendall)", "🆚 Diferencia en Diferencias", "💰 ABC / Alto costo", "📋 Tabla y descarga",
    "🔍 Exploración (base adicional)", "🔀 Análisis cruzado",
])


def add_breaks(fig: go.Figure) -> None:
    for k, bdate in active_breaks.items():
        fig.add_vline(x=bdate, line_dash="dot", line_color="gray", opacity=0.7)
        fig.add_annotation(x=bdate, y=1.04, yref="paper", showarrow=False,
                            text=f"<b>{BREAK_CODES.get(k, k)}</b>", font=dict(size=12, color="#555555"))


def add_series(fig: go.Figure, x, y, name: str, color: str) -> None:
    mode = "lines+markers" if show_markers else "lines"
    if chart_type == "Barras":
        fig.add_trace(go.Bar(x=x, y=y, name=name, marker_color=color,
                              text=[f"{v:,.0f}" for v in y] if show_labels else None, textposition="outside"))
    else:
        fig.add_trace(go.Scatter(x=x, y=y, mode=mode, name=name, line=dict(width=2, color=color),
                                  marker=dict(size=5),
                                  text=[f"{v:,.0f}" for v in y] if show_labels else None,
                                  textposition="top center"))


def style_fig(fig: go.Figure, height: int = 380, title: str | None = None) -> None:
    layout_kwargs = dict(
        height=height, hovermode="x unified", yaxis_title="Soles (S/)",
        yaxis_type="log" if log_scale else "linear", legend=dict(orientation="h", y=-0.25),
        margin=dict(t=45 if title else 30), barmode="group",
    )
    if title:
        layout_kwargs["title"] = title
    fig.update_layout(**layout_kwargs)


# ============================== TAB 1: Serie temporal ========================
with tab_series:
    st.subheader("Gasto mensual")
    st.caption("Cada casilla marcada en el panel lateral (Diagnóstico, IAFAS o Tipo de consumo) genera su "
               "propio gráfico debajo, agregando sobre las demás casillas marcadas en los otros paneles.")

    def build_chart(sub_filtered: pd.DataFrame, title: str | None, height: int = 360) -> go.Figure:
        s_total = monthly_series(sub_filtered)
        fig = go.Figure()
        add_series(fig, s_total.index, s_total.values, "Gasto observado", PRIMARY_COLOR)
        res = segmented_its(s_total, BREAKS, BREAK_LABELS, alpha=alpha) if len(s_total) >= 12 else None
        if res is not None:
            fig.add_trace(go.Scatter(x=res.fitted.index, y=res.fitted.values, mode="lines",
                                      name="Ajuste segmentado (ITS)", line=dict(width=2, color=SECONDARY_COLOR, dash="dash")))
        if show_lowess and len(s_total) >= 6:
            sm_y = smooth_series(s_total.values)
            if sm_y is not None:
                fig.add_trace(go.Scatter(x=s_total.index, y=sm_y, mode="lines", name="LOWESS",
                                          line=dict(width=2, color=OK_COLOR)))
        add_breaks(fig)
        style_fig(fig, height=height, title=title)
        return fig

    def render_chart_group(subheader: str, items: list[tuple[str, list[str], list[str], list[str]]]) -> None:
        """items: lista de (título, diag_filter, iafas_filter, tipo_filter)."""
        if not items:
            return
        st.markdown(f'<div class="mini-title">{subheader}</div>', unsafe_allow_html=True)
        n_charts = len(items)
        ncols = 1 if n_charts == 1 else 2
        chart_height = 480 if n_charts == 1 else 360
        cols = st.columns(ncols)
        for i, (title, d_f, i_f, t_f) in enumerate(items):
            sub = filter_monthly(monthly_df, d_f, i_f, t_f, date_lo, date_hi)
            with cols[i % ncols]:
                st.plotly_chart(build_chart(sub, title, height=chart_height), use_container_width=True, key=f"chart_{subheader}_{title}")

    if not diag_sel and not iafas_sel and not tipo_sel:
        st.info("Marca al menos una casilla en el panel lateral para generar un gráfico.")
    else:
        main_col, legend_col = st.columns([5, 1])
        with main_col:
            render_chart_group("Por diagnóstico", [(d, [d], iafas_sel, tipo_sel) for d in diag_sel])
            render_chart_group("Por IAFAS", [(i, diag_sel, [i], tipo_sel) for i in iafas_sel])
            render_chart_group("Por tipo de consumo", [(t.capitalize(), diag_sel, iafas_sel, [t]) for t in tipo_sel])
        with legend_col:
            if active_breaks:
                st.markdown('<div class="mini-title">Puntos de corte</div>', unsafe_allow_html=True)
                for k in _sorted_break_ids:
                    st.markdown(
                        f'<div style="font-size:0.8rem; margin-bottom:0.5rem;"><b>{BREAK_CODES[k]}</b> — {BREAK_LABELS.get(k, k)}</div>',
                        unsafe_allow_html=True,
                    )

    st.markdown(
        """<div class="interp-box">La línea punteada naranja no es una tendencia libre: es el ajuste de un
        modelo que sigue el gasto mes a mes pero al que se le permite cambiar de pendiente justo en cada punto
        de corte marcado en el panel lateral — así se puede ver si el ritmo de crecimiento se acelera o
        desacelera alrededor de esa norma, en vez de forzar una sola línea recta para todo el período.</div>""",
        unsafe_allow_html=True,
    )

# ============================== TAB 2: Punto de corte ========================
with tab_break:
    st.subheader("Puntos de corte normativos (ITS de interrupciones múltiples)")
    st.caption("Es un solo modelo ajustado a TODA la serie de una vez (no una regresión distinta por tramo): "
               "una tendencia de fondo que corre por los 56 meses, más un ajuste de nivel y de pendiente en "
               "cada punto de corte. Por eso el resultado de una norma puede cambiar según qué otras normas "
               "estén marcadas — todos los tramos se estiman juntos, no de forma aislada.")
    st.markdown(
        """<div class="interp-box"><b>Salto de nivel</b> es un cambio brusco justo en el mes del punto de
        corte — el gasto pega un salto (o una caída) de un mes al siguiente. <b>Cambio de pendiente</b> es
        distinto: es que el RITMO de crecimiento mensual cambia de ahí en adelante, sin que haya necesariamente
        un salto puntual. Pueden darse por separado — una norma puede producir un salto de nivel significativo
        sin cambiar la pendiente, o cambiar la pendiente sin ningún salto inmediato, o ambas cosas a la vez.</div>""",
        unsafe_allow_html=True,
    )
    if len(active_breaks) >= 2:
        sorted_breaks = sorted(active_breaks.items(), key=lambda x: x[1])
        close_pairs = [
            (BREAK_LABELS.get(a, a), BREAK_LABELS.get(b, b))
            for (a, da), (b, db) in zip(sorted_breaks, sorted_breaks[1:])
            if (db - da).days < 90
        ]
        if close_pairs:
            pairs_txt = "; ".join(f"{a} → {b}" for a, b in close_pairs)
            st.warning(
                f"⚠️ Estos puntos de corte quedan a menos de 3 meses entre sí: **{pairs_txt}**. Con series "
                f"mensuales, un segmento tan corto tiene muy pocos datos propios y su pendiente puede ser "
                f"estadísticamente inestable — interpreta esos coeficientes con cautela, no como evidencia firme "
                f"de esa norma en particular."
            )
    if its_quick is None or not its_quick.segments:
        st.info("No hay suficientes observaciones o puntos de corte dentro del rango seleccionado.")
    else:
        st.caption(f"n = {its_quick.n_obs} meses · R² = {its_quick.r2:.3f} · pendiente pre-primera norma = "
                   f"{its_quick.pre_slope:.4f} S/mill/mes (p = {its_quick.pre_slope_p:.4f})")
        rows = []
        for seg in its_quick.segments:
            rows.append({
                "Punto de corte": seg.label, "Fecha efectiva": seg.date.strftime("%b-%Y"),
                "N meses posteriores": seg.n_post,
                "Pendiente antes (S/mill/mes)": round(seg.slope_before, 4),
                "Cambio de nivel (S/mill)": round(seg.level_change, 4), "p (nivel)": round(seg.level_p, 4),
                "Significancia (nivel)": significance_label(seg.level_p, alpha),
                "Cambio de pendiente (S/mill/mes)": round(seg.slope_change, 4), "p (pendiente)": round(seg.slope_p, 4),
                "Significancia (pendiente)": significance_label(seg.slope_p, alpha),
                "Pendiente después (S/mill/mes)": round(seg.slope_after, 4),
            })
        tab_res = pd.DataFrame(rows)

        def _color_sig(val):
            if val == "Significativo":
                return f"background-color: {OK_COLOR}33"
            if val == "Marginal":
                return f"background-color: {WARN_COLOR}33"
            return f"background-color: {BAD_COLOR}33"

        st.dataframe(
            tab_res.style.map(_color_sig, subset=["Significancia (nivel)", "Significancia (pendiente)"]),
            hide_index=True, use_container_width=True,
        )
        for seg in its_quick.segments:
            direction = "aceleración" if seg.slope_change > 0 else "desaceleración"
            st.markdown(
                f"- **{seg.label}**: pendiente pasa de {seg.slope_before:.3f} a {seg.slope_after:.3f} S/mill/mes "
                f"({direction}); cambio de pendiente **{significance_label(seg.slope_p, alpha).lower()}** "
                f"(p={seg.slope_p:.3f}); salto de nivel **{significance_label(seg.level_p, alpha).lower()}** (p={seg.level_p:.3f})."
            )

        st.divider()
        st.markdown("##### Comparar el mismo punto de corte entre varios diagnósticos")
        compare_diags = st.multiselect("Diagnósticos a comparar", options=ONCO_DIAGS, default=ONCO_DIAGS, key="cmp_diags")
        if compare_diags:
            comp_rows = []
            for d in compare_diags:
                s = monthly_series(filter_monthly(monthly_df, [d], iafas_sel, tipo_sel, date_lo, date_hi))
                res = segmented_its(s, BREAKS, BREAK_LABELS, alpha=alpha) if len(s) >= 12 else None
                if not res or not res.segments:
                    continue
                for seg in res.segments:
                    comp_rows.append({"Diagnóstico": d, "Punto de corte": seg.label,
                                       "Cambio de pendiente (S/mill/mes)": round(seg.slope_change, 4)})
            if comp_rows:
                pivot = pd.DataFrame(comp_rows).pivot(index="Diagnóstico", columns="Punto de corte", values="Cambio de pendiente (S/mill/mes)")
                st.dataframe(pivot, use_container_width=True)

    st.divider()
    with st.expander("🔍 Detección automática del punto de quiebre (sin fecha impuesta)"):
        st.caption("Busca el mes que mejor divide la serie seleccionada en dos tramos (mínima suma de cuadrados "
                   "residual) y aplica un test de Chow contra el modelo único. Igual metodología que "
                   "`detect_breakpoint()` en el dashboard de casos, adaptada a meses.")
        auto = detect_breakpoint_auto(serie) if len(serie) >= 12 else None
        if auto is None:
            st.info("Selección con muy pocas observaciones.")
        else:
            ac1, ac2, ac3 = st.columns(3)
            ac1.metric("Mes de quiebre detectado", f"{auto.date:%b-%Y}")
            ac2.metric("p-valor (test de Chow)", f"{auto.p_value:.4f}", significance_label(auto.p_value, alpha))
            ac3.metric("Pendiente antes → después", f"{auto.slope_before:,.0f} → {auto.slope_after:,.0f} S//mes")
            figb = go.Figure()
            figb.add_trace(go.Scatter(x=serie.index, y=serie.values, mode="markers", name="Observado", marker=dict(color=PRIMARY_COLOR, size=5)))
            figb.add_trace(go.Scatter(x=auto.x_before, y=auto.y_pred_before, mode="lines", name="Ajuste antes", line=dict(color=SECONDARY_COLOR, width=2)))
            figb.add_trace(go.Scatter(x=auto.x_after, y=auto.y_pred_after, mode="lines", name="Ajuste después", line=dict(color=OK_COLOR, width=2)))
            figb.add_vline(x=auto.date, line_dash="dot", line_color="gray")
            style_fig(figb, height=340)
            st.plotly_chart(figb, use_container_width=True)

    with st.expander("🏛️ Quiebre en una fecha específica (evento elegido por el usuario)"):
        st.caption("A diferencia de la detección automática, ancla el test de Chow en una fecha que tú definas "
                   "(una ley, la apertura de un centro, etc.), con una ventana de implementación opcional a excluir justo después.")
        ec1, ec2 = st.columns(2)
        event_date = ec1.date_input("Fecha del evento", value=pd.Timestamp("2021-08-01"),
                                     min_value=MIN_MONTH.to_pydatetime(), max_value=MAX_MONTH.to_pydatetime())
        lag = ec2.slider("Meses de implementación a excluir tras el evento", 0, 12, 3)
        ev = chow_test_event(serie, pd.Timestamp(event_date), implementation_lag_months=lag) if len(serie) >= 8 else None
        if ev is None:
            st.info("No hay suficientes observaciones antes/después de esa fecha (con la ventana excluida) para el test.")
        else:
            ev1, ev2, ev3 = st.columns(3)
            ev1.metric("N antes / después", f"{ev.n_before} / {ev.n_after}")
            ev2.metric("p-valor (test de Chow)", f"{ev.p_value:.4f}", significance_label(ev.p_value, alpha))
            ev3.metric("Pendiente antes → después", f"{ev.slope_before:,.0f} → {ev.slope_after:,.0f} S//mes")
            fige = go.Figure()
            fige.add_trace(go.Scatter(x=serie.index, y=serie.values, mode="markers", name="Observado", marker=dict(color=PRIMARY_COLOR, size=5)))
            fige.add_trace(go.Scatter(x=ev.x_before, y=ev.y_pred_before, mode="lines", name="Ajuste antes", line=dict(color=SECONDARY_COLOR, width=2)))
            fige.add_trace(go.Scatter(x=ev.x_after, y=ev.y_pred_after, mode="lines", name="Ajuste después", line=dict(color=OK_COLOR, width=2)))
            fige.add_vline(x=pd.Timestamp(event_date), line_dash="dot", line_color="gray")
            style_fig(fige, height=340)
            st.plotly_chart(fige, use_container_width=True)

    st.divider()
    with st.expander("🧪 Comparador no oncológico (control) — ¿el quiebre es específico del cáncer?", expanded=False):
        st.caption("Si el gasto NO oncológico del INEN (comorbilidades ajenas al cáncer: infecciones, "
                   "enfermedades respiratorias, renales, digestivas, endocrinas, etc. — ver detalle en "
                   "data_processing.py) muestra los MISMOS quiebres que el gasto oncológico, es señal de un "
                   "efecto presupuestal general de la institución, no específico de la normativa oncológica. "
                   "Si NO los muestra (o van en otra dirección), refuerza que el efecto observado en oncología "
                   "es atribuible a esas normas en particular.")
        no_onco_filt = filter_monthly(monthly_df, [NO_ONCO_LABEL], iafas_sel, tipo_sel, date_lo, date_hi)
        no_onco_serie = monthly_series(no_onco_filt)
        onco_filt = filter_monthly(monthly_df, [ALL_DIAGS_LABEL], iafas_sel, tipo_sel, date_lo, date_hi)
        onco_filt = onco_filt[onco_filt["diag_group"] != NO_ONCO_LABEL]
        onco_serie = monthly_series(onco_filt)
        res_onco = segmented_its(onco_serie, BREAKS, BREAK_LABELS, alpha=alpha) if len(onco_serie) >= 12 else None
        res_no_onco = segmented_its(no_onco_serie, BREAKS, BREAK_LABELS, alpha=alpha) if len(no_onco_serie) >= 12 else None
        if res_onco is None or res_no_onco is None:
            st.info("Selección con muy pocas observaciones para comparar ambos grupos.")
        else:
            figc = go.Figure()
            figc.add_trace(go.Scatter(x=onco_serie.index, y=onco_serie.values / onco_serie.values[0] * 100, mode="lines",
                                       name="Oncológico (índice, base=100 en el primer mes)", line=dict(color=PRIMARY_COLOR, width=2)))
            figc.add_trace(go.Scatter(x=no_onco_serie.index, y=no_onco_serie.values / no_onco_serie.values[0] * 100, mode="lines",
                                       name="No oncológico (índice, base=100 en el primer mes)", line=dict(color=SECONDARY_COLOR, width=2)))
            add_breaks(figc)
            figc.update_layout(height=380, hovermode="x unified", yaxis_title="Índice (primer mes = 100)",
                                legend=dict(orientation="h", y=-0.25), margin=dict(t=30))
            st.plotly_chart(figc, use_container_width=True)

            comp_rows2 = []
            for seg_o, seg_n in zip(res_onco.segments, res_no_onco.segments):
                comp_rows2.append({
                    "Punto de corte": seg_o.label,
                    "Cambio pendiente ONCOLÓGICO (S/mill/mes)": round(seg_o.slope_change, 4),
                    "Signif. oncológico": significance_label(seg_o.slope_p, alpha),
                    "Cambio pendiente NO ONCOLÓGICO (S/mill/mes)": round(seg_n.slope_change, 4),
                    "Signif. no oncológico": significance_label(seg_n.slope_p, alpha),
                    "¿Coinciden en dirección?": "Sí" if np.sign(seg_o.slope_change) == np.sign(seg_n.slope_change) else "No",
                })
            comp_df2 = pd.DataFrame(comp_rows2)
            st.dataframe(comp_df2, hide_index=True, use_container_width=True)
            st.caption("«¿Coinciden en dirección?» compara solo el signo del cambio de pendiente (acelera/desacelera "
                       "en ambos grupos a la vez), no si ambos son estadísticamente significativos — cruza con las "
                       "columnas de significancia antes de sacar conclusiones sobre un punto de corte específico.")

# ============================== TAB 3: Proyección =============================
with tab_proj:
    st.subheader("Proyección del gasto a futuro (análisis de sensibilidad)")
    st.caption("3 escenarios: conservador (promedio últimos 3 meses, sin tendencia), recomendado (Holt con "
               "tendencia amortiguada + banda de incertidumbre por bootstrap) y lineal (pendiente de Sen sin amortiguar).")
    horizon = st.slider("Horizonte de proyección (meses)", 1, 24, 6)
    proj = project_series(serie, horizon=horizon) if len(serie) >= 12 else None
    if proj is None:
        st.info("Selección con muy pocas observaciones para proyectar (mínimo 12 meses).")
    else:
        figp = go.Figure()
        figp.add_trace(go.Scatter(x=proj.idx_hist, y=proj.values_hist, mode="lines+markers", name="Histórico", line=dict(color=PRIMARY_COLOR, width=2)))
        figp.add_trace(go.Scatter(x=proj.idx_future, y=proj.recommended, mode="lines+markers", name="Recomendado (Holt amortiguado)", line=dict(color=SECONDARY_COLOR, width=2)))
        figp.add_trace(go.Scatter(x=proj.idx_future, y=proj.conservative, mode="lines", name="Conservador", line=dict(color=WARN_COLOR, width=2, dash="dot")))
        figp.add_trace(go.Scatter(x=proj.idx_future, y=proj.optimistic, mode="lines", name="Lineal (Sen)", line=dict(color=OK_COLOR, width=2, dash="dash")))
        figp.add_trace(go.Scatter(x=list(proj.idx_future) + list(proj.idx_future[::-1]),
                                   y=list(proj.recommended_upper) + list(proj.recommended_lower[::-1]),
                                   fill="toself", fillcolor=hex_to_rgba(SECONDARY_COLOR, 0.13), line=dict(color="rgba(0,0,0,0)"),
                                   name="Banda 90% (recomendado)", showlegend=True))
        style_fig(figp, height=420)
        st.plotly_chart(figp, use_container_width=True)
        st.markdown(
            f"""<div class="interp-box">{proj.method_note} La banda de incertidumbre se ensancha mes a mes —
            cuanto más lejos se proyecta, menos confiable es el número puntual; a partir de {'~6-9 meses' if horizon > 9 else 'unos meses'}
            la proyección refleja sobre todo el patrón histórico, no información nueva. Ninguno de los tres
            escenarios anticipa normas futuras ni cambios de política: son extrapolaciones estadísticas de lo ya
            observado, no un pronóstico presupuestal oficial.</div>""",
            unsafe_allow_html=True,
        )
        proj_table = pd.DataFrame({
            "Mes": proj.idx_future.strftime("%b-%Y"), "Conservador": proj.conservative.round(0),
            "Recomendado": proj.recommended.round(0), "Rec. banda inf. (90%)": proj.recommended_lower.round(0),
            "Rec. banda sup. (90%)": proj.recommended_upper.round(0), "Lineal (Sen)": proj.optimistic.round(0),
        })
        st.dataframe(
            fmt_df(proj_table, money_cols=["Conservador", "Recomendado", "Rec. banda inf. (90%)", "Rec. banda sup. (90%)", "Lineal (Sen)"]),
            hide_index=True, use_container_width=True,
        )

# ============================== TAB 4: Ranking =================================
with tab_rank:
    st.subheader("Ranking por período")
    rc1, rc2, rc3 = st.columns(3)
    rank_group = rc1.selectbox("Agrupar por", ["Diagnóstico", "IAFAS", "Tipo de consumo"])
    rank_years = sorted(monthly_df["tiempo"].dt.year.unique())
    rank_year = rc2.selectbox("Año", rank_years, index=len(rank_years) - 2 if len(rank_years) > 1 else 0)
    rank_scope = rc3.radio("Alcance", ["Ese año", "Todo el período seleccionado"], horizontal=False)
    col_map = {"Diagnóstico": "diag_group", "IAFAS": "iafas", "Tipo de consumo": "tipo_consumo"}
    if rank_scope == "Ese año":
        p_lo, p_hi = pd.Timestamp(f"{rank_year}-01-01"), pd.Timestamp(f"{rank_year}-12-01")
    else:
        p_lo, p_hi = date_lo, date_hi
    rank_df = ranking_table(filtered, col_map[rank_group], p_lo, p_hi)
    if rank_group == "Diagnóstico" and not rank_df.empty:
        rank_df[col_map[rank_group]] = rank_df[col_map[rank_group]].replace({"Otro": OTRO_LABEL})
    if rank_df.empty:
        st.info("Sin datos para este período/selección.")
    else:
        colors = shades_of(PRIMARY_COLOR, len(rank_df))
        figr = go.Figure(go.Bar(
            x=rank_df["total_neto"], y=rank_df[col_map[rank_group]], orientation="h",
            marker_color=colors, text=[f"#{r} · S/ {v:,.0f}" for r, v in zip(rank_df["ranking"], rank_df["total_neto"])],
            textposition="outside",
        ))
        figr.update_layout(height=max(320, 40 * len(rank_df)), xaxis_title="Gasto (S/)", yaxis=dict(autorange="reversed"), margin=dict(t=20))
        st.plotly_chart(figr, use_container_width=True)
        rank_display = rank_df.rename(columns={col_map[rank_group]: rank_group, "total_neto": "Gasto (S/)", "ranking": "Puesto"})
        st.dataframe(
            fmt_df(rank_display, money_cols=["Gasto (S/)"], int_cols=["Puesto"]),
            hide_index=True, use_container_width=True,
        )

# ============================== TAB 5: Tendencia ================================
with tab_trend:
    st.subheader("¿Hay o no tendencia? Mann-Kendall + pendiente de Sen")
    st.caption("La regresión lineal simple asume residuos independientes; el gasto mensual suele estar "
               "autocorrelacionado. Mann-Kendall (Hamed-Rao, respaldo Yue-Wang) corrige la varianza por esa "
               "autocorrelación y no asume forma lineal.")
    mk_res = mann_kendall_trend(serie.values) if len(serie) >= 8 else None
    if mk_res is None:
        st.info("Selección con muy pocas observaciones.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Tendencia (Mann-Kendall)", mk_res.trend, f"variante: {mk_res.variant_used}")
        c2.metric("p-valor", f"{mk_res.p_value:.4f}", significance_label(mk_res.p_value, alpha))
        c3.metric("Pendiente de Sen", f"S/ {mk_res.sen_slope:,.0f} /mes")

    st.divider()
    st.subheader("Evolución mensual y tendencia")
    eval_mode = st.radio(
        "Evaluar tendencia en", ["Toda la serie", "Segmentada por un punto de corte"], horizontal=True,
    )

    def _midx(idx: pd.DatetimeIndex) -> np.ndarray:
        return np.arange(len(idx), dtype=float)

    if eval_mode == "Toda la serie":
        if len(serie) < 3:
            st.info("Selecciona un rango con al menos 3 meses.")
        else:
            tr = linear_trend(_midx(serie.index), serie.values / 1e6)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=serie.index, y=serie.values, mode="lines+markers", name="Gasto observado (S/)", line=dict(color=PRIMARY_COLOR, width=2)))
            if tr:
                fig2.add_trace(go.Scatter(x=serie.index, y=tr.y_pred * 1e6, mode="lines", name="Tendencia lineal",
                                           line=dict(color=SECONDARY_COLOR, width=2, dash="dash")))
            add_breaks(fig2)
            style_fig(fig2, height=420)
            st.plotly_chart(fig2, use_container_width=True)
            if tr:
                st.markdown(
                    f"""<div class="interp-box">Tendencia lineal mensual: {tr.slope:+.4f} S/ millones/mes
                    (equivalente a {tr.slope*12:+.2f} S/ millones/año), p={tr.p_value:.4f}, R²={tr.r2:.3f} —
                    <b>{significance_label(tr.p_value, alpha)}</b>. Compárese con la pendiente de Sen (no
                    paramétrica) de arriba: {mk_res.sen_slope*12/1e6 if mk_res else float('nan'):+.2f} S/
                    millones/año equivalentes.</div>""",
                    unsafe_allow_html=True,
                )
    else:
        if not active_breaks:
            st.info("No hay puntos de corte marcados en el panel lateral dentro del rango seleccionado. "
                    "Marca al menos uno en «Puntos de corte normativos» para segmentar.")
        else:
            seg_options = [k for k in _sorted_break_ids if k in active_breaks]
            seg_pick = st.selectbox("Segmentar en", seg_options, format_func=lambda k: BREAK_LABELS.get(k, k))
            split_date = active_breaks[seg_pick]
            pre = serie[serie.index < split_date]
            post = serie[serie.index >= split_date]

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=serie.index, y=serie.values, mode="lines+markers", name="Gasto observado (S/)", line=dict(color=PRIMARY_COLOR, width=2)))
            tr_pre = linear_trend(_midx(pre.index), pre.values / 1e6) if len(pre) >= 3 else None
            tr_post = linear_trend(_midx(post.index), post.values / 1e6) if len(post) >= 3 else None
            if tr_pre:
                fig2.add_trace(go.Scatter(x=pre.index, y=tr_pre.y_pred * 1e6, mode="lines", name="Tendencia antes",
                                           line=dict(color=SECONDARY_COLOR, width=2, dash="dash")))
            if tr_post:
                fig2.add_trace(go.Scatter(x=post.index, y=tr_post.y_pred * 1e6, mode="lines", name="Tendencia después",
                                           line=dict(color=OK_COLOR, width=2, dash="dash")))
            fig2.add_vline(x=split_date, line_dash="dot", line_color="gray")
            style_fig(fig2, height=420)
            st.plotly_chart(fig2, use_container_width=True)

            mk_pre = mann_kendall_trend(pre.values) if len(pre) >= 8 else None
            mk_post = mann_kendall_trend(post.values) if len(post) >= 8 else None
            colpre, colpost = st.columns(2)
            with colpre:
                st.markdown(f'<div class="mini-title">Antes de {BREAK_LABELS.get(seg_pick, seg_pick)} (n={len(pre)})</div>', unsafe_allow_html=True)
                if mk_pre:
                    st.metric("Tendencia (Mann-Kendall)", mk_pre.trend, f"p={mk_pre.p_value:.4f}")
                    st.metric("Pendiente de Sen", f"S/ {mk_pre.sen_slope:,.0f} /mes")
                else:
                    st.info("Muy pocas observaciones antes del punto de corte (mínimo 8 meses).")
            with colpost:
                st.markdown(f'<div class="mini-title">Después de {BREAK_LABELS.get(seg_pick, seg_pick)} (n={len(post)})</div>', unsafe_allow_html=True)
                if mk_post:
                    st.metric("Tendencia (Mann-Kendall)", mk_post.trend, f"p={mk_post.p_value:.4f}")
                    st.metric("Pendiente de Sen", f"S/ {mk_post.sen_slope:,.0f} /mes")
                else:
                    st.info("Muy pocas observaciones después del punto de corte (mínimo 8 meses).")
            st.caption("Esta segmentación por Mann-Kendall/Sen es independiente del modelo de regresión "
                       "segmentada (ITS) de la pestaña «Punto de corte normativo» — aquí se evalúa cada tramo "
                       "como una serie separada, sin asumir una forma funcional única para toda la serie.")

# ============================== TAB: Diferencia en Diferencias ==================
with tab_did:
    st.subheader("Diferencia en Diferencias (DiD) — impacto neto vs. comparador no oncológico")
    st.caption("Compara el gasto de cada diagnóstico marcado en el panel lateral contra el grupo control "
               "(gasto NO oncológico) antes y después de cada punto de corte marcado. Cada diagnóstico genera su "
               "propio gráfico debajo, igual que en Serie temporal. Modelo en log(gasto): "
               "log(y) = β0 + β1·t + β2·tratamiento + β3·post + β4·(tratamiento×t) + β5·(tratamiento×post); "
               "β5 es el efecto DiD, β4 es el chequeo de tendencias paralelas antes del quiebre.")

    treat_groups = [d for d in diag_sel if d != NO_ONCO_LABEL]
    if not treat_groups:
        st.info("Marca al menos un diagnóstico (distinto de «No oncológico (comparador)») en el panel lateral.")
    elif not active_breaks:
        st.info("No hay puntos de corte marcados en el panel lateral dentro del rango seleccionado.")
    else:
        control_serie = monthly_series(filter_monthly(monthly_df, [NO_ONCO_LABEL], iafas_sel, tipo_sel, date_lo, date_hi))

        def build_treat_serie(label: str) -> pd.Series:
            if label == ALL_DIAGS_LABEL:
                f = filter_monthly(monthly_df, [ALL_DIAGS_LABEL], iafas_sel, tipo_sel, date_lo, date_hi)
                f = f[f["diag_group"] != NO_ONCO_LABEL]
            else:
                f = filter_monthly(monthly_df, [label], iafas_sel, tipo_sel, date_lo, date_hi)
            return monthly_series(f)

        def build_did_chart(label: str) -> go.Figure:
            treat_serie = build_treat_serie(label)
            idx_common = treat_serie.index.intersection(control_serie.index)
            t_al, c_al = treat_serie.reindex(idx_common), control_serie.reindex(idx_common)
            first_break = min(active_breaks.values())
            pre_mask = idx_common < first_break
            t_base = t_al[pre_mask].mean() if pre_mask.any() else t_al.iloc[0]
            c_base = c_al[pre_mask].mean() if pre_mask.any() else c_al.iloc[0]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=idx_common, y=t_al / t_base * 100, mode="lines", name="Tratamiento", line=dict(color=PRIMARY_COLOR, width=2)))
            fig.add_trace(go.Scatter(x=idx_common, y=c_al / c_base * 100, mode="lines", name="Control (no oncológico)", line=dict(color=SECONDARY_COLOR, width=2)))
            add_breaks(fig)
            fig.update_layout(height=340, hovermode="x unified", yaxis_title="Índice (base=100 antes del 1er quiebre)",
                               legend=dict(orientation="h", y=-0.25), margin=dict(t=45), title=label)
            return fig

        main_col, legend_col = st.columns([5, 1])
        with main_col:
            n_charts = len(treat_groups)
            ncols = 1 if n_charts == 1 else 2
            cols = st.columns(ncols)
            for i, label in enumerate(treat_groups):
                with cols[i % ncols]:
                    st.plotly_chart(build_did_chart(label), use_container_width=True, key=f"did_chart_{label}")
        with legend_col:
            st.markdown('<div class="mini-title">Puntos de corte</div>', unsafe_allow_html=True)
            for k in _sorted_break_ids:
                if k not in active_breaks:
                    continue
                st.markdown(
                    f'<div style="font-size:0.8rem; margin-bottom:0.5rem;"><b>{BREAK_CODES[k]}</b> — {BREAK_LABELS.get(k, k)}</div>',
                    unsafe_allow_html=True,
                )

        st.divider()
        st.markdown("##### Tabla de resultados — todos los grupos × todos los puntos de corte")
        did_rows = []
        for label in treat_groups:
            treat_serie = build_treat_serie(label)
            for k in _sorted_break_ids:
                if k not in active_breaks:
                    continue
                res = diff_in_diff(treat_serie, control_serie, active_breaks[k], alpha=alpha)
                if res is None:
                    continue
                did_rows.append({
                    "Grupo tratamiento": label,
                    "Punto de corte": BREAK_LABELS.get(k, k),
                    "Efecto % aprox.": round(res.pct_effect, 1),
                    "p-valor DiD": round(res.did_p, 4),
                    "Significancia DiD": significance_label(res.did_p, alpha),
                    "Tendencias paralelas (pre)": "OK" if res.parallel_trends_ok else "Cuestionable",
                    "N antes / después": f"{res.n_pre} / {res.n_post}",
                })
        if not did_rows:
            st.info("Ninguna combinación de diagnóstico × punto de corte tiene suficientes observaciones antes y después para el DiD.")
        else:
            did_summary = pd.DataFrame(did_rows)

            def _color_did_sig(val):
                if val in ("Significativo", "OK"):
                    return f"background-color: {OK_COLOR}33"
                if val == "Marginal":
                    return f"background-color: {WARN_COLOR}33"
                if val in ("No significativo", "Cuestionable"):
                    return f"background-color: {BAD_COLOR}33"
                return ""

            st.dataframe(
                did_summary.style.map(_color_did_sig, subset=["Significancia DiD", "Tendencias paralelas (pre)"]),
                hide_index=True, use_container_width=True,
            )
            st.markdown(
                """<div class="interp-box">Cada fila responde: ¿el gasto del diagnóstico creció más (o menos)
                de lo que ya crecía el gasto no oncológico en ese mismo período? Ese exceso o défict es el
                "Efecto %"; es lo atribuible al punto de corte, no a una tendencia general del INEN. Solo confía
                en una fila si "Significancia DiD" dice Significativo Y "Tendencias paralelas (pre)" dice OK —
                si sale Cuestionable, tratamiento y control ya venían por caminos distintos antes de la norma, y
                el efecto podría no ser de la norma sino de esa diferencia previa.</div>""",
                unsafe_allow_html=True,
            )

# ============================== TAB 6: ABC / Alto costo =========================
with tab_abc:
    st.subheader("Clasificación ABC (Pareto 80/15/5)")
    st.caption("Restringido a los 7 diagnósticos oncológicos que financia FISSAL. Zona A = hasta 80% del gasto "
               "acumulado; B = 80-95%; C = 95-100%.")
    c1, c2, c3 = st.columns(3)
    abc_tipo = c1.selectbox("Tipo de consumo", TIPO_CONSUMO_LIST, index=0)
    abc_iafas = c2.multiselect("IAFAS", IAFAS_LIST, default=IAFAS_LIST)
    abc_diag = c3.multiselect("Diagnóstico", ONCO_DIAGS, default=ONCO_DIAGS)

    years_avail = sorted(product_df["ANNIO"].unique())
    pc1, pc2, pc3 = st.columns(3)
    period_mode = pc1.radio("Período de análisis", ["Todo el período (agregado)", "Año específico"], horizontal=False)
    if period_mode == "Año específico":
        abc_year = pc2.selectbox("Año", years_avail, index=len(years_avail) - 1)
        deflate = False
    else:
        abc_year = None
        base_year = max(years_avail)
        deflate = pc2.checkbox(
            f"Ajustar por inflación (a soles de {base_year}, usando UIT como proxy)", value=False,
            help=f"Multiplica cada año por UIT({base_year})/UIT(año), para que los años más antiguos no queden "
                 "subestimados solo por el paso del tiempo. Es una aproximación — la UIT no es un deflactor de "
                 "precios de salud exacto, pero es la referencia oficial más cercana disponible."
        )
        pc3.caption(f"UIT por año: {', '.join(f'{y}=S/{v:,}' for y, v in UIT_BY_YEAR.items())}")

    base_pdf = product_df[
        (product_df["tipo_consumo"] == abc_tipo)
        & (product_df["iafas"].isin(abc_iafas if abc_iafas else IAFAS_LIST))
        & (product_df["diag_group"].isin(abc_diag if abc_diag else ONCO_DIAGS))
    ].copy()

    pdf = base_pdf if abc_year is None else base_pdf[base_pdf["ANNIO"] == abc_year]
    scope_label = f"año {abc_year}" if abc_year is not None else f"{years_avail[0]}-{years_avail[-1]}" + (
        f", deflactado a soles de {max(years_avail)}" if deflate else ", sin ajustar por inflación")
    if deflate and abc_year is None:
        base_uit = UIT_BY_YEAR[max(years_avail)]
        pdf["total_neto"] = pdf["total_neto"] * (base_uit / pdf["ANNIO"].map(UIT_BY_YEAR))

    st.caption(f"Ámbito actual: **{scope_label}**")
    abc_res = abc_classification(pdf)
    if abc_res is None:
        st.info("No hay datos para esta combinación de filtros.")
    else:
        top_n = st.slider("Ítems a graficar (curva de Pareto)", 10, min(100, len(abc_res.table)), min(30, len(abc_res.table)))
        top = abc_res.table.head(top_n)
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=top["ranking"], y=top["valor_total"], name="Gasto (S/)", marker_color=PRIMARY_COLOR))
        fig3.add_trace(go.Scatter(x=top["ranking"], y=top["%_acumulado"], name="% acumulado", yaxis="y2", line=dict(color=SECONDARY_COLOR, width=2)))
        fig3.update_layout(height=420, yaxis_title="Gasto (S/)", yaxis2=dict(title="% acumulado", overlaying="y", side="right", range=[0, 105]),
                            legend=dict(orientation="h", y=-0.2), margin=dict(t=20))
        st.plotly_chart(fig3, use_container_width=True)

        zc1, zc2 = st.columns([1, 2])
        zone_disp = abc_res.zone_summary.rename(columns={"Zona_ABC": "Zona", "n_items": "N° ítems", "%_items": "% ítems", "valor": "Gasto (S/)", "%_valor": "% gasto"})
        zc1.dataframe(fmt_df(zone_disp, money_cols=["Gasto (S/)"], int_cols=["N° ítems"], pct_cols=["% ítems", "% gasto"]),
                      hide_index=True, use_container_width=True)
        a_pct = abc_res.zone_summary.loc[abc_res.zone_summary["Zona_ABC"] == "A", "%_items"].values
        zc2.markdown(
            f"""<div class="interp-box">La Zona A concentra el <b>{a_pct[0] if len(a_pct) else 0:.1f}%</b> de los
            ítems y aproximadamente 80% del gasto ({scope_label}): son los candidatos prioritarios para control de
            stock estricto, compras centralizadas o negociación de precios. Cruza esta lista con un criterio clínico
            VEN antes de usarla para racionamiento.</div>""",
            unsafe_allow_html=True,
        )
        with st.expander("Ver tabla completa"):
            st.dataframe(
                fmt_df(abc_res.table, money_cols=["valor_total"], int_cols=["cantidad_total", "ranking"], pct_cols=["%_del_total", "%_acumulado"]),
                hide_index=True, use_container_width=True,
            )

        with st.expander("📅 Estabilidad interanual — ¿la Zona A es la misma todos los años?"):
            st.caption("Zona ABC de cada uno de los ítems Top 12 (según el ámbito seleccionado arriba) calculada "
                       "AÑO POR AÑO, por separado. Si un ítem es Zona A en la mayoría de los años, su clasificación "
                       "es una tendencia consistente; si solo es Zona A en un año puntual, esa clasificación agregada "
                       "puede estar dominada por un pico aislado de gasto en ese año — un caso concreto de \"efecto "
                       "del año\" que conviene tener presente antes de usar la lista para negociación de precios.")
            top_items = abc_res.table.head(12)["DESC_CONSUMO"].tolist()
            stability_data = {}
            for y in years_avail:
                yr_pdf = base_pdf[base_pdf["ANNIO"] == y]
                yr_res = abc_classification(yr_pdf)
                zone_map = dict(zip(yr_res.table["DESC_CONSUMO"], yr_res.table["Zona_ABC"])) if yr_res is not None else {}
                stability_data[str(y)] = [zone_map.get(item, "—") for item in top_items]
            stability_df = pd.DataFrame(stability_data, index=top_items)

            def _color_zone(val):
                if val == "A":
                    return f"background-color: {BAD_COLOR}55"
                if val == "B":
                    return f"background-color: {WARN_COLOR}55"
                if val == "C":
                    return f"background-color: {OK_COLOR}55"
                return ""

            st.dataframe(stability_df.style.map(_color_zone), use_container_width=True)

    st.divider()
    st.subheader("Medicamentos de alto costo (umbral DIGEMID: 9 UIT/paciente-año)")
    st.markdown(
        """<div class="interp-box">⚠️ El umbral de 9 UIT se define por <b>paciente-año</b>; esta base solo
        identifica <b>unidades dispensadas</b>, sin trazabilidad por paciente. El costo unitario aquí (gasto/cantidad
        dispensada) es una aproximación, no una réplica exacta del criterio DIGEMID. El umbral ya varía por año
        automáticamente según la UIT vigente cada año (ver tabla en «Período de análisis» arriba).</div>""",
        unsafe_allow_html=True,
    )
    hc_scope = product_df[
        (product_df["iafas"].isin(abc_iafas if abc_iafas else IAFAS_LIST))
        & (product_df["diag_group"].isin(abc_diag if abc_diag else ONCO_DIAGS))
    ]
    if abc_year is not None:
        hc_scope = hc_scope[hc_scope["ANNIO"] == abc_year]
    hc = high_cost_medications(hc_scope)
    n_alto = int(hc["alto_costo_9UIT"].sum())
    st.metric("Productos-año con costo unitario > 9 UIT", n_alto)
    hc_disp = hc.head(30)[["ANNIO", "DESC_CONSUMO", "diag_group", "iafas", "total_cantidad", "costo_unitario", "umbral_9UIT", "total_neto"]] \
        .rename(columns={"ANNIO": "Año", "DESC_CONSUMO": "Medicamento", "diag_group": "Diagnóstico", "total_cantidad": "Cantidad",
                          "costo_unitario": "Costo unitario (S/)", "umbral_9UIT": "Umbral 9 UIT (S/)", "total_neto": "Gasto total (S/)"})
    st.dataframe(
        fmt_df(hc_disp, money_cols=["Costo unitario (S/)", "Umbral 9 UIT (S/)", "Gasto total (S/)"], int_cols=["Cantidad"]),
        hide_index=True, use_container_width=True,
    )

# ============================== TAB 7: Tabla y descarga ==========================
with tab_table:
    st.subheader("Tabla dinámica y descarga")
    group_by = st.radio("Agrupar por", ["Diagnóstico", "IAFAS", "Tipo de consumo"], horizontal=True)
    col_map2 = {"Diagnóstico": "diag_group", "IAFAS": "iafas", "Tipo de consumo": "tipo_consumo"}
    piv = filtered.pivot_table(index="tiempo", columns=col_map2[group_by], values="total_neto", aggfunc="sum").fillna(0)
    st.dataframe(piv.style.format("{:,.0f}"), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Descargar tabla dinámica (CSV)", piv.to_csv().encode("utf-8"), file_name="gasto_pivot.csv", mime="text/csv")
    c2.download_button("⬇️ Descargar datos filtrados (CSV, sin pivotear)", filtered.to_csv(index=False).encode("utf-8"), file_name="gasto_filtrado.csv", mime="text/csv")

# ============================== TAB 9: Exploración base adicional ================
with tab_sec_exp:
    st.subheader("Exploración de la base de datos adicional")

    if "secondary_serie" not in st.session_state:
        st.info(
            "Aún no se ha cargado ninguna base de datos adicional. "
            "Usa el panel lateral → **📊 Base de datos adicional** para subir un archivo CSV o Excel."
        )
    else:
        _sec_s: pd.Series = st.session_state["secondary_serie"]
        _sec_lbl: str = st.session_state.get("secondary_label", "Serie adicional")
        _sec_units_lbl: str = st.session_state.get("secondary_units", "Valor")
        _sec_freq_exp: str = st.session_state.get("secondary_freq", "MS")
        _is_annual_exp = _sec_freq_exp == "YS"

        if _is_annual_exp:
            st.caption(
                f"**{_sec_lbl}** · {len(_sec_s)} año(s) · "
                f"{_sec_s.index.min():%Y} a {_sec_s.index.max():%Y}"
            )
        else:
            st.caption(
                f"**{_sec_lbl}** · {len(_sec_s)} meses · "
                f"{_sec_s.index.min():%b-%Y} a {_sec_s.index.max():%b-%Y}"
            )

        # ── Helpers ──────────────────────────────────────────────────────────
        _sec_cols_dict_kpi = st.session_state.get("secondary_cols_dict", {})
        _sec_val_cols_kpi = st.session_state.get("secondary_val_cols", [])
        _first_fmt = "%Y" if _is_annual_exp else "%b-%y"
        _peak_fmt = "%Y" if _is_annual_exp else "%b-%Y"

        def _cagr(s: pd.Series) -> str:
            _snn = s[s > 0].dropna()
            if len(_snn) < 2:
                return "n/d"
            _n = (_snn.index[-1] - _snn.index[0]).days / 365.25
            if _n <= 0 or _snn.iloc[0] <= 0:
                return "n/d"
            return f"{((_snn.iloc[-1] / _snn.iloc[0]) ** (1 / _n) - 1) * 100:+.1f}%/año"

        def _render_kpi_row(sf: pd.Series, label: str):
            """6 KPIs: Total, Promedio, Variación, CAGR, Año pico, MK."""
            _k1, _k2, _k3, _k4, _k5, _k6 = st.columns(6)
            _k1.metric(f"Total — {label}", f"{sf.sum():,.2f}")
            _k2.metric("Promedio anual" if _is_annual_exp else "Promedio mensual", f"{sf.mean():,.2f}")
            _var = (f"{(sf.iloc[-1] / sf.iloc[0] - 1) * 100:+.1f}%"
                    if len(sf) >= 2 and sf.iloc[0] != 0 else "n/d")
            _k3.metric(
                f"Variación {sf.index[0]:{_first_fmt}} → {sf.index[-1]:{_first_fmt}}"
                if len(sf) >= 2 else "Variación", _var,
            )
            _k4.metric("CAGR (crec. anual compuesto)", _cagr(sf))
            _peak = sf.idxmax() if len(sf) else None
            _k5.metric(
                "Año pico" if _is_annual_exp else "Mes pico",
                f"{_peak:{_peak_fmt}}" if _peak is not None else "n/d",
                f"{sf.max():,.2f}" if len(sf) else "",
            )
            _mk = mann_kendall_trend(sf.values) if len(sf) >= 4 else None
            _k6.metric("Tendencia (Mann-Kendall)",
                       _mk.trend if _mk else "n/d",
                       f"p={_mk.p_value:.3f}" if _mk else "")
            return _mk

        # ── KPIs por columna (filtro de años independiente por columna) ───────
        _mk_by_col: dict = {}
        if len(_sec_val_cols_kpi) > 1 and _sec_cols_dict_kpi:
            for _kpi_col in _sec_val_cols_kpi:
                _ks = _sec_cols_dict_kpi.get(_kpi_col)
                if _ks is None:
                    continue
                st.caption(f"**{_kpi_col}**")
                _excl_col = st.multiselect(
                    "Excluir años",
                    options=sorted({ts.year for ts in _ks.index}),
                    default=[], key=f"xp_excl_{_kpi_col}",
                    label_visibility="collapsed",
                    placeholder="Excluir años del análisis (opcional)…",
                    help=f"Los años seleccionados se excluyen del análisis KPI de {_kpi_col} (no afecta el gráfico principal).",
                )
                _ks_f = _ks[~_ks.index.year.isin(_excl_col)] if _excl_col else _ks
                _mk_by_col[_kpi_col] = _render_kpi_row(_ks_f, _kpi_col)
            _kpi_lbl = _sec_val_cols_kpi[0]
            _kpi_s: pd.Series = _sec_cols_dict_kpi.get(_kpi_lbl, _sec_s)
            _mk_sec = _mk_by_col.get(_kpi_lbl)
        else:
            _excl_single = st.multiselect(
                "Excluir años del análisis KPI / estadísticas",
                options=sorted({ts.year for ts in _sec_s.index}),
                default=[], key="xp_excl_single",
                placeholder="Excluir años del análisis (opcional)…",
            )
            _ks_single = _sec_s[~_sec_s.index.year.isin(_excl_single)] if _excl_single else _sec_s
            _kpi_s = _ks_single
            _kpi_lbl = _sec_lbl
            _mk_sec = _render_kpi_row(_ks_single, _sec_lbl)
            _mk_by_col[_sec_lbl] = _mk_sec

        st.divider()

        # --- Opciones de visualización locales ---
        _xp_c1, _xp_c2, _xp_c3 = st.columns(3)
        _xp_chart = _xp_c1.radio("Tipo de gráfico", ["Línea", "Barras"], horizontal=True, key="xp_chart")
        _xp_markers = _xp_c2.checkbox("Marcadores", value=True, key="xp_markers")
        _xp_lowess = _xp_c3.checkbox("LOWESS (suavizado)", value=False, key="xp_lowess")
        _xp_log = _xp_c1.checkbox("Escala logarítmica", value=False, key="xp_log")
        _xp_its = _xp_c2.checkbox(
            "Superponer ITS (puntos de corte activos)", value=False, key="xp_its",
            help="Aplica el mismo modelo ITS de la base principal sobre esta serie, usando los puntos de corte normativos marcados en el panel lateral.",
        )
        _xp_breaks = _xp_c3.checkbox("Mostrar líneas de corte normativo", value=False, key="xp_breaks")

        # --- Rango de fecha para exploración ---
        _xp_lo = _sec_s.index.min().to_pydatetime()
        _xp_hi = _sec_s.index.max().to_pydatetime()
        _slider_lbl = "Rango de años (exploración)" if _is_annual_exp else "Rango de meses (exploración)"
        _slider_fmt = "YYYY" if _is_annual_exp else "MMM YYYY"
        if _xp_lo < _xp_hi:
            _xp_range = st.slider(
                _slider_lbl,
                min_value=_xp_lo, max_value=_xp_hi,
                value=(_xp_lo, _xp_hi), format=_slider_fmt, key="xp_range",
            )
            _sec_s_filt = _sec_s[
                (_sec_s.index >= pd.Timestamp(_xp_range[0])) &
                (_sec_s.index <= pd.Timestamp(_xp_range[1]))
            ]
        else:
            # Serie de un solo período: no hay rango que filtrar
            _solo_lbl = f"{_xp_lo:%Y}" if _is_annual_exp else f"{_xp_lo:%b-%Y}"
            st.caption(f"Serie de un único período: {_solo_lbl}")
            _sec_s_filt = _sec_s.copy()

        # --- Helper: serie filtrada por rango + exclusión de años por columna ---
        def _filtered_for_stats(col_name: str, raw_s: pd.Series) -> pd.Series:
            """Aplica rango de fecha del slider + exclusión de años por columna."""
            # 1) Filtro por rango de fechas
            if _xp_lo < _xp_hi:
                _s2 = raw_s[
                    (raw_s.index >= pd.Timestamp(_xp_range[0])) &
                    (raw_s.index <= pd.Timestamp(_xp_range[1]))
                ]
            else:
                _s2 = raw_s.copy()
            # 2) Exclusión de años (lee de session_state según clave del widget)
            _excl_key = f"xp_excl_{col_name}" if len(_sec_val_cols_kpi) > 1 else "xp_excl_single"
            _excl_yrs = st.session_state.get(_excl_key, [])
            if _excl_yrs:
                _s2 = _s2[~_s2.index.year.isin(_excl_yrs)]
            return _s2

        # --- Selector de columnas para el gráfico (solo si hay varias) ---
        _sec_cols_dict_exp = _sec_cols_dict_kpi   # alias (ya definido arriba para KPIs)
        _sec_val_cols_exp = _sec_val_cols_kpi
        _xp_show_cols: list[str] = []
        if len(_sec_val_cols_exp) > 1:
            _xp_show_cols = st.multiselect(
                "Columnas a graficar individualmente",
                options=_sec_val_cols_exp,
                default=_sec_val_cols_exp,
                key="xp_show_cols",
                help="Cada columna se traza como una línea independiente. La serie combinada siempre se muestra si no hay ninguna columna seleccionada.",
            )

        # --- Estadísticas descriptivas + Distribución ---
        _desc_col, _dist_col = st.columns([1, 1])
        _rename_map = {
            "count": "N obs.", "mean": "Media", "std": "Desv. estándar",
            "min": "Mínimo", "25%": "Percentil 25", "50%": "Mediana",
            "75%": "Percentil 75", "max": "Máximo",
        }
        with _desc_col:
            st.markdown('<div class="mini-title">Estadísticas descriptivas</div>', unsafe_allow_html=True)
            if len(_sec_val_cols_kpi) > 1 and _sec_cols_dict_kpi:
                _desc_frames = {}
                for _dcol in _sec_val_cols_kpi:
                    _ds_raw = _sec_cols_dict_kpi.get(_dcol)
                    if _ds_raw is None:
                        continue
                    _ds_filt = _filtered_for_stats(_dcol, _ds_raw)
                    _desc_frames[_dcol] = _ds_filt.describe().rename(_rename_map)
                if _desc_frames:
                    _desc_multi = pd.DataFrame(_desc_frames)
                    st.dataframe(_desc_multi.style.format("{:,.4f}"), use_container_width=True)
                # MK para todas las columnas
                for _mkcol, _mkres in _mk_by_col.items():
                    if _mkres:
                        st.markdown(
                            f'<div class="interp-box">Mann-Kendall <b>{_mkcol}</b>: tendencia '
                            f'<b>{_mkres.trend}</b> (p={_mkres.p_value:.4f}, '
                            f'variante: {_mkres.variant_used}). '
                            f'Pendiente de Sen: <b>{_mkres.sen_slope:,.4f}</b> u/período.</div>',
                            unsafe_allow_html=True,
                        )
            else:
                _ds_filt_single = _filtered_for_stats(_sec_lbl, _sec_s)
                _desc = _ds_filt_single.describe().rename(_rename_map)
                st.dataframe(
                    _desc.to_frame(name=_sec_lbl).style.format("{:,.4f}"),
                    use_container_width=True,
                )
                if _mk_sec:
                    st.markdown(
                        f'<div class="interp-box">Mann-Kendall ({_kpi_lbl}): tendencia '
                        f'<b>{_mk_sec.trend}</b> (p={_mk_sec.p_value:.4f}, '
                        f'variante: {_mk_sec.variant_used}). '
                        f'Pendiente de Sen: <b>{_mk_sec.sen_slope:,.4f}</b> u/período.</div>',
                        unsafe_allow_html=True,
                    )

        with _dist_col:
            st.markdown('<div class="mini-title">Distribución</div>', unsafe_allow_html=True)
            _hist_cols = (
                list(_sec_cols_dict_kpi.items())
                if (len(_sec_val_cols_kpi) > 1 and _sec_cols_dict_kpi)
                else [(_sec_lbl, _sec_s)]
            )
            _hist_palette = DEFAULT_PALETTE
            _hist_pairs = [_hist_cols[i:i+2] for i in range(0, len(_hist_cols), 2)]
            for _hrow in _hist_pairs:
                _hcols_ui = st.columns(len(_hrow))
                for _hci, (_hcol, _hs) in enumerate(_hrow):
                    _hs_filt = _filtered_for_stats(_hcol, _hs)
                    _hvnz = _hs_filt[_hs_filt > 0]
                    _hcolor = _hist_palette[list(dict(_hist_cols).keys()).index(_hcol) % len(_hist_palette)]
                    _fig_h = go.Figure()
                    if len(_hvnz) > 0:
                        _fig_h.add_trace(go.Histogram(
                            x=_hvnz.values, nbinsx=15, name=_hcol,
                            marker_color=_hcolor, opacity=0.85,
                        ))
                        _fig_h.add_vline(
                            x=float(_hvnz.mean()), line_dash="dash",
                            line_color=SECONDARY_COLOR, annotation_text="Media",
                        )
                        _fig_h.add_vline(
                            x=float(_hvnz.median()), line_dash="dot",
                            line_color=OK_COLOR, annotation_text="Mediana",
                        )
                    _fig_h.update_layout(
                        title=dict(text=_hcol, font=dict(size=12)),
                        height=260, xaxis_title=_sec_units_lbl, yaxis_title="Frecuencia",
                        margin=dict(t=35, b=30), showlegend=False,
                    )
                    _hcols_ui[_hci].plotly_chart(_fig_h, use_container_width=True)

        st.divider()

        # --- Gráfico principal (debajo de estadísticas) ---
        _fig_sec = go.Figure()
        _mode_sec = "lines+markers" if _xp_markers else "lines"

        if _xp_show_cols and _sec_cols_dict_exp:
            _xp_palette = DEFAULT_PALETTE
            for _ci, _col_name in enumerate(_xp_show_cols):
                _col_s = _sec_cols_dict_exp.get(_col_name)
                if _col_s is None:
                    continue
                _col_s_filt = _col_s[
                    (_col_s.index >= pd.Timestamp(_xp_range[0])) &
                    (_col_s.index <= pd.Timestamp(_xp_range[1]))
                ]
                _col_color = _xp_palette[_ci % len(_xp_palette)]
                if _xp_chart == "Barras":
                    _fig_sec.add_trace(go.Bar(
                        x=_col_s_filt.index, y=_col_s_filt.values,
                        name=_col_name, marker_color=_col_color,
                    ))
                else:
                    _fig_sec.add_trace(go.Scatter(
                        x=_col_s_filt.index, y=_col_s_filt.values, mode=_mode_sec,
                        name=_col_name, line=dict(width=2, color=_col_color), marker=dict(size=5),
                    ))
        else:
            if _xp_chart == "Barras":
                _fig_sec.add_trace(go.Bar(
                    x=_sec_s_filt.index, y=_sec_s_filt.values,
                    name=_sec_lbl, marker_color=PRIMARY_COLOR,
                ))
            else:
                _fig_sec.add_trace(go.Scatter(
                    x=_sec_s_filt.index, y=_sec_s_filt.values, mode=_mode_sec,
                    name=_sec_lbl, line=dict(width=2, color=PRIMARY_COLOR), marker=dict(size=5),
                ))

        if _xp_its and len(_sec_s_filt) >= 12:
            _its_sec = segmented_its(_sec_s_filt, BREAKS, BREAK_LABELS, alpha=alpha)
            if _its_sec is not None:
                _fig_sec.add_trace(go.Scatter(
                    x=_its_sec.fitted.index, y=_its_sec.fitted.values, mode="lines",
                    name="Ajuste ITS", line=dict(width=2, color=SECONDARY_COLOR, dash="dash"),
                ))

        if _xp_lowess and len(_sec_s_filt) >= 6:
            _sm_sec = smooth_series(_sec_s_filt.values)
            if _sm_sec is not None:
                _fig_sec.add_trace(go.Scatter(
                    x=_sec_s_filt.index, y=_sm_sec, mode="lines",
                    name="LOWESS", line=dict(width=2, color=OK_COLOR),
                ))

        if _xp_breaks:
            for _k, _bdate in active_breaks.items():
                if pd.Timestamp(_xp_range[0]) < _bdate <= pd.Timestamp(_xp_range[1]):
                    _fig_sec.add_vline(x=_bdate, line_dash="dot", line_color="gray", opacity=0.7)
                    _fig_sec.add_annotation(
                        x=_bdate, y=1.04, yref="paper", showarrow=False,
                        text=f"<b>{BREAK_CODES.get(_k, _k)}</b>", font=dict(size=12, color="#555555"),
                    )

        # Eje X: mostrar todos los años (dtick="M12" fuerza una marca por año en datos anuales y mensuales)
        _xaxis_cfg = dict(dtick="M12", tickformat="%Y", tickangle=-45)

        _fig_sec.update_layout(
            height=420, hovermode="x unified",
            yaxis_title=_sec_units_lbl,
            yaxis_type="log" if _xp_log else "linear",
            xaxis=_xaxis_cfg,
            legend=dict(orientation="h", y=-0.25), margin=dict(t=30), barmode="group",
        )
        st.plotly_chart(_fig_sec, use_container_width=True)

        st.divider()

        # --- Análisis de tendencia por columna ---
        st.markdown("#### 📈 Análisis de tendencia detallado")
        _trend_cols_list = (
            list(_sec_cols_dict_kpi.items())
            if (len(_sec_val_cols_kpi) > 1 and _sec_cols_dict_kpi)
            else [(_sec_lbl, _sec_s)]
        )
        import numpy as np
        for _tcol, _ts_raw in _trend_cols_list:
            _ts = _filtered_for_stats(_tcol, _ts_raw)
            with st.expander(f"🔎 Tendencia — {_tcol}", expanded=False):
                if len(_ts) < 4:
                    st.info("Se necesitan al menos 4 períodos para el análisis de tendencia.")
                    continue
                _mk_t = mann_kendall_trend(_ts.values)
                _t1, _t2, _t3 = st.columns(3)
                _t1.metric("Tendencia (Mann-Kendall)", _mk_t.trend if _mk_t else "n/d")
                _t2.metric("p-valor", f"{_mk_t.p_value:.4f}" if _mk_t else "n/d",
                           significance_label(_mk_t.p_value, alpha) if _mk_t else "")
                _t3.metric("Pendiente de Sen (u/período)",
                           f"{_mk_t.sen_slope:,.4f}" if _mk_t else "n/d")
                # Línea de tendencia lineal simple
                _ts_x = np.arange(len(_ts), dtype=float)
                _ts_y = _ts.values.astype(float)
                _valid = ~np.isnan(_ts_y)
                if _valid.sum() >= 2:
                    _slope, _intercept = np.polyfit(_ts_x[_valid], _ts_y[_valid], 1)
                    _trend_y = _slope * _ts_x + _intercept
                else:
                    _trend_y = None
                _fig_t = go.Figure()
                _fig_t.add_trace(go.Scatter(
                    x=_ts.index, y=_ts.values, mode="lines+markers",
                    name=_tcol, line=dict(width=2, color=PRIMARY_COLOR), marker=dict(size=5),
                ))
                if _trend_y is not None:
                    _fig_t.add_trace(go.Scatter(
                        x=_ts.index, y=_trend_y, mode="lines",
                        name="Tendencia lineal", line=dict(width=2, color=SECONDARY_COLOR, dash="dash"),
                    ))
                _fig_t.update_layout(
                    height=300, hovermode="x unified",
                    yaxis_title=_sec_units_lbl,
                    xaxis=dict(dtick="M12", tickformat="%Y", tickangle=-45),
                    legend=dict(orientation="h", y=-0.3), margin=dict(t=20),
                )
                st.plotly_chart(_fig_t, use_container_width=True)
                if _mk_t:
                    st.markdown(
                        f'<div class="interp-box">Tendencia <b>{_mk_t.trend}</b> '
                        f'(p={_mk_t.p_value:.4f}, variante: {_mk_t.variant_used}). '
                        f'Pendiente de Sen: <b>{_mk_t.sen_slope:,.4f}</b> unidades/período. '
                        f'{"Tendencia estadísticamente significativa." if _mk_t.p_value < alpha else "No significativa al nivel α=" + str(alpha) + "."}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        st.divider()

        # --- Detección de quiebre automático ---
        with st.expander("🔍 Detección automática de punto de quiebre"):
            # Selector de variable cuando hay varias columnas
            _bp_col_options = (
                list(_sec_cols_dict_kpi.keys())
                if (len(_sec_val_cols_kpi) > 1 and _sec_cols_dict_kpi)
                else [_sec_lbl]
            )
            if len(_bp_col_options) > 1:
                _bp_sel_col = st.selectbox(
                    "Variable a analizar", options=_bp_col_options, key="bp_sel_col",
                )
                _bp_raw = _sec_cols_dict_kpi.get(_bp_sel_col, _sec_s)
            else:
                _bp_sel_col = _sec_lbl
                _bp_raw = _sec_s
            # Aplicar rango de fecha del slider
            _bp_series = _bp_raw[
                (_bp_raw.index >= pd.Timestamp(_xp_range[0])) &
                (_bp_raw.index <= pd.Timestamp(_xp_range[1]))
            ] if _xp_lo < _xp_hi else _bp_raw.copy()

            _auto_sec = detect_breakpoint_auto(_bp_series) if len(_bp_series) >= 12 else None
            if _auto_sec is None:
                st.info("Necesitas al menos 12 períodos en el rango seleccionado.")
            else:
                _bp_lbl = "Año" if _is_annual_exp else "Mes"
                _ac1, _ac2, _ac3 = st.columns(3)
                _ac1.metric(
                    f"{_bp_lbl} de quiebre detectado",
                    f"{_auto_sec.date:%Y}" if _is_annual_exp else f"{_auto_sec.date:%b-%Y}",
                )
                _ac2.metric("p-valor (test de Chow)", f"{_auto_sec.p_value:.4f}",
                            significance_label(_auto_sec.p_value, alpha))
                _ac3.metric("Pendiente antes → después",
                            f"{_auto_sec.slope_before:,.4f} → {_auto_sec.slope_after:,.4f}")
                _fig_ab = go.Figure()
                _fig_ab.add_trace(go.Scatter(
                    x=_bp_series.index, y=_bp_series.values, mode="markers",
                    name="Observado", marker=dict(color=PRIMARY_COLOR, size=5),
                ))
                _fig_ab.add_trace(go.Scatter(
                    x=_auto_sec.x_before, y=_auto_sec.y_pred_before, mode="lines",
                    name="Ajuste antes", line=dict(color=SECONDARY_COLOR, width=2),
                ))
                _fig_ab.add_trace(go.Scatter(
                    x=_auto_sec.x_after, y=_auto_sec.y_pred_after, mode="lines",
                    name="Ajuste después", line=dict(color=OK_COLOR, width=2),
                ))
                _fig_ab.add_vline(x=_auto_sec.date, line_dash="dot", line_color="gray")
                _fig_ab.update_layout(
                    height=320, hovermode="x unified",
                    yaxis_title=_sec_units_lbl, margin=dict(t=20),
                    xaxis=dict(dtick="M12", tickformat="%Y", tickangle=-45),
                    legend=dict(orientation="h", y=-0.3),
                )
                st.plotly_chart(_fig_ab, use_container_width=True)

        # --- Descarga ---
        st.download_button(
            "⬇️ Descargar serie adicional (CSV)",
            _sec_s_filt.reset_index().rename(columns={"index": "tiempo", 0: _sec_lbl}).to_csv(index=False).encode("utf-8"),
            file_name="serie_adicional.csv", mime="text/csv",
        )

# ============================== TAB 10: Análisis cruzado =========================
with tab_cross:
    st.subheader("Análisis cruzado: base principal vs. base adicional")

    if "secondary_serie" not in st.session_state:
        st.info(
            "Carga una base de datos adicional desde el panel lateral "
            "(📊 **Base de datos adicional**) para habilitar este análisis."
        )
    else:
        _sec_s_x: pd.Series = st.session_state["secondary_serie"]
        _sec_lbl_x: str = st.session_state.get("secondary_label", "Serie adicional")
        _sec_units_x: str = st.session_state.get("secondary_units", "Valor")
        _sec_freq_x: str = st.session_state.get("secondary_freq", "MS")

        # Si la serie adicional es anual, expandirla a frecuencia mensual
        # usando forward-fill para que la intersección con la serie mensual
        # principal funcione sin forzar ceros en los meses intermedios.
        if _sec_freq_x == "YS":
            _sec_monthly_idx = pd.date_range(
                _sec_s_x.index.min(), _sec_s_x.index.max() + pd.offsets.YearEnd(), freq="MS"
            )
            _sec_s_x = _sec_s_x.reindex(_sec_monthly_idx).ffill()

        # Alinear ambas series al período de intersección
        _prim_s: pd.Series = serie  # serie de la selección actual del panel lateral
        _idx_common = _prim_s.index.intersection(_sec_s_x.index)

        if len(_idx_common) < 6:
            st.warning(
                f"Las dos series solo se solapan en {len(_idx_common)} mes(es). "
                "Ajusta el rango de fechas en el panel lateral o sube una base adicional "
                "con más solapamiento con la serie principal."
            )
        else:
            _p = _prim_s.reindex(_idx_common)
            _s = _sec_s_x.reindex(_idx_common)

            st.caption(
                f"Período de solapamiento: {_idx_common.min():%b-%Y} a {_idx_common.max():%b-%Y} "
                f"· {len(_idx_common)} meses en común"
            )

            # -----------------------------------------------------------
            # Sección 0: Gráfico cruzado directo (valores brutos, doble eje)
            # -----------------------------------------------------------
            st.markdown("#### Gráfico cruzado: serie temporal principal + base adicional")

            _cx0_c1, _cx0_c2, _cx0_c3, _cx0_c4 = st.columns(4)
            _cx0_markers = _cx0_c1.checkbox("Marcadores", value=True, key="cx0_markers")
            _cx0_log_l  = _cx0_c2.checkbox("Log eje izq.", value=False, key="cx0_log_l")
            _cx0_log_r  = _cx0_c3.checkbox("Log eje dcho.", value=False, key="cx0_log_r")
            _cx0_breaks = _cx0_c4.checkbox("Líneas de corte", value=True, key="cx0_breaks")

            # Selector de columnas adicionales (si se cargaron varias)
            _cx0_cols_dict = st.session_state.get("secondary_cols_dict", {})
            _cx0_val_cols  = st.session_state.get("secondary_val_cols", [])
            _cx0_sel_cols: list[str] = []
            if len(_cx0_val_cols) > 1:
                _cx0_sel_cols = st.multiselect(
                    "Columnas adicionales a superponer",
                    options=_cx0_val_cols,
                    default=_cx0_val_cols,
                    key="cx0_sel_cols",
                    help="Cada columna se añade como línea independiente en el eje derecho.",
                )

            _cx0_mode = "lines+markers" if _cx0_markers else "lines"
            _fig_cx0 = go.Figure()

            # Eje izquierdo — serie principal
            _fig_cx0.add_trace(go.Scatter(
                x=_p.index, y=_p.values, mode=_cx0_mode,
                name="Gasto oncológico principal",
                line=dict(color=PRIMARY_COLOR, width=2), marker=dict(size=5),
                yaxis="y1",
            ))

            # Eje derecho — columnas individuales o serie combinada
            _cx0_palette = [SECONDARY_COLOR, OK_COLOR, WARN_COLOR, "#8850C4", "#5C6AC4"]
            if _cx0_sel_cols and _cx0_cols_dict:
                _sec_s_x_monthly = _sec_s_x  # ya ffill-eado arriba si era anual
                for _ci, _col_name in enumerate(_cx0_sel_cols):
                    _col_s_raw = _cx0_cols_dict.get(_col_name)
                    if _col_s_raw is None:
                        continue
                    # expandir a mensual si era anual
                    if _sec_freq_x == "YS":
                        _col_monthly_idx = pd.date_range(
                            _col_s_raw.index.min(),
                            _col_s_raw.index.max() + pd.offsets.YearEnd(),
                            freq="MS",
                        )
                        _col_s_raw = _col_s_raw.reindex(_col_monthly_idx).ffill()
                    _col_aligned = _col_s_raw.reindex(_idx_common)
                    _fig_cx0.add_trace(go.Scatter(
                        x=_col_aligned.index, y=_col_aligned.values, mode=_cx0_mode,
                        name=_col_name,
                        line=dict(color=_cx0_palette[_ci % len(_cx0_palette)], width=2),
                        marker=dict(size=5), yaxis="y2",
                    ))
            else:
                _fig_cx0.add_trace(go.Scatter(
                    x=_s.index, y=_s.values, mode=_cx0_mode,
                    name=_sec_lbl_x,
                    line=dict(color=SECONDARY_COLOR, width=2), marker=dict(size=5),
                    yaxis="y2",
                ))

            if _cx0_breaks:
                for _k, _bdate in active_breaks.items():
                    if _idx_common.min() < _bdate <= _idx_common.max():
                        _fig_cx0.add_vline(x=_bdate, line_dash="dot", line_color="gray", opacity=0.7)
                        _fig_cx0.add_annotation(
                            x=_bdate, y=1.04, yref="paper", showarrow=False,
                            text=f"<b>{BREAK_CODES.get(_k, _k)}</b>",
                            font=dict(size=12, color="#555555"),
                        )

            _fig_cx0.update_layout(
                height=440, hovermode="x unified",
                yaxis=dict(
                    title=dict(text="Gasto oncológico principal (S/)", font=dict(color=PRIMARY_COLOR)),
                    type="log" if _cx0_log_l else "linear",
                    tickfont=dict(color=PRIMARY_COLOR),
                ),
                yaxis2=dict(
                    title=dict(text=f"{_sec_lbl_x} ({_sec_units_x})", font=dict(color=SECONDARY_COLOR)),
                    type="log" if _cx0_log_r else "linear",
                    overlaying="y", side="right",
                    tickfont=dict(color=SECONDARY_COLOR),
                ),
                legend=dict(orientation="h", y=-0.28),
                margin=dict(t=35),
            )
            st.plotly_chart(_fig_cx0, use_container_width=True)

            st.divider()

            # -----------------------------------------------------------
            # Sección 1: Comparación visual normalizada
            # -----------------------------------------------------------
            st.markdown("#### 1. Comparación visual (series normalizadas, base = 100 en primer mes)")

            _cx_opts_c1, _cx_opts_c2 = st.columns(2)
            _cx_show_raw = _cx_opts_c1.checkbox(
                "Mostrar también valores absolutos (doble eje)", value=False, key="cx_raw"
            )
            _cx_breaks_on = _cx_opts_c2.checkbox(
                "Mostrar puntos de corte normativos", value=True, key="cx_breaks"
            )

            _p_base = _p.iloc[0] if _p.iloc[0] != 0 else 1
            _s_base = _s.iloc[0] if _s.iloc[0] != 0 else 1

            _fig_cmp = go.Figure()
            _fig_cmp.add_trace(go.Scatter(
                x=_idx_common, y=_p.values / _p_base * 100, mode="lines",
                name=f"Principal (gasto oncológico) — índice",
                line=dict(color=PRIMARY_COLOR, width=2),
            ))
            _fig_cmp.add_trace(go.Scatter(
                x=_idx_common, y=_s.values / _s_base * 100, mode="lines",
                name=f"{_sec_lbl_x} — índice",
                line=dict(color=SECONDARY_COLOR, width=2),
            ))

            if _cx_show_raw:
                _fig_cmp.add_trace(go.Scatter(
                    x=_idx_common, y=_p.values, mode="lines",
                    name="Principal (S/, eje dcho.)", yaxis="y2",
                    line=dict(color=PRIMARY_COLOR, width=1, dash="dot"),
                ))
                _fig_cmp.add_trace(go.Scatter(
                    x=_idx_common, y=_s.values, mode="lines",
                    name=f"{_sec_lbl_x} (eje dcho.)", yaxis="y2",
                    line=dict(color=SECONDARY_COLOR, width=1, dash="dot"),
                ))
                _fig_cmp.update_layout(
                    yaxis2=dict(title="Valores absolutos", overlaying="y", side="right"),
                )

            if _cx_breaks_on:
                for _k, _bdate in active_breaks.items():
                    if _idx_common.min() < _bdate <= _idx_common.max():
                        _fig_cmp.add_vline(x=_bdate, line_dash="dot", line_color="gray", opacity=0.7)
                        _fig_cmp.add_annotation(
                            x=_bdate, y=1.04, yref="paper", showarrow=False,
                            text=f"<b>{BREAK_CODES.get(_k, _k)}</b>",
                            font=dict(size=12, color="#555555"),
                        )

            _fig_cmp.update_layout(
                height=400, hovermode="x unified",
                yaxis_title="Índice (primer mes = 100)",
                legend=dict(orientation="h", y=-0.25), margin=dict(t=30),
            )
            st.plotly_chart(_fig_cmp, use_container_width=True)

            st.divider()

            # -----------------------------------------------------------
            # Sección 2: Correlación
            # -----------------------------------------------------------
            st.markdown("#### 2. Correlación contemporánea")

            from scipy.stats import pearsonr, spearmanr

            _mask_valid = (_p > 0) & (_s > 0)
            _p_v = _p[_mask_valid].values
            _s_v = _s[_mask_valid].values

            if len(_p_v) >= 4:
                _r_pear, _p_pear = pearsonr(_p_v, _s_v)
                _r_spear, _p_spear = spearmanr(_p_v, _s_v)

                _cc1, _cc2, _cc3, _cc4 = st.columns(4)
                _cc1.metric("Pearson r", f"{_r_pear:.4f}")
                _cc2.metric("p-valor (Pearson)", f"{_p_pear:.4f}", significance_label(_p_pear, alpha))
                _cc3.metric("Spearman ρ", f"{_r_spear:.4f}")
                _cc4.metric("p-valor (Spearman)", f"{_p_spear:.4f}", significance_label(_p_spear, alpha))

                # Scatter plot coloreado por fecha
                _colors_scatter = np.arange(len(_p_v))
                _fig_sc = go.Figure()
                _fig_sc.add_trace(go.Scatter(
                    x=_p_v, y=_s_v, mode="markers",
                    marker=dict(
                        color=_colors_scatter, colorscale="Blues",
                        showscale=True, size=7,
                        colorbar=dict(title="Mes (cronológico)"),
                    ),
                    text=[f"{d:%b-%Y}" for d in _idx_common[_mask_valid]],
                    hovertemplate="<b>%{text}</b><br>Principal: %{x:,.2f}<br>"
                                  + f"{_sec_lbl_x}: " + "%{y:,.2f}<extra></extra>",
                    name="Meses",
                ))
                # Línea de tendencia
                if len(_p_v) >= 3:
                    _z = np.polyfit(_p_v, _s_v, 1)
                    _x_trend = np.linspace(_p_v.min(), _p_v.max(), 100)
                    _fig_sc.add_trace(go.Scatter(
                        x=_x_trend, y=np.polyval(_z, _x_trend), mode="lines",
                        name=f"Tendencia (r={_r_pear:.3f})",
                        line=dict(color=SECONDARY_COLOR, width=2, dash="dash"),
                    ))
                _fig_sc.update_layout(
                    height=380, xaxis_title="Gasto oncológico principal (S/)",
                    yaxis_title=f"{_sec_lbl_x} ({_sec_units_x})",
                    legend=dict(orientation="h", y=-0.25), margin=dict(t=20),
                )
                st.plotly_chart(_fig_sc, use_container_width=True)

                interp_r = (
                    "correlación positiva fuerte" if _r_pear >= 0.7
                    else "correlación positiva moderada" if _r_pear >= 0.4
                    else "correlación positiva débil" if _r_pear >= 0.1
                    else "correlación negativa fuerte" if _r_pear <= -0.7
                    else "correlación negativa moderada" if _r_pear <= -0.4
                    else "correlación negativa débil" if _r_pear <= -0.1
                    else "sin correlación lineal apreciable"
                )
                st.markdown(
                    f'<div class="interp-box">Las dos series muestran <b>{interp_r}</b> '
                    f'(Pearson r={_r_pear:.3f}, p={_p_pear:.4f}; Spearman ρ={_r_spear:.3f}, '
                    f'p={_p_spear:.4f}). El scatter plot está coloreado cronológicamente — '
                    f'el patrón de dispersión en la dirección del tiempo puede revelar si la '
                    f'asociación cambió en distintos subperíodos.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("Muy pocos meses con valores positivos en ambas series para calcular correlación.")

            st.divider()

            # -----------------------------------------------------------
            # Sección 3: Función de correlación cruzada (CCF)
            # -----------------------------------------------------------
            st.markdown("#### 3. Correlación cruzada con rezago (lead-lag)")
            st.caption(
                "Muestra si la serie adicional ANTICIPA o SIGUE al gasto oncológico. "
                "Un rezago negativo (lag<0) significativo indica que la variable adicional "
                "predice el gasto oncológico con ese número de meses de anticipación."
            )

            _max_lag = st.slider(
                "Rezagos máximos a explorar (meses)", 3, min(18, len(_idx_common) // 3), 6, key="cx_lag"
            )

            _pv = _p.values.astype(float)
            _sv = _s.values.astype(float)
            _pv_std = (_pv - _pv.mean()) / (_pv.std() + 1e-12)
            _sv_std = (_sv - _sv.mean()) / (_sv.std() + 1e-12)

            _lags = np.arange(-_max_lag, _max_lag + 1)
            _ccf_vals = []
            for _lag in _lags:
                if _lag < 0:
                    _ccf_vals.append(float(np.corrcoef(_pv_std[-_lag:], _sv_std[:_lag])[0, 1]))
                elif _lag > 0:
                    _ccf_vals.append(float(np.corrcoef(_pv_std[:-_lag], _sv_std[_lag:])[0, 1]))
                else:
                    _ccf_vals.append(float(np.corrcoef(_pv_std, _sv_std)[0, 1]))

            _sig_band = 1.96 / np.sqrt(len(_idx_common))
            _bar_colors = [
                OK_COLOR if abs(v) >= _sig_band else "#AAAAAA" for v in _ccf_vals
            ]
            _fig_ccf = go.Figure()
            _fig_ccf.add_trace(go.Bar(
                x=_lags, y=_ccf_vals, marker_color=_bar_colors, name="CCF",
            ))
            _fig_ccf.add_hline(y=_sig_band, line_dash="dot", line_color=WARN_COLOR,
                                annotation_text=f"±{_sig_band:.2f} (banda 95%)")
            _fig_ccf.add_hline(y=-_sig_band, line_dash="dot", line_color=WARN_COLOR)
            _fig_ccf.update_layout(
                height=320, xaxis_title="Rezago (meses; negativo = serie adicional lidera)",
                yaxis_title="Correlación cruzada", legend=dict(orientation="h", y=-0.3),
                margin=dict(t=20),
                xaxis=dict(tickmode="linear", dtick=1),
            )
            st.plotly_chart(_fig_ccf, use_container_width=True)

            _best_lag_idx = int(np.argmax(np.abs(_ccf_vals)))
            _best_lag = int(_lags[_best_lag_idx])
            _best_ccf = _ccf_vals[_best_lag_idx]
            if _best_lag < 0:
                _lag_interp = f"la serie adicional lidera al gasto oncológico en {abs(_best_lag)} mes(es)"
            elif _best_lag > 0:
                _lag_interp = f"el gasto oncológico lidera a la serie adicional en {_best_lag} mes(es)"
            else:
                _lag_interp = "las dos series están más correlacionadas sin rezago (contemporáneamente)"
            st.markdown(
                f'<div class="interp-box">El rezago de máxima correlación es <b>{_best_lag:+d} meses</b> '
                f'(r={_best_ccf:.3f}), lo que sugiere que <b>{_lag_interp}</b>. '
                f'Las barras en verde superan la banda de significancia al 95% — interpreta con '
                f'cautela si hay pocos meses en común, ya que la banda se estrecha con más observaciones.</div>',
                unsafe_allow_html=True,
            )

            st.divider()

            # -----------------------------------------------------------
            # Sección 4: ITS comparado en los mismos puntos de corte
            # -----------------------------------------------------------
            st.markdown("#### 4. Efecto normativo comparado (ITS en ambas series)")
            st.caption(
                "Aplica el mismo modelo de series temporales interrumpidas sobre cada serie "
                "por separado, usando los puntos de corte normativos activos en el panel lateral. "
                "Permite ver si las normas oncológicas tuvieron un efecto paralelo en la variable adicional."
            )

            _active_breaks_common = {
                k: v for k, v in active_breaks.items()
                if _idx_common.min() < v <= _idx_common.max()
            }

            if not _active_breaks_common:
                st.info("No hay puntos de corte normativos dentro del período de solapamiento de ambas series.")
            elif len(_idx_common) < 12:
                st.info("Necesitas al menos 12 meses de solapamiento para el modelo ITS.")
            else:
                _its_prim = segmented_its(_p, _active_breaks_common, BREAK_LABELS, alpha=alpha)
                _its_sec_x = segmented_its(_s, _active_breaks_common, BREAK_LABELS, alpha=alpha)

                if _its_prim and _its_sec_x and _its_prim.segments and _its_sec_x.segments:
                    _its_rows = []
                    for _seg_p, _seg_s in zip(_its_prim.segments, _its_sec_x.segments):
                        _its_rows.append({
                            "Punto de corte": _seg_p.label,
                            "Salto de nivel — PRINCIPAL (S/mill/mes)": round(_seg_p.level_change, 4),
                            "Signif. nivel (principal)": significance_label(_seg_p.level_p, alpha),
                            f"Salto de nivel — {_sec_lbl_x}": round(_seg_s.level_change, 4),
                            f"Signif. nivel ({_sec_lbl_x})": significance_label(_seg_s.level_p, alpha),
                            "Cambio pendiente — PRINCIPAL": round(_seg_p.slope_change, 4),
                            "Signif. pendiente (principal)": significance_label(_seg_p.slope_p, alpha),
                            f"Cambio pendiente — {_sec_lbl_x}": round(_seg_s.slope_change, 4),
                            f"Signif. pendiente ({_sec_lbl_x})": significance_label(_seg_s.slope_p, alpha),
                            "¿Misma dirección de cambio?": (
                                "Sí" if np.sign(_seg_p.slope_change) == np.sign(_seg_s.slope_change) else "No"
                            ),
                        })
                    _its_df = pd.DataFrame(_its_rows)

                    def _color_its(val):
                        if val in ("Significativo", "Sí"):
                            return f"background-color: {OK_COLOR}33"
                        if val == "Marginal":
                            return f"background-color: {WARN_COLOR}33"
                        if val in ("No significativo", "No"):
                            return f"background-color: {BAD_COLOR}33"
                        return ""

                    _sig_cols = [c for c in _its_df.columns if "Signif." in c or "¿Misma" in c]
                    st.dataframe(
                        _its_df.style.map(_color_its, subset=_sig_cols),
                        hide_index=True, use_container_width=True,
                    )

                    # Gráfico dual ITS
                    _fig_its2 = go.Figure()
                    _fig_its2.add_trace(go.Scatter(
                        x=_p.index, y=_p.values, mode="lines",
                        name="Principal (observado)", line=dict(color=PRIMARY_COLOR, width=2),
                    ))
                    _fig_its2.add_trace(go.Scatter(
                        x=_its_prim.fitted.index, y=_its_prim.fitted.values, mode="lines",
                        name="Principal (ajuste ITS)", line=dict(color=PRIMARY_COLOR, width=2, dash="dash"),
                    ))
                    # serie adicional en eje secundario (escala diferente)
                    _fig_its2.add_trace(go.Scatter(
                        x=_s.index, y=_s.values, mode="lines",
                        name=f"{_sec_lbl_x} (observado)", yaxis="y2",
                        line=dict(color=SECONDARY_COLOR, width=2),
                    ))
                    _fig_its2.add_trace(go.Scatter(
                        x=_its_sec_x.fitted.index, y=_its_sec_x.fitted.values, mode="lines",
                        name=f"{_sec_lbl_x} (ajuste ITS)", yaxis="y2",
                        line=dict(color=SECONDARY_COLOR, width=2, dash="dash"),
                    ))
                    for _k, _bdate in _active_breaks_common.items():
                        _fig_its2.add_vline(x=_bdate, line_dash="dot", line_color="gray", opacity=0.7)
                        _fig_its2.add_annotation(
                            x=_bdate, y=1.04, yref="paper", showarrow=False,
                            text=f"<b>{BREAK_CODES.get(_k, _k)}</b>",
                            font=dict(size=12, color="#555555"),
                        )
                    _fig_its2.update_layout(
                        height=420, hovermode="x unified",
                        yaxis_title="Gasto oncológico (S/)",
                        yaxis2=dict(
                            title=f"{_sec_lbl_x} ({_sec_units_x})",
                            overlaying="y", side="right",
                        ),
                        legend=dict(orientation="h", y=-0.3), margin=dict(t=40),
                    )
                    st.plotly_chart(_fig_its2, use_container_width=True)
                    st.markdown(
                        f'<div class="interp-box">Si ambas series muestran cambios significativos de '
                        f'nivel o pendiente en el mismo punto de corte y en la <b>misma dirección</b>, '
                        f'el efecto normativo podría ser transversal o reflejar un factor externo común. '
                        f'Si solo la serie oncológica muestra el quiebre, refuerza la especificidad del '
                        f'efecto normativo sobre el gasto en cáncer.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("No se pudo ajustar el modelo ITS en alguna de las dos series (muy pocas observaciones en el solapamiento).")

            st.divider()

            # Descarga conjunta
            _joint_df = pd.DataFrame({
                "tiempo": _idx_common,
                "gasto_oncologico_principal_S": _p.values,
                _sec_lbl_x: _s.values,
            })
            st.download_button(
                "⬇️ Descargar tabla cruzada alineada (CSV)",
                _joint_df.to_csv(index=False).encode("utf-8"),
                file_name="analisis_cruzado.csv", mime="text/csv",
            )

st.markdown(
    """<p class="source-note">Fuente: base de consumo línea-de-gasto del INEN, clasificada por código CIE-10
(2021_2025_CONSUMO.xlsx). Los valores son el gasto valorizado reportado al SIS/FISSAL, no la incidencia de
cáncer. Setiembre de 2025 se excluyó por reporte administrativo incompleto al momento de la extracción.</p>
<p class="source-note" style="text-align:center; margin-top:0.6rem;">© Luis A. Orrego Ferreyros, DDS, Econ.,
MCE, MMD, PhD(c), CQRM · Epidemiólogo y Economista de la Salud — INEN</p>""",
    unsafe_allow_html=True,
)

_cite_year = pd.Timestamp.today().year
_cite_url = "https://impactonormativocancer.streamlit.app"

st.markdown(
    f"""<div class="citation-box">
<h4>REFERENCIA WEB (formato AMA)</h4>
<p>Orrego-Ferreyros LA. Observatorio del Financiamiento Oncológico INEN: Impacto Normativo en el Tiempo.
Instituto Nacional de Enfermedades Neoplásicas. Published {_cite_year}. Accessed [Mes Día, Año]. {_cite_url}</p>
<h4>WEB REFERENCE (AMA format)</h4>
<p>Orrego-Ferreyros LA. INEN Oncology Financing Observatory: Regulatory Impact Over Time. National Institute
of Neoplastic Diseases. Published {_cite_year}. Accessed [Month Day, Year]. {_cite_url}</p>
</div>""",
    unsafe_allow_html=True,
)

_ris_content = (
    "TY  - ELEC\n"
    "AU  - Orrego-Ferreyros, Luis Alexander\n"
    "TI  - Observatorio del Financiamiento Oncológico INEN: Impacto Normativo en el Tiempo / "
    "INEN Oncology Financing Observatory: Regulatory Impact Over Time\n"
    f"PY  - {_cite_year}\n"
    "PB  - Instituto Nacional de Enfermedades Neoplásicas\n"
    f"UR  - {_cite_url}\n"
    "ER  - \n"
)
_bib_content = (
    f"@misc{{orregoferreyros{_cite_year}ofo,\n"
    "  author = {Orrego-Ferreyros, Luis Alexander},\n"
    "  title = {Observatorio del Financiamiento Oncol{\\'o}gico INEN: Impacto Normativo en el Tiempo "
    "/ INEN Oncology Financing Observatory: Regulatory Impact Over Time},\n"
    f"  year = {{{_cite_year}}},\n"
    "  publisher = {Instituto Nacional de Enfermedades Neopl{\\'a}sicas},\n"
    f"  url = {{{_cite_url}}}\n"
    "}\n"
)
_endnote_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<xml><records><record>
<ref-type name="Web Page">12</ref-type>
<contributors><authors><author>Orrego-Ferreyros, Luis Alexander</author></authors></contributors>
<titles><title>Observatorio del Financiamiento Oncol\u00f3gico INEN: Impacto Normativo en el Tiempo / INEN Oncology Financing Observatory: Regulatory Impact Over Time</title></titles>
<dates><year>{_cite_year}</year></dates>
<publisher>Instituto Nacional de Enfermedades Neopl\u00e1sicas</publisher>
<urls><related-urls><url>{_cite_url}</url></related-urls></urls>
</record></records></xml>
"""
_dl1, _dl2, _dl3 = st.columns(3)
_dl1.download_button("⬇️ BibTeX (*.bib)", _bib_content, file_name="ofo_inen.bib", mime="text/plain")
_dl2.download_button("⬇️ EndNote XML (*.xml)", _endnote_xml, file_name="ofo_inen.xml", mime="application/xml")
_dl3.download_button("⬇️ RIS (*.ris)", _ris_content, file_name="ofo_inen.ris", mime="application/x-research-info-systems")
