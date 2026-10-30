"""
qa_engine.py
------------
Motor de la sección «Pregunta a los datos» del Observatorio del
Financiamiento Oncológico INEN.

Dos modos:

  - Modo básico por reglas (por defecto): interpreta la pregunta en español
    con reglas léxicas (IAFAS, diagnóstico, tipo de consumo, años, producto,
    alto costo, top N, evolución anual/mensual) y la traduce a una
    agregación de pandas sobre los datos del observatorio.
  - Modo IA (opcional): un modelo de Claude solo (1) traduce la pregunta a la
    misma especificación estructurada y (2) redacta una lectura breve del
    resultado. Las cifras SIEMPRE salen de las agregaciones locales; al
    modelo solo se envían el catálogo de valores posibles y la tabla
    resumida del resultado, sin identificadores de pacientes (la base ya
    viene agregada a mes/año).

Las funciones de este módulo no dependen de Streamlit, para poder
probarlas fuera de la app.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from data_processing import (
    ALL_DIAGS_LABEL,
    IAFAS_LIST,
    NO_ONCO_LABEL,
    ONCO_DIAGS,
    OTRO_INTERNAL,
    OTRO_LABEL,
    TIPO_CONSUMO_LIST,
)

PRIMARY_COLOR = "#2C5F8A"
PALETTE = [PRIMARY_COLOR, "#D9542F", "#2F9E44", "#8850C4", "#C9A13B", "#5C6AC4", "#7D7D7D", "#4AA8C9",
           "#B5487A", "#3E8E7E"]

DIM_LABELS = {"iafas": "IAFAS", "diag": "Diagnóstico", "tipo": "Tipo de consumo", "producto": "Producto"}
DIM_COLS = {"iafas": "iafas", "diag": "diag_group", "tipo": "tipo_consumo", "producto": "producto"}
TIPO_LABELS = {"MEDICAMENTO": "Medicamento", "INSUMO": "Insumo", "PROCEDIMIENTO": "Procedimiento"}

EXAMPLES = [
    "¿Cuál es el gasto total por IAFAS?",
    "Top 10 medicamentos de FISSAL en 2024",
    "Evolución anual del gasto en cáncer de mama",
    "Gasto por diagnóstico en 2024",
    "Evolución mensual por IAFAS en 2025",
    "Medicamentos de alto costo por IAFAS",
    "Top 10 procedimientos con más gasto en 2025",
]


# ---------------------------------------------------------------------------
# Especificación de consulta
# ---------------------------------------------------------------------------

@dataclass
class QASpec:
    intent: str = "total"                 # total | breakdown | trend | unsupported
    dims: list[str] = field(default_factory=list)   # subconjunto de iafas, diag, tipo, producto
    time: str | None = None               # None | year | month
    top_n: int | None = None
    order: str = "desc"                   # desc | asc
    metric: str = "gasto"                 # gasto | cantidad
    iafas: list[str] = field(default_factory=list)
    diags: list[str] = field(default_factory=list)   # valores internos de diag_group
    tipos: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    products: list[str] = field(default_factory=list)  # términos de búsqueda en el nombre del producto
    alto_costo: bool = False
    unsupported_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QAResult:
    question: str
    spec: QASpec
    title: str
    summary: str
    table: pd.DataFrame | None = None
    fig: go.Figure | None = None
    value: float | None = None
    notes: list[str] = field(default_factory=list)
    scope: str = ""
    ok: bool = True


# ---------------------------------------------------------------------------
# Utilidades de texto
# ---------------------------------------------------------------------------

def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s.lower()).strip()


_DIAG_PATTERNS = [
    (r"cuello uterino|cervix|cervical|cervico ?uterino", "Cuello uterino"),
    (r"\bmamas?\b|\bseno\b", "Mama"),
    (r"\bcolon\b|colorrectal|\bcolorectal", "Colon"),
    (r"estomago|gastrico", "Estómago"),
    (r"prostata", "Próstata"),
    (r"leucemias?", "Leucemias"),
    (r"linfomas?", "Linfomas"),
    (r"otros? canceres|otros? cancer|otras? neoplasias|\botros diagnosticos", OTRO_INTERNAL),
    (r"no oncologic|comparador|comorbilidad", NO_ONCO_LABEL),
]
_TIPO_PATTERNS = [
    (r"medicamentos?|farmacos?|drogas?", "MEDICAMENTO"),
    (r"insumos?|dispositivos?", "INSUMO"),
    (r"procedimientos?", "PROCEDIMIENTO"),
]
_UNSUPPORTED = [
    (r"paciente|atendid|atencion(es)?\b|personas|beneficiari",
     "La base del observatorio está agregada a gasto mensual/anual: no contiene número de pacientes ni "
     "de atenciones, por lo que no se puede calcular gasto por paciente."),
    (r"saludpol|oncosalud|essalud|ff\.? ?aa|fuerzas armadas|\bpnp\b|privad",
     "Este observatorio cubre solo el gasto reportado al SIS y a FISSAL. Para SaludPol, FF.AA. u Oncosalud "
     "usa el Observatorio de Consumo de IAFAS."),
    (r"incidencia|mortalidad|\bcasos\b|\bmuertes\b",
     "Este observatorio mide gasto valorizado, no incidencia ni mortalidad (ver el dashboard «Cáncer en el "
     "Tiempo — Perú»)."),
    (r"\bambito\b|ambulatori|hospitaliz|emergencia",
     "La base no distingue ámbito de atención (ambulatorio, hospitalización, emergencia)."),
]

_STOP = set("""
a al algo ante anual anuales ano anos aumento cada cambio como con contra cual cuales cuanto cuantos cuanta
de del desde donde dos e el ella ellos en entre era es esa ese esta este esto evolucion explica explican
fue fueron gasto gastos gastado gasta gastan ha han hasta hay la las le lo los mas me mayor mayores menor menores menos mes meses mensual
mi mis mucho muy ni no nos o os otra otro para pero poco por porque que quien se segun ser si sin sobre su
sus tambien tan tanto te tendencia tiene tienen todo todos total totales tu un una uno unos y ya
cancer canceres oncologico oncologica oncologicos diagnostico diagnosticos iafas sis fissal financiador
tipo tipos consumo consumos medicamento medicamentos insumo insumos procedimiento procedimientos producto
productos item items top principales ranking lista listado costo costos alto altos elevado caro caros
monto montos soles valor valores cantidad cantidades unidades participacion porcentaje proporcion
compara comparar comparacion versus serie tiempo periodo ultimo ultimos primer primeros mostrar muestra
dame dime quiero ver grafico tabla crecimiento crecio variacion aumento disminucion bajo subio cayo concentra
concentran representa representan durante respecto frente vs cuanto hubo fueron registrado registrada reportado
reportada inen datos dato pregunta consulta
""".split())


def _years_in(q: str, years_avail: list[int]) -> list[int]:
    yrs = sorted({int(y) for y in re.findall(r"\b(20\d{2})\b", q)})
    rng = re.search(r"(?:entre|de|desde)\s+(20\d{2})\s+(?:y|a|al|hasta|-)\s+(20\d{2})", q) or \
        re.search(r"\b(20\d{2})\s*[-–]\s*(20\d{2})\b", q)
    if rng:
        a, b = sorted((int(rng.group(1)), int(rng.group(2))))
        yrs = list(range(a, b + 1))
    if not yrs and re.search(r"ultimo ano|este ano|ano actual|ano en curso", q):
        yrs = [max(years_avail)]
    if not yrs and re.search(r"ano pasado|ano anterior", q):
        yrs = [max(years_avail) - 1]
    return [y for y in yrs if y in years_avail] or ([] if not yrs else [-1])  # -1 = año fuera de rango


# ---------------------------------------------------------------------------
# Índice de productos (para reconocer nombres de medicamentos/procedimientos)
# ---------------------------------------------------------------------------

def build_product_index(product_df: pd.DataFrame) -> pd.DataFrame:
    idx = product_df[["DESC_CONSUMO"]].drop_duplicates().copy()
    idx["norm"] = idx["DESC_CONSUMO"].map(norm_text)
    idx["words"] = idx["norm"].str.findall(r"[a-z0-9]+")
    return idx.reset_index(drop=True)


def match_products(term: str, pindex: pd.DataFrame) -> list[str]:
    t = norm_text(term)
    if not t:
        return []
    if " " in t:
        mask = pindex["norm"].str.contains(re.escape(t), regex=True)
    elif len(t) >= 5:
        mask = pindex["words"].map(lambda ws: any(w.startswith(t) for w in ws))
    else:
        mask = pindex["words"].map(lambda ws: t in ws)
    return pindex.loc[mask, "DESC_CONSUMO"].tolist()


# ---------------------------------------------------------------------------
# Interpretación por reglas
# ---------------------------------------------------------------------------

def parse_question(question: str, years_avail: list[int], pindex: pd.DataFrame) -> QASpec:
    q = norm_text(question)
    spec = QASpec()

    for pat, reason in _UNSUPPORTED:
        if re.search(pat, q):
            spec.intent = "unsupported"
            spec.unsupported_reason = reason
            return spec

    consumed: set[str] = set()

    def _eat(pattern: str):
        for m in re.finditer(pattern, q):
            consumed.update(re.findall(r"[a-z0-9]+", m.group(0)))

    # IAFAS
    if re.search(r"\bfissal\b", q):
        spec.iafas.append("FISSAL")
    if re.search(r"\bsis\b", q):
        spec.iafas.append("SIS")
    # Diagnósticos
    for pat, val in _DIAG_PATTERNS:
        if re.search(pat, q):
            spec.diags.append(val)
            _eat(pat)
    if re.search(r"(todos los|todo el|total de) cancer|todos los diagnosticos", q):
        spec.diags = []
    # Tipo de consumo
    for pat, val in _TIPO_PATTERNS:
        if re.search(r"\b(" + pat + r")\b", q):
            spec.tipos.append(val)
    # Alto costo
    if re.search(r"alto costo|alto precio|no pnume", q):
        spec.alto_costo = True
        spec.tipos = ["MEDICAMENTO"]
    # Años
    spec.years = _years_in(q, years_avail)

    # Métrica
    if re.search(r"\b(cantidad|cantidades|unidades|volumen)\b", q):
        spec.metric = "cantidad"

    # Top N / orden
    m = re.search(r"\btop\s*(\d{1,3})\b", q) or re.search(r"\b(\d{1,3})\s+(?:principales|primeros|mayores|menores|mas|productos|medicamentos|insumos|procedimientos|diagnosticos)", q)
    if m:
        spec.top_n = max(1, min(int(m.group(1)), 100))
    elif re.search(r"\btop\b|principales|ranking|mas costos|que mas|cuales? (son )?(los|las)", q):
        spec.top_n = 10
    if re.search(r"\b(menor|menores|menos)\b", q) and not re.search(r"al menos|menos de", q):
        spec.order = "asc"

    # Dimensiones de desagregación
    dims: list[str] = []
    if re.search(r"(por|segun|cada|entre)\s+(las\s+|el\s+|la\s+)?(iafas|financiador)", q) or len(set(spec.iafas)) > 1 \
            or re.search(r"\bvs\.?\b|versus|compar", q) and len(set(spec.iafas)) > 1:
        dims.append("iafas")
    if re.search(r"(por|segun|cada|entre)\s+(los\s+|el\s+|tipo de\s+)?(diagnosticos?|cancer(es)?|neoplasias?|tipos? de cancer)", q) \
            or re.search(r"\bdiagnosticos\b|\bcanceres\b", q) or len(set(spec.diags)) > 1:
        dims.append("diag")
    if re.search(r"tipos? de consumo|(por|segun|cada)\s+tipo\b(?! de cancer)", q) or len(set(spec.tipos)) > 1:
        dims.append("tipo")
    plural_tipo = re.search(r"\b(medicamentos|farmacos|insumos|dispositivos|procedimientos)\b", q)
    if re.search(r"(por|cada|segun)\s+(medicamento|producto|item|insumo|procedimiento|farmaco)\b", q) \
            or re.search(r"\b(productos|items)\b", q) \
            or (plural_tipo and (spec.top_n or re.search(r"\b(que|cuales|explican|concentran|lista|listado|mayor|mas)\b", q))) \
            or spec.alto_costo and not dims:
        dims.append("producto")
    if spec.top_n and not dims:
        dims.append("producto")

    # Tiempo
    if re.search(r"mensual|por mes|cada mes|mes a mes", q):
        spec.time = "month"
    elif re.search(r"anual|por ano|cada ano|ano a ano|evolucion|tendencia|en el tiempo|serie|historic|crec|variacion|interanual", q):
        spec.time = "year"
    if spec.time == "year" and len(spec.years) == 1 and not re.search(r"mensual|evolucion", q):
        spec.time = None
    if spec.time is None and len(spec.years) > 1 and not dims:
        spec.time = "year"

    # Nombres de productos (p. ej. «pembrolizumab», «tomografía de tórax»)
    tokens = [t for t in re.findall(r"[a-z][a-z0-9]{3,}", q) if t not in _STOP and t not in consumed]
    for t in tokens:
        hits = match_products(t, pindex)
        if 0 < len(hits) <= 150:
            spec.products.append(t)
    if spec.products and not dims and not spec.time:
        spec.time = "year"
    # si se pidió un producto puntual, el tipo de consumo genérico no debe excluirlo
    if spec.products and not plural_tipo:
        spec.tipos = [] if not spec.alto_costo else spec.tipos

    # Si la dimensión está también como filtro único, no tiene sentido desagregar
    if "iafas" in dims and len(set(spec.iafas)) == 1:
        dims.remove("iafas")
    if "tipo" in dims and len(set(spec.tipos)) == 1 and "producto" not in dims:
        dims.remove("tipo")
    if "diag" in dims and len(set(spec.diags)) == 1:
        dims.remove("diag")

    spec.dims = list(dict.fromkeys(dims))[:2]
    spec.iafas = list(dict.fromkeys(spec.iafas))
    spec.diags = list(dict.fromkeys(spec.diags))
    spec.tipos = list(dict.fromkeys(spec.tipos))
    spec.intent = "trend" if spec.time else ("breakdown" if spec.dims else "total")
    return spec


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------

def fmt_soles(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/d"
    ax = abs(x)
    if ax >= 1e6:
        return f"S/ {x / 1e6:,.2f} millones"
    return f"S/ {x:,.0f}"


def fmt_qty(x: float) -> str:
    return f"{x:,.0f} unidades"


def _diag_display(v: str) -> str:
    return OTRO_LABEL if v == OTRO_INTERNAL else v


def _resolve_panel_diags(diags: list[str]) -> list[str]:
    if not diags or ALL_DIAGS_LABEL in diags:
        return []
    return [OTRO_INTERNAL if d == OTRO_LABEL else d for d in diags]


def _scope_text(eff: dict, source: str, year_lo: int, year_hi: int, date_lo, date_hi) -> str:
    parts = []
    parts.append("IAFAS: " + ("/".join(eff["iafas"]) if eff["iafas"] else "SIS y FISSAL"))
    parts.append("Diagnóstico: " + (", ".join(_diag_display(d) for d in eff["diags"]) if eff["diags"] else "todos"))
    parts.append("Tipo: " + (", ".join(TIPO_LABELS.get(t, t) for t in eff["tipos"]) if eff["tipos"] else "todos"))
    if source == "monthly" and date_lo is not None:
        parts.append(f"Período: {date_lo:%b-%Y} a {date_hi:%b-%Y}")
    else:
        parts.append(f"Período: {year_lo}" + (f" a {year_hi}" if year_hi != year_lo else ""))
    return " · ".join(parts)


def run_query(
    question: str,
    spec: QASpec,
    monthly_df: pd.DataFrame,
    product_df: pd.DataFrame,
    listado_df: pd.DataFrame,
    pindex: pd.DataFrame,
    panel: dict | None = None,
) -> QAResult:
    """Ejecuta la especificación sobre los datos. `panel` (opcional) trae los
    filtros activos de la barra lateral: diags (etiquetas de interfaz),
    iafas, tipos, date_lo, date_hi. Si la pregunta menciona explícitamente
    una dimensión, esa mención prevalece sobre el panel para esa dimensión."""
    notes: list[str] = []

    if spec.intent == "unsupported":
        return QAResult(question, spec, "No disponible en esta base", spec.unsupported_reason, ok=False)
    if spec.years == [-1]:
        yrs = sorted(product_df["ANNIO"].unique())
        return QAResult(question, spec, "Año fuera de rango",
                        f"La base cubre {yrs[0]} a {yrs[-1]} (2025 hasta agosto).", ok=False)

    # ---- Filtros efectivos (pregunta > panel)
    panel = panel or {}
    eff = {
        "iafas": spec.iafas or (list(panel.get("iafas") or []) if set(panel.get("iafas") or []) != set(IAFAS_LIST) else []),
        "diags": spec.diags or _resolve_panel_diags(panel.get("diags") or []),
        "tipos": spec.tipos or (list(panel.get("tipos") or []) if set(panel.get("tipos") or []) != set(TIPO_CONSUMO_LIST) else []),
    }
    if panel:
        _full = {"iafas": set(IAFAS_LIST), "tipos": set(TIPO_CONSUMO_LIST)}
        overridden = []
        for k, lbl in (("iafas", "IAFAS"), ("diags", "diagnóstico"), ("tipos", "tipo de consumo")):
            pv = panel.get(k) or []
            restricted = bool(_resolve_panel_diags(pv)) if k == "diags" else (pv and set(pv) != _full[k])
            if getattr(spec, k) and restricted:
                overridden.append(lbl)
        if overridden:
            notes.append("La pregunta fija " + ", ".join(overridden) + "; para esa dimensión se ignora el panel.")

    use_product = bool("producto" in spec.dims or spec.products or spec.alto_costo or spec.metric == "cantidad")
    source = "product" if use_product else "monthly"

    if source == "monthly":
        d = monthly_df.copy()
        d["year"] = d["tiempo"].dt.year
        d["valor"] = d["total_neto"]
    else:
        d = product_df.rename(columns={"ANNIO": "year", "DESC_CONSUMO": "producto"}).copy()
        d["valor"] = d["total_cantidad"] if spec.metric == "cantidad" else d["total_neto"]
        if spec.time == "month":
            spec.time = "year"
            notes.append("El detalle por producto es anual; se muestra la evolución por año.")

    if eff["iafas"]:
        d = d[d["iafas"].isin(eff["iafas"])]
    if eff["diags"]:
        d = d[d["diag_group"].isin(eff["diags"])]
    if eff["tipos"]:
        d = d[d["tipo_consumo"].isin(eff["tipos"])]

    date_lo = date_hi = None
    if spec.years:
        d = d[d["year"].isin(spec.years)]
        if source == "monthly":
            date_lo, date_hi = d["tiempo"].min(), d["tiempo"].max()
    elif panel.get("date_lo") is not None:
        p_lo, p_hi = pd.Timestamp(panel["date_lo"]), pd.Timestamp(panel["date_hi"])
        if source == "monthly":
            d = d[(d["tiempo"] >= p_lo) & (d["tiempo"] <= p_hi)]
            date_lo, date_hi = p_lo, p_hi
        else:
            d = d[d["year"].between(p_lo.year, p_hi.year)]
            if p_lo.month != 1 or p_hi.month != 12:
                notes.append("El detalle por producto es anual: el rango de meses del panel se aplica por año completo.")
    elif source == "monthly" and len(d):
        date_lo, date_hi = d["tiempo"].min(), d["tiempo"].max()

    if spec.alto_costo:
        codes = set(listado_df["cod_sismed"].astype(str).str.strip())
        d = d[d["cod_sismed"].astype(str).str.strip().isin(codes)]
        notes.append("Alto costo = medicamentos del listado institucional «NO PNUME - Alto costo», cruzados por código SISMED.")

    if spec.products:
        names: set[str] = set()
        for t in spec.products:
            names.update(match_products(t, pindex))
        d = d[d["producto"].isin(names)]
        if names:
            shown = sorted(names)
            notes.append("Productos que coinciden con «" + ", ".join(spec.products) + "»: "
                         + "; ".join(shown[:6]) + (f" y {len(shown) - 6} más" if len(shown) > 6 else "") + ".")

    year_lo = int(d["year"].min()) if len(d) else 0
    year_hi = int(d["year"].max()) if len(d) else 0
    scope = _scope_text(eff, source, year_lo, year_hi, date_lo, date_hi)
    if len(d) == 0 or d["valor"].sum() == 0:
        return QAResult(question, spec, "Sin datos", "No hay registros para esa combinación de filtros.",
                        notes=notes, scope=scope, ok=False)
    if 2025 in d["year"].unique():
        notes.append("2025 incluye solo enero a agosto (setiembre excluido por reporte incompleto).")

    metric_lbl = "Cantidad" if spec.metric == "cantidad" else "Gasto (S/)"
    fmt = fmt_qty if spec.metric == "cantidad" else fmt_soles
    total = float(d["valor"].sum())

    # ---- Agregación
    cat_cols = [DIM_COLS[k] for k in spec.dims if k in DIM_COLS and DIM_COLS[k] in d.columns]
    time_col = None
    if spec.time == "year":
        time_col = "year"
    elif spec.time == "month":
        time_col = "tiempo"

    # Top N sobre categorías (se calcula sobre el total del período)
    asc = spec.order == "asc"
    if cat_cols:
        rank = d.groupby(cat_cols)["valor"].sum()
        rank = rank[rank != 0].sort_values(ascending=asc)
        n = spec.top_n or (15 if "producto" in spec.dims else None)
        if n and len(rank) > n:
            keep = rank.head(n).index
            d = d.set_index(cat_cols).loc[keep].reset_index()
            notes.append(f"Se muestran {n} de {len(rank)} categorías ({'menor' if asc else 'mayor'} valor).")

    group = ([time_col] if time_col else []) + cat_cols
    if not group:
        by_year = d.groupby("year")["valor"].sum().reset_index()
        by_year.columns = ["Año", metric_lbl]
        summary = f"El {'consumo total' if spec.metric == 'cantidad' else 'gasto total'} en el ámbito consultado es **{fmt(total)}**."
        if len(by_year) > 1:
            ylead = by_year.loc[by_year[metric_lbl].idxmax()]
            summary += f" El año de mayor valor fue {int(ylead['Año'])} ({fmt(ylead[metric_lbl])})."
        fig = _bar_fig(by_year.assign(Año=by_year["Año"].astype(str)), "Año", metric_lbl, horizontal=False)
        return QAResult(question, spec, "Total", summary, table=by_year, fig=fig, value=total,
                        notes=notes, scope=scope)

    t = d.groupby(group, as_index=False)["valor"].sum()
    rename = {"valor": metric_lbl, "year": "Año", "tiempo": "Mes"}
    for k in spec.dims:
        rename[DIM_COLS[k]] = DIM_LABELS[k]
    if "diag_group" in t.columns:
        t["diag_group"] = t["diag_group"].map(_diag_display)
    if "tipo_consumo" in t.columns:
        t["tipo_consumo"] = t["tipo_consumo"].map(lambda v: TIPO_LABELS.get(v, v))
    t = t.rename(columns=rename)
    cat_lbls = [DIM_LABELS[k] for k in spec.dims]
    time_lbl = rename.get(time_col) if time_col else None

    if time_lbl is None:
        # Desagregación categórica
        t = t.sort_values(metric_lbl, ascending=asc).reset_index(drop=True)
        t["% del total"] = t[metric_lbl] / total * 100 if total else np.nan
        lead = t.iloc[0]
        label_of = lambda r: " · ".join(_short(r[c], 70) for c in cat_lbls)
        summary = (f"De un total de **{fmt(total)}**, **{label_of(lead)}** "
                   f"{'registra el menor valor' if asc else 'concentra el mayor valor'}: {fmt(lead[metric_lbl])} "
                   f"({lead['% del total']:.1f}%).")
        if len(t) > 1 and not asc:
            seg = [f"{label_of(r)} ({fmt(r[metric_lbl])}, {r['% del total']:.1f}%)" for _, r in t.iloc[1:3].iterrows()]
            summary += " Le sigue(n) " + " y ".join(seg) + "."
            if len(t) >= 3:
                top3 = t["% del total"].head(3).sum()
                summary += f" Los tres primeros suman {top3:.1f}% del total consultado."
        if len(cat_lbls) == 1:
            fig = _bar_fig(t, cat_lbls[0], metric_lbl, horizontal=True)
        else:
            fig = _stacked_fig(t, cat_lbls[0], cat_lbls[1], metric_lbl)
        title = "Distribución por " + " y ".join(c.lower() for c in cat_lbls)
        return QAResult(question, spec, title, summary, table=t, fig=fig, value=total, notes=notes, scope=scope)

    # Serie temporal (con o sin desagregación)
    t = t.sort_values([time_lbl] + cat_lbls).reset_index(drop=True)
    if not cat_lbls:
        s = t.set_index(time_lbl)[metric_lbl]
        if time_lbl == "Año":
            t["Var. % interanual"] = s.pct_change().values * 100
        first, last = s.iloc[0], s.iloc[-1]
        p_first = _period(s.index[0], time_lbl)
        p_last = _period(s.index[-1], time_lbl)
        var = (last / first - 1) * 100 if first else np.nan
        summary = (f"Pasó de **{fmt(first)}** en {p_first} a **{fmt(last)}** en {p_last}"
                   + (f" ({var:+.1f}%)." if not np.isnan(var) else "."))
        summary += f" Máximo en {_period(s.idxmax(), time_lbl)} ({fmt(s.max())}); total del período {fmt(total)}."
        if time_lbl == "Año" and 2025 in s.index and len(s) > 1:
            summary += " Ojo: 2025 es un año parcial (8 meses)."
        fig = _line_fig(t, time_lbl, metric_lbl, None)
    else:
        key = cat_lbls[0] if len(cat_lbls) == 1 else None
        if key is None:
            t["Serie"] = t[cat_lbls[0]].astype(str) + " · " + t[cat_lbls[1]].astype(str)
            key = "Serie"
        t["% del período"] = t[metric_lbl] / t.groupby(time_lbl)[metric_lbl].transform("sum") * 100
        last_p = t[time_lbl].max()
        first_p = t[time_lbl].min()
        lastrows = t[t[time_lbl] == last_p].sort_values(metric_lbl, ascending=False)
        lead = lastrows.iloc[0]
        summary = (f"En {_period(last_p, time_lbl)} lidera **{_short(lead[key], 70)}** con {fmt(lead[metric_lbl])} "
                   f"({lead['% del período']:.1f}% del período).")
        if time_lbl == "Año":
            chg = []
            for g, gd in t.groupby(key):
                gd = gd.set_index(time_lbl)[metric_lbl]
                if first_p in gd.index and last_p in gd.index and gd[first_p]:
                    chg.append((g, (gd[last_p] / gd[first_p] - 1) * 100))
            if chg and last_p != first_p:
                chg.sort(key=lambda x: -abs(x[1]))
                summary += f" Cambio {first_p}→{last_p}: " + "; ".join(f"{_short(g, 50)} {v:+.1f}%" for g, v in chg[:4]) + "."
                if last_p == 2025:
                    summary += " (2025 es parcial: 8 meses.)"
        fig = _line_fig(t, time_lbl, metric_lbl, key)
    title = ("Evolución " + ("anual" if time_lbl == "Año" else "mensual")
             + ((" por " + " y ".join(c.lower() for c in cat_lbls)) if cat_lbls else ""))
    if time_lbl == "Mes":
        t["Mes"] = pd.to_datetime(t["Mes"]).dt.strftime("%Y-%m")
    return QAResult(question, spec, title, summary, table=t, fig=fig, value=total, notes=notes, scope=scope)


def _period(v, time_lbl: str) -> str:
    if time_lbl == "Mes":
        return f"{pd.Timestamp(v):%b-%Y}"
    return str(int(v))


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------

def _short(s: str, n: int = 48) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor="white",
                      legend=dict(orientation="h", y=-0.18), font=dict(size=12))
    fig.update_xaxes(gridcolor="#eef1f5")
    fig.update_yaxes(gridcolor="#eef1f5")
    return fig


def _bar_fig(t: pd.DataFrame, cat: str, val: str, horizontal: bool) -> go.Figure:
    if horizontal:
        tt = t.iloc[::-1]
        fig = go.Figure(go.Bar(x=tt[val], y=tt[cat].map(_short), orientation="h", marker_color=PRIMARY_COLOR,
                               hovertext=tt[cat], hovertemplate="%{hovertext}<br>%{x:,.0f}<extra></extra>"))
        return _layout(fig, max(260, 30 * len(tt) + 60))
    fig = go.Figure(go.Bar(x=t[cat], y=t[val], marker_color=PRIMARY_COLOR,
                           hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>"))
    return _layout(fig, 320)


def _stacked_fig(t: pd.DataFrame, cat: str, color: str, val: str) -> go.Figure:
    order = t.groupby(cat)[val].sum().sort_values().index
    fig = go.Figure()
    for i, (g, gd) in enumerate(t.groupby(color)):
        gd = gd.set_index(cat).reindex(order)
        fig.add_bar(y=[_short(x) for x in order], x=gd[val], name=str(g), orientation="h",
                    marker_color=PALETTE[i % len(PALETTE)], hovertemplate="%{y}<br>%{x:,.0f}<extra>" + str(g) + "</extra>")
    fig.update_layout(barmode="stack")
    return _layout(fig, max(280, 30 * len(order) + 90))


def _line_fig(t: pd.DataFrame, x: str, val: str, key: str | None) -> go.Figure:
    fig = go.Figure()
    if key is None:
        fig.add_scatter(x=t[x], y=t[val], mode="lines+markers", line=dict(color=PRIMARY_COLOR, width=2.5),
                        name=val, hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>")
    else:
        for i, (g, gd) in enumerate(t.groupby(key, sort=False)):
            fig.add_scatter(x=gd[x], y=gd[val], mode="lines+markers", name=_short(g, 40),
                            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                            hovertemplate="%{x}<br>%{y:,.0f}<extra>" + _short(g, 40) + "</extra>")
    if x == "Año":
        fig.update_xaxes(dtick=1)
    return _layout(fig, 360)


# ---------------------------------------------------------------------------
# Modo IA (opcional)
# ---------------------------------------------------------------------------

AI_MODELS = {
    "Claude Haiku 5.5 (rápido)": "claude-haiku-5-5",
    "Claude Sonnet 5.5 (equilibrado)": "claude-sonnet-5-5",
    "Claude Opus 5.5 (máxima calidad)": "claude-opus-5-5",
}


def _client(api_key: str):
    try:
        import anthropic  # import diferido: el modo básico no lo necesita
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Falta el paquete «anthropic» (pip install anthropic).") from e
    return anthropic.Anthropic(api_key=api_key)


def _text_of(resp) -> str:
    return "".join(getattr(b, "text", "") for b in resp.content).strip()


def ai_parse(question: str, api_key: str, model: str, years_avail: list[int]) -> QASpec:
    """Pide al modelo que traduzca la pregunta a la especificación. Solo se
    envía la pregunta y el catálogo de valores válidos (sin datos)."""
    catalog = {
        "intent": ["total", "breakdown", "trend", "unsupported"],
        "dims (máx. 2)": ["iafas", "diag", "tipo", "producto"],
        "time": [None, "year", "month"],
        "metric": ["gasto", "cantidad"],
        "order": ["desc", "asc"],
        "iafas": IAFAS_LIST,
        "diags": ONCO_DIAGS + [OTRO_INTERNAL, NO_ONCO_LABEL],
        "tipos": TIPO_CONSUMO_LIST,
        "years": years_avail,
    }
    system = (
        "Traduces preguntas en español sobre el gasto oncológico del INEN reportado a SIS/FISSAL a una "
        "especificación JSON. Datos disponibles: gasto mensual 2021-01 a 2025-08 por IAFAS, diagnóstico y tipo de "
        "consumo; y gasto/cantidad anual por producto (nombre del medicamento, insumo o procedimiento). No hay "
        "pacientes, atenciones, ámbito de atención ni otras IAFAS. Responde SOLO con un objeto JSON con las claves: "
        "intent, dims, time, top_n, order, metric, iafas, diags, tipos, years, products, alto_costo, "
        "unsupported_reason. 'products' es una lista de términos de búsqueda del nombre del producto (p. ej. "
        "'pembrolizumab'), vacía si no se pide un producto puntual. 'alto_costo' es true si se pregunta por "
        "medicamentos de alto costo. Si la pregunta no puede responderse con estos datos usa intent='unsupported' y "
        "explica brevemente por qué en unsupported_reason.\nValores válidos: " + json.dumps(catalog, ensure_ascii=False)
    )
    resp = _client(api_key).messages.create(
        model=model, max_tokens=500, system=system,
        messages=[{"role": "user", "content": question}],
    )
    raw = _text_of(resp)
    m = re.search(r"\{.*\}", raw, re.S)
    data = json.loads(m.group(0) if m else raw)
    return _spec_from_dict(data, years_avail)


def _spec_from_dict(data: dict, years_avail: list[int]) -> QASpec:
    s = QASpec()
    s.intent = data.get("intent") if data.get("intent") in ("total", "breakdown", "trend", "unsupported") else "total"
    s.dims = [x for x in (data.get("dims") or []) if x in DIM_COLS][:2]
    s.time = data.get("time") if data.get("time") in ("year", "month") else None
    try:
        s.top_n = int(data["top_n"]) if data.get("top_n") else None
    except (TypeError, ValueError):
        s.top_n = None
    s.order = "asc" if data.get("order") == "asc" else "desc"
    s.metric = "cantidad" if data.get("metric") == "cantidad" else "gasto"
    s.iafas = [x for x in (data.get("iafas") or []) if x in IAFAS_LIST]
    valid_d = set(ONCO_DIAGS + [OTRO_INTERNAL, NO_ONCO_LABEL])
    s.diags = [OTRO_INTERNAL if x == OTRO_LABEL else x for x in (data.get("diags") or [])]
    s.diags = [x for x in s.diags if x in valid_d]
    s.tipos = [x for x in (data.get("tipos") or []) if x in TIPO_CONSUMO_LIST]
    yrs = []
    for y in data.get("years") or []:
        try:
            yrs.append(int(y))
        except (TypeError, ValueError):
            pass
    s.years = [y for y in yrs if y in years_avail] or ([-1] if yrs else [])
    s.products = [str(p) for p in (data.get("products") or []) if str(p).strip()][:5]
    s.alto_costo = bool(data.get("alto_costo"))
    if s.alto_costo:
        s.tipos = ["MEDICAMENTO"]
    s.unsupported_reason = str(data.get("unsupported_reason") or "")
    if s.intent != "unsupported":
        s.intent = "trend" if s.time else ("breakdown" if s.dims else "total")
    return s


def ai_narrate(res: QAResult, api_key: str, model: str) -> str:
    """Redacta una lectura breve a partir de la tabla resumida (máx. 40 filas)."""
    tbl = res.table.head(40) if res.table is not None else pd.DataFrame()
    payload = tbl.round(2).to_csv(index=False)
    system = (
        "Eres analista de economía de la salud del INEN (Perú). Redacta en español una lectura de 3 a 5 oraciones, "
        "en prosa fluida, sin viñetas, sin guiones largos y sin usar dos puntos salvo que sea indispensable. Usa "
        "EXCLUSIVAMENTE las cifras de la tabla y del ámbito entregados; no inventes ni estimes valores. Expresa los "
        "montos en soles (S/). Menciona limitaciones si las notas lo indican (p. ej. 2025 parcial)."
    )
    user = (f"Pregunta: {res.question}\nÁmbito: {res.scope}\nTotal del ámbito: {res.value}\n"
            f"Notas: {' '.join(res.notes)}\nTabla resumida (CSV):\n{payload}")
    resp = _client(api_key).messages.create(
        model=model, max_tokens=600, system=system, messages=[{"role": "user", "content": user}],
    )
    return _text_of(resp)
