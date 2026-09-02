"""
data_processing.py
-------------------
Carga y preparación de datos para el dashboard "Gasto Oncológico INEN —
SIS/FISSAL". Dos fuentes, ambas ya agregadas a partir de la base de
consumo línea-de-gasto del INEN (2021_2025_CONSUMO.xlsx) para mantener
el repositorio liviano:

  - data/monthly_diag_iafas_tipo.csv
        Serie MENSUAL: tiempo x diag_group x iafas x tipo_consumo -> total_neto.
        Base para las pestañas de serie temporal, punto de corte normativo
        y tendencia (Mann-Kendall).
  - data/product_annual_onco.csv
        Serie ANUAL a nivel de producto/ítem (ANNIO x DESC_CONSUMO x
        diag_group x iafas x tipo_consumo -> total_neto, total_cantidad),
        restringida a los 7 diagnósticos oncológicos que financia FISSAL.
        Base para la clasificación ABC y el análisis de alto costo (9 UIT).
  - data/norms.csv
        Cronología normativa y los 4 puntos de corte analíticos (fecha de
        publicación + tiempo de implementación + 3 meses de maduración;
        normas con vigencias efectivas separadas por <3 meses se agrupan
        en un único punto de corte). Editable sin tocar código: agregar
        una fila aquí y se refleja en toda la app.

Diagnósticos cubiertos (los 7 que FISSAL financia de forma explícita) +
"Otro" (resto de diagnósticos oncológicos atendidos por el INEN, fuera
del mandato explícito de FISSAL) + "Total INEN" (agregado de todos).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"
MONTHLY_FILE = DATA_DIR / "monthly_diag_iafas_tipo.csv"
PRODUCT_FILE = DATA_DIR / "product_annual_onco.csv"
NORMS_FILE = DATA_DIR / "norms.csv"

ONCO_DIAGS = ["Cuello uterino", "Mama", "Colon", "Estómago", "Próstata", "Leucemias", "Linfomas"]
ALL_DIAGS_LABEL = "Todos los cánceres"
OTRO_INTERNAL = "Otro"  # valor tal como aparece en diag_group dentro de los CSV
OTRO_LABEL = "Otros cánceres"  # etiqueta mostrada en la interfaz
NO_ONCO_LABEL = "No oncológico (comparador)"
"""Diagnósticos NO oncológicos atendidos por el INEN (comorbilidades: infecciones,
enfermedades respiratorias/digestivas/renales/endocrinas, trastornos de sangre no
neoplásicos, etc. — capítulos CIE-10 A,B,E,F,G,I,J,K,L,M,N,O,Q,S,T,U y D50-D89).
Sirve como grupo de comparación/control: si su gasto muestra los mismos quiebres
que el gasto oncológico, sugiere un efecto presupuestal general del INEN y no uno
específico de la normativa oncológica. Se excluyen deliberadamente: otros tipos de
cáncer no priorizados (otros C), tumores in situ/benignos (D00-D48), códigos Z de
sesiones de quimio/radioterapia o control post-tratamiento oncológico, y códigos R
de síntomas (dolor, náuseas) — todos quedan en 'Otro' por ser oncológicos o
ambiguos en el contexto de un hospital especializado en cáncer."""
IAFAS_LIST = ["SIS", "FISSAL"]
TIPO_CONSUMO_LIST = ["MEDICAMENTO", "INSUMO", "PROCEDIMIENTO"]

# Valor de la UIT (Unidad Impositiva Tributaria) por año, para el análisis
# de medicamentos de alto costo (DIGEMID: umbral = 9 UIT/paciente-año).
UIT_BY_YEAR = {2021: 4400, 2022: 4600, 2023: 4950, 2024: 5150, 2025: 5350}


@st.cache_data(show_spinner="Cargando datos mensuales del INEN...")
def load_monthly(path: Path = MONTHLY_FILE) -> pd.DataFrame:
    """Serie mensual tiempo x diag_group x iafas x tipo_consumo."""
    df = pd.read_csv(path, parse_dates=["tiempo"])
    df["diag_group"] = df["diag_group"].fillna("Otro")
    df = df.rename(columns={"TOTAL_NETO": "total_neto"})
    return df


@st.cache_data(show_spinner="Cargando datos de productos (ABC / alto costo)...")
def load_product_annual(path: Path = PRODUCT_FILE) -> pd.DataFrame:
    """Serie anual a nivel de producto/ítem, solo los 7 diagnósticos FISSAL."""
    df = pd.read_csv(path)
    df["UIT"] = df["ANNIO"].map(UIT_BY_YEAR)
    df["umbral_9UIT"] = df["UIT"] * 9
    return df


@st.cache_data(show_spinner=False)
def load_norms(path: Path = NORMS_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["fecha_publicacion", "fecha_efectiva"])
    return df


NON_ONCO_CHAPTERS = set("ABEFGIJKLMNOQSTU")
"""Capítulos CIE-10 (primera letra del código) que se consideran NO oncológicos
para el grupo comparador. Ver NO_ONCO_LABEL más arriba para el detalle completo
de qué se incluye y qué se excluye deliberadamente."""


def classify_diagnosis(code) -> str | None:
    """Clasifica un código CIE-10 en uno de los 7 diagnósticos oncológicos
    priorizados por FISSAL, OTRO_INTERNAL (otro cáncer, tumor in situ/benigno,
    código Z de sesión oncológica, o código R de síntoma — ambiguos en un
    hospital especializado en cáncer), NO_ONCO_LABEL (comorbilidad no
    oncológica) o None si el código viene vacío. Es la MISMA lógica usada para
    construir monthly_diag_iafas_tipo.csv originalmente; se expone aquí para
    poder reclasificar datos nuevos subidos desde la interfaz de forma
    idéntica a como se clasificaron los históricos."""
    if code is None or (isinstance(code, float) and pd.isna(code)):
        return None
    code = str(code).strip().upper()
    if code in ("", "NAN", "NONE"):
        return None
    c = code[:3]
    if c == "C53":
        return "Cuello uterino"
    if c == "C50":
        return "Mama"
    if c == "C18":
        return "Colon"
    if c == "C16":
        return "Estómago"
    if c == "C61":
        return "Próstata"
    if c in ("C91", "C92", "C93", "C94", "C95") or code == "C901":
        return "Leucemias"
    if c in ("C81", "C82", "C83", "C84", "C85") or code == "C963":
        return "Linfomas"
    ch = code[0]
    if ch in NON_ONCO_CHAPTERS:
        return NO_ONCO_LABEL
    if ch == "D":
        sub = code[1:3]
        try:
            return NO_ONCO_LABEL if int(sub) > 48 else OTRO_INTERNAL
        except ValueError:
            return OTRO_INTERNAL
    return OTRO_INTERNAL


def process_raw_upload(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Toma un DataFrame en el mismo formato crudo que 2021_2025_CONSUMO.xlsx
    (columnas ANNIO, MES, ESFISSAL, TIPO_CONSUMO o TIPOCONSUMO, COD_DIAG,
    TOTAL_NETO) y lo agrega a (tiempo, diag_group, iafas, tipo_consumo,
    total_neto), listo para combinarse con la serie mensual vía
    merge_monthly_data(). Lanza ValueError con un mensaje claro si faltan
    columnas requeridas."""
    df = df_raw.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    if "TIPOCONSUMO" in df.columns and "TIPO_CONSUMO" not in df.columns:
        df = df.rename(columns={"TIPOCONSUMO": "TIPO_CONSUMO"})
    required = ["ANNIO", "MES", "ESFISSAL", "TIPO_CONSUMO", "COD_DIAG", "TOTAL_NETO"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(missing)}")

    df["tiempo"] = pd.to_datetime(
        {"year": df["ANNIO"].astype(int), "month": df["MES"].astype(int), "day": 1}
    )
    df["iafas"] = df["ESFISSAL"].astype(str).str.strip().str.upper().map({"S": "FISSAL", "N": "SIS"})
    df["tipo_consumo"] = df["TIPO_CONSUMO"].astype(str).str.strip().str.upper()
    df["diag_group"] = df["COD_DIAG"].apply(classify_diagnosis)
    n_dropped = df["iafas"].isna().sum() + df["diag_group"].isna().sum()
    df = df.dropna(subset=["iafas", "diag_group"])

    g = df.groupby(["tiempo", "diag_group", "iafas", "tipo_consumo"], as_index=False)["TOTAL_NETO"].sum()
    g = g.rename(columns={"TOTAL_NETO": "total_neto"})
    g.attrs["n_dropped"] = int(n_dropped)
    return g


def merge_monthly_data(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Combina datos nuevos con el histórico: si (tiempo, diag_group, iafas,
    tipo_consumo) ya existía, el valor nuevo REEMPLAZA al viejo; si la
    combinación no existía, se agrega."""
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["tiempo", "diag_group", "iafas", "tipo_consumo"], keep="last")
    return combined.sort_values(["tiempo", "diag_group", "iafas", "tipo_consumo"]).reset_index(drop=True)



def breakpoint_view(norms: pd.DataFrame, grouped: bool) -> pd.DataFrame:
    """Vista de los puntos de corte lista para usar en la app, según el
    modo elegido:

      - grouped=False (por defecto): cada norma es su propio punto de
        corte — una fila = una norma, exactamente como en norms.csv.
      - grouped=True: las normas que comparten `grupo_id` (porque su
        vigencia efectiva cae a menos de 3 meses de otra) se combinan en
        un único punto de corte, usando la fecha efectiva más tardía del
        grupo. Útil cuando los segmentos mensuales entre normas muy
        próximas son demasiado cortos para estimarse por separado con
        precisión (ver columna `nota` de las normas individuales).

    En ambos casos devuelve un DataFrame con las columnas id, label,
    norma, fecha_publicacion, fecha_efectiva, nota — listo para
    breaks_dict() y para mostrar en la interfaz."""
    if not grouped:
        return norms[["id", "label", "norma", "fecha_publicacion", "fecha_efectiva", "nota"]].copy()

    rows = []
    for gid, g in norms.groupby("grupo_id", sort=False):
        g = g.sort_values("fecha_efectiva")
        rows.append({
            "id": gid,
            "label": g["grupo_label"].iloc[0],
            "norma": " + ".join(g["norma"]),
            "fecha_publicacion": g["fecha_publicacion"].min(),
            "fecha_efectiva": g["fecha_efectiva"].max(),
            "nota": (
                f"Paquete de {len(g)} norma(s) agrupadas por proximidad; punto de corte en la vigencia "
                f"efectiva más tardía del grupo." if len(g) > 1 else "Norma individual (grupo de una sola)."
            ),
        })
    out = pd.DataFrame(rows)
    # conserva el orden cronológico de aparición en el archivo original
    order = norms.drop_duplicates("grupo_id")["grupo_id"].tolist()
    out["_order"] = out["id"].map({gid: i for i, gid in enumerate(order)})
    return out.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def breaks_dict(norms: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """{id: fecha_efectiva} en el orden del archivo, listo para
    stats_analysis.segmented_its(). Acepta tanto la vista individual como
    la agrupada de breakpoint_view()."""
    return dict(zip(norms["id"], norms["fecha_efectiva"]))


def diag_options(include_total: bool = True, include_otro: bool = True, include_no_onco: bool = True) -> list[str]:
    opts = list(ONCO_DIAGS)
    if include_otro:
        opts.append(OTRO_LABEL)
    if include_no_onco:
        opts.append(NO_ONCO_LABEL)
    if include_total:
        opts = [ALL_DIAGS_LABEL] + opts
    return opts


def _resolve_diag_values(diagnosticos: list[str]) -> list[str]:
    """Traduce etiquetas de interfaz a los valores reales de diag_group
    en los CSV (por ahora solo difiere OTRO_LABEL -> OTRO_INTERNAL)."""
    return [OTRO_INTERNAL if d == OTRO_LABEL else d for d in diagnosticos]


def filter_monthly(
    df: pd.DataFrame,
    diagnosticos: list[str],
    iafas: list[str],
    tipos: list[str],
    date_lo: pd.Timestamp | None = None,
    date_hi: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Filtra la serie mensual según las selecciones del panel lateral.
    Si `diagnosticos` incluye ALL_DIAGS_LABEL, no se filtra por diagnóstico
    (se agregan todos, incl. 'Otro')."""
    d = df.copy()
    if diagnosticos and ALL_DIAGS_LABEL not in diagnosticos:
        d = d[d["diag_group"].isin(_resolve_diag_values(diagnosticos))]
    if iafas:
        d = d[d["iafas"].isin(iafas)]
    if tipos:
        d = d[d["tipo_consumo"].isin(tipos)]
    if date_lo is not None:
        d = d[d["tiempo"] >= date_lo]
    if date_hi is not None:
        d = d[d["tiempo"] <= date_hi]
    return d


def monthly_series(d_filtered: pd.DataFrame) -> pd.Series:
    """Colapsa un subconjunto ya filtrado a una serie mensual única
    (tiempo -> total_neto), con huecos rellenados en 0."""
    s = d_filtered.groupby("tiempo")["total_neto"].sum()
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="MS") if len(s) else s.index
    return s.reindex(full_idx, fill_value=0.0)


def annual_series(d_filtered: pd.DataFrame) -> pd.Series:
    ann = d_filtered.copy()
    ann["year"] = ann["tiempo"].dt.year
    return ann.groupby("year")["total_neto"].sum()
