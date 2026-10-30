"""
qa_section.py
-------------
Interfaz Streamlit de «Pregunta a los datos»: el bloque del panel lateral
(caja de texto, ejemplos, interruptor de filtros, modo IA) y el recuadro de
respuesta en el área principal. La lógica de interpretación y cálculo está
en qa_engine.py.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from qa_engine import (
    AI_MODELS,
    EXAMPLES,
    ai_narrate,
    ai_parse,
    build_product_index,
    parse_question,
    run_query,
)

_CSS = """
<style>
.st-key-qa_examples button { padding: 0.05rem 0; min-height: 0; justify-content: flex-start; }
.st-key-qa_examples button p { font-style: italic; font-size: 0.84rem; color: #2C5F8A; text-align: left; }
.st-key-qa_examples button:hover p { text-decoration: underline; }
.st-key-qa_examples [data-testid="stVerticalBlock"] { gap: 0.15rem; }
.qa-mode { font-size: 0.8rem; color: #555; margin-top: 0.4rem; line-height: 1.4; }
.qa-scope { font-size: 0.82rem; color: #555; margin: 0.2rem 0 0.6rem 0; }
.qa-summary { background: #f4f7fb; border-left: 4px solid #2C5F8A; border-radius: 6px;
              padding: 0.7rem 1rem; margin: 0.4rem 0 0.8rem 0; font-size: 0.95rem; color: #1F2D3D; }
</style>
"""


def _secret_key() -> str:
    try:
        return str(st.secrets.get("ANTHROPIC_API_KEY", "") or "")
    except Exception:
        return ""


def _submit() -> None:
    q = (st.session_state.get("qa_q") or "").strip()
    st.session_state["qa_active"] = q or None


def _use_example(q: str) -> None:
    st.session_state["qa_q"] = q
    st.session_state["qa_active"] = q


def _clear() -> None:
    st.session_state["qa_q"] = ""
    st.session_state["qa_active"] = None


@st.cache_data(show_spinner=False)
def _product_index(product_df: pd.DataFrame) -> pd.DataFrame:
    return build_product_index(product_df)


def render_qa_panel() -> None:
    """Bloque «Pregunta a los datos» del panel lateral (llamar dentro de st.sidebar)."""
    st.markdown(_CSS, unsafe_allow_html=True)
    st.session_state.setdefault("qa_q", "")
    st.session_state.setdefault("qa_active", None)

    with st.expander("💬 Pregunta a los datos", expanded=True):
        st.toggle(
            "Aplicar los filtros del panel", value=True, key="qa_apply",
            help="Si está activo, la respuesta se calcula sobre la selección del panel (IAFAS, tipo de consumo, "
                 "diagnóstico y rango de meses). Lo que la pregunta mencione explícitamente prevalece sobre el "
                 "panel para esa dimensión.",
        )
        st.text_area(
            "Pregunta", key="qa_q", label_visibility="collapsed", height=80,
            placeholder="Ej.: ¿qué medicamentos explican más gasto en FISSAL en 2024?",
        )
        c1, c2 = st.columns(2)
        c1.button("Consultar", type="primary", use_container_width=True, on_click=_submit, key="qa_btn_go")
        c2.button("Limpiar", use_container_width=True, on_click=_clear, key="qa_btn_clear")

        st.caption("Ejemplos de preguntas (tócalos para consultar)")
        with st.container(key="qa_examples"):
            for i, ex in enumerate(EXAMPLES):
                st.button(ex, type="tertiary", key=f"qa_ex_{i}", on_click=_use_example, args=(ex,))

        ai_key = st.session_state.get("qa_ai_key_input") or _secret_key()
        ai_on = bool(st.session_state.get("qa_ai_on")) and bool(ai_key)
        if ai_on:
            model_lbl = st.session_state.get("qa_ai_model", next(iter(AI_MODELS)))
            st.markdown(
                f'<div class="qa-mode">✨ <b>Modo IA activo</b> ({model_lbl.split(" (")[0]}). El modelo interpreta '
                f'la pregunta y redacta la lectura; las cifras siguen saliendo de agregaciones locales y al modelo '
                f'solo se envían tablas resumidas, sin identificadores.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="qa-mode">🌘 Modo básico por reglas. Las cifras salen de agregaciones sobre los datos '
                'del observatorio; al modelo solo se envían tablas resumidas, sin identificadores.</div>',
                unsafe_allow_html=True,
            )

        with st.popover("Activar modo IA", use_container_width=True):
            st.toggle("Usar modo IA (Claude)", key="qa_ai_on")
            if not _secret_key():
                st.text_input(
                    "Clave de API de Anthropic", type="password", key="qa_ai_key_input",
                    help="Se usa solo durante esta sesión del navegador y no se guarda. En Streamlit Cloud puedes "
                         "definir ANTHROPIC_API_KEY en Secrets para no tener que escribirla.",
                )
            else:
                st.caption("Clave configurada en los Secrets de la app.")
            st.selectbox("Modelo", list(AI_MODELS), key="qa_ai_model")
            st.caption(
                "Qué se envía: la pregunta, el catálogo de valores válidos (IAFAS, diagnósticos, tipos, años) y la "
                "tabla agregada del resultado (máx. 40 filas). Nunca se envían registros individuales."
            )
            if st.session_state.get("qa_ai_on") and not ai_key:
                st.warning("Ingresa una clave de API para activar el modo IA.")


def _fmt_table(t: pd.DataFrame):
    fmts = {}
    for c in t.columns:
        if c.startswith("Gasto"):
            fmts[c] = "S/ {:,.0f}"
        elif c == "Cantidad":
            fmts[c] = "{:,.0f}"
        elif c.startswith("%") or c.startswith("Var."):
            fmts[c] = "{:,.1f}%"
    return t.style.format(fmts, na_rep="")


def render_qa_answer(monthly_df: pd.DataFrame, product_df: pd.DataFrame, listado_df: pd.DataFrame,
                     panel: dict) -> None:
    """Recuadro de respuesta en el área principal (solo si hay una pregunta activa)."""
    q = st.session_state.get("qa_active")
    if not q:
        return

    apply_panel = st.session_state.get("qa_apply", True)
    panel_used = panel if apply_panel else None
    pindex = _product_index(product_df)
    years_avail = sorted(int(y) for y in product_df["ANNIO"].unique())

    ai_key = st.session_state.get("qa_ai_key_input") or _secret_key()
    ai_on = bool(st.session_state.get("qa_ai_on")) and bool(ai_key)
    model = AI_MODELS.get(st.session_state.get("qa_ai_model", ""), next(iter(AI_MODELS.values())))

    ai_note = None
    narrative = None
    spec = parse_question(q, years_avail, pindex)
    if ai_on:
        cache = st.session_state.setdefault("qa_ai_cache", {})
        ck = json.dumps([q, model, apply_panel, {k: str(v) for k, v in (panel_used or {}).items()}], ensure_ascii=False)
        if ck in cache:
            spec, narrative, ai_note = cache[ck]
        else:
            try:
                with st.spinner("Interpretando la pregunta con IA..."):
                    spec = ai_parse(q, ai_key, model, years_avail)
            except Exception as e:  # cae al modo por reglas
                ai_note = f"No se pudo usar el modo IA para interpretar ({type(e).__name__}); se usaron las reglas."
            res_tmp = run_query(q, spec, monthly_df, product_df, listado_df, pindex, panel_used)
            if res_tmp.ok and res_tmp.table is not None:
                try:
                    with st.spinner("Redactando la lectura..."):
                        narrative = ai_narrate(res_tmp, ai_key, model)
                except Exception as e:
                    ai_note = (ai_note or "") + f" No se pudo redactar la lectura con IA ({type(e).__name__})."
            cache[ck] = (spec, narrative, ai_note)

    res = run_query(q, spec, monthly_df, product_df, listado_df, pindex, panel_used)

    with st.container(border=True):
        h1, h2 = st.columns([0.88, 0.12])
        h1.markdown(f"#### 💬 {q}")
        h2.button("✕ Cerrar", key="qa_close", on_click=_clear, use_container_width=True)
        mode_txt = "✨ Modo IA" if ai_on and not (ai_note or "").startswith("No se pudo usar") else "🌘 Modo por reglas"
        filt_txt = "con filtros del panel" if apply_panel else "sin filtros del panel"
        if res.scope:
            st.markdown(f'<div class="qa-scope">{mode_txt} · {filt_txt} · {res.scope}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="qa-scope">{mode_txt}</div>', unsafe_allow_html=True)

        if not res.ok:
            st.info(f"**{res.title}.** {res.summary}")
            if not res.scope:
                st.caption("Prueba con una de las preguntas de ejemplo del panel lateral.")
            return

        st.markdown(f'<div class="qa-summary">{_md_bold(narrative or res.summary)}</div>', unsafe_allow_html=True)
        if narrative:
            st.caption("Lectura redactada por IA a partir de la tabla de abajo; verifica las cifras en la tabla.")
        if ai_note:
            st.caption(ai_note.strip())

        if res.fig is not None:
            st.plotly_chart(res.fig, use_container_width=True, key="qa_fig")
        if res.table is not None:
            with st.expander("Ver tabla del resultado", expanded=False):
                st.dataframe(_fmt_table(res.table), hide_index=True, use_container_width=True)
                st.download_button(
                    "⬇️ Descargar resultado (CSV)", res.table.to_csv(index=False).encode("utf-8"),
                    file_name="pregunta_a_los_datos.csv", mime="text/csv", key="qa_dl",
                )
        for n in res.notes:
            st.caption("ℹ️ " + n)
        with st.expander("¿Cómo se interpretó la pregunta?", expanded=False):
            st.json({k: v for k, v in res.spec.to_dict().items() if v not in ([], None, "", False)})


def _md_bold(s: str) -> str:
    """Convierte **negritas** de markdown a <b> para el recuadro HTML."""
    import html
    import re
    s = html.escape(s)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
