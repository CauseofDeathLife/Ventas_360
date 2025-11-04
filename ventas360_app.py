# ventas360_app.py
# ---------------------------------------------------------------------
# Dataset FIJO: data/ventas_5k.csv
# - No lista otros CSV ni lee nada desde out/
# - Filtros estándar (fecha, vendedor, ciudad, producto)
# - Estado forzado a "Cerrado" (sin widget)
# - KPIs coherentes, utilidad auto (5%) si no existe
# - Comisión = 0 cuando estado != "Cerrado"
# ---------------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# =========================
# Configuración de página
# =========================
st.set_page_config(
    page_title="Ventas360 — Analítica",
    layout="wide",
    page_icon="📊"
)

# =========================
# Constantes
# =========================
DATASET_PATH = Path("data/ventas_5k.csv")
UTILIDAD_PCT_DEF = 0.05       # utilidad virtual si falta columna
FORZAR_ESTADO_CERRADO = True  # fuerza estado = "Cerrado" sin mostrar widget

# =========================
# Utilidades
# =========================
def money(n: float) -> str:
    try:
        return f"${n:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "—"

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    return df

# =========================
# Carga y preparación
# =========================
@st.cache_data
def cargar_dataset_principal(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(
            f"No encuentro {path}.\n\n"
            "Genera el dataset con:\n"
            "python ventas360_cli.py generate --n 5000 --csv data/ventas_5k.csv"
        )
        st.stop()

    df = pd.read_csv(path)
    df = normalize_cols(df)

    # fecha
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    # valor_venta si faltara
    if "valor_venta" not in df.columns and {"precio", "cantidad"}.issubset(df.columns):
        df["valor_venta"] = (df["precio"].astype(float) * df["cantidad"].astype(float)).astype(float)

    # utilidad si faltara
    if "utilidad" not in df.columns and "valor_venta" in df.columns:
        df["utilidad"] = (df["valor_venta"] * UTILIDAD_PCT_DEF).round(2)

    # comisión = 0 si no está cerrado (si existen ambas columnas)
    if {"comision", "estado"}.issubset(df.columns):
        df["comision"] = np.where(df.get("estado", "").astype(str).str.strip().eq("Cerrado"), df["comision"], 0.0)

    return df

df_base = cargar_dataset_principal(DATASET_PATH)
st.sidebar.caption(f"Dataset fijo: **{DATASET_PATH.as_posix()}**")

# =========================
# Filtros (SIN selector de datasets)
# =========================
df = df_base.copy()

# Estado forzado a "Cerrado" (no se muestra widget)
if FORZAR_ESTADO_CERRADO and "estado" in df.columns:
    df = df[df["estado"].astype(str).str.strip().eq("Cerrado")]

# Rango de fechas
if "fecha" in df.columns and df["fecha"].notna().any():
    fmin = pd.to_datetime(df["fecha"].min())
    fmax = pd.to_datetime(df["fecha"].max())
    r = st.sidebar.date_input("Rango de fechas", value=(fmin.date(), fmax.date()))
    if isinstance(r, tuple) and len(r) == 2:
        start, end = pd.to_datetime(r[0]), pd.to_datetime(r[1])
        df = df[(df["fecha"] >= start) & (df["fecha"] <= end)]

# Filtro por vendedor
if "vendedor" in df.columns:
    vendedores = ["[Todos]"] + sorted(df["vendedor"].dropna().astype(str).unique().tolist())
    sel_v = st.sidebar.multiselect("Vendedor", vendedores, default=["[Todos]"])
    if sel_v and "[Todos]" not in sel_v:
        df = df[df["vendedor"].isin(sel_v)]

# Filtro por ciudad
if "ciudad" in df.columns:
    ciudades = ["[Todos]"] + sorted(df["ciudad"].dropna().astype(str).unique().tolist())
    sel_c = st.sidebar.multiselect("Ciudad", ciudades, default=["[Todos]"])
    if sel_c and "[Todos]" not in sel_c:
        df = df[df["ciudad"].isin(sel_c)]

# Filtro por producto
if "producto" in df.columns:
    productos = ["[Todos]"] + sorted(df["producto"].dropna().astype(str).unique().tolist())
    sel_p = st.sidebar.multiselect("Producto", productos, default=["[Todos]"])
    if sel_p and "[Todos]" not in sel_p:
        df = df[df["producto"].isin(sel_p)]

# Filtro por categoría (nuevo)
if "categoria" in df.columns:
    categorias = ["[Todos]"] + sorted(df["categoria"].dropna().astype(str).unique().tolist())
    sel_cat = st.sidebar.multiselect("Categoría", categorias, default=["[Todos]"])
    if sel_cat and "[Todos]" not in sel_cat:
        df = df[df["categoria"].isin(sel_cat)]


# =========================
# KPIs
# =========================
ventas_total = float(df["valor_venta"].sum()) if "valor_venta" in df.columns else 0.0
utilidad_total = float(df["utilidad"].sum()) if "utilidad" in df.columns else 0.0
filas = int(len(df))
margen_pct = (utilidad_total / ventas_total * 100.0) if ventas_total > 0 else 0.0

st.title("Ventas360 — Analítica")
st.caption("Cálculos basados exclusivamente en `data/ventas_5k.csv`")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Ventas (Σ)", money(ventas_total))
k2.metric("Utilidad (Σ)", money(utilidad_total))
k3.metric("Margen %", f"{margen_pct:.2f} %")
k4.metric("Filas", f"{filas:,}".replace(",", "."))

# =========================
# Vista previa
# =========================
with st.expander("Vista previa (primeras 50 filas)", expanded=False):
    st.dataframe(df.head(50), use_container_width=True)

# =========================
# Gráficos sencillos
# =========================
def bar_chart(df_plot: pd.DataFrame, x_col: str, y_col: str, titulo: str):
    chart = (
        alt.Chart(df_plot)
        .mark_bar()
        .encode(
            x=alt.X(x_col, sort="-y"),
            y=alt.Y(y_col),
            tooltip=[x_col, y_col]
        )
        .properties(height=360, title=titulo)
    )
    st.altair_chart(chart, use_container_width=True)

# Ventas por vendedor
if {"vendedor", "valor_venta"}.issubset(df.columns):
    tmp = df.groupby("vendedor", dropna=False)["valor_venta"].sum().reset_index().sort_values("valor_venta", ascending=False)
    bar_chart(tmp, "vendedor", "valor_venta", "Ventas por vendedor")

# Ventas por ciudad
if {"ciudad", "valor_venta"}.issubset(df.columns):
    tmp = df.groupby("ciudad", dropna=False)["valor_venta"].sum().reset_index().sort_values("valor_venta", ascending=False)
    bar_chart(tmp, "ciudad", "valor_venta", "Ventas por ciudad")

# Ventas por producto (top N)
if {"producto", "valor_venta"}.issubset(df.columns):
    tmp = df.groupby("producto", dropna=False)["valor_venta"].sum().reset_index().sort_values("valor_venta", ascending=False).head(20)
    bar_chart(tmp, "producto", "valor_venta", "Top 20 productos por ventas")

# Serie temporal de ventas
if {"fecha", "valor_venta"}.issubset(df.columns):
    serie = df.dropna(subset=["fecha"]).copy()
    if not serie.empty:
        serie = serie.groupby(pd.Grouper(key="fecha", freq="D"))["valor_venta"].sum().reset_index()
        line = (
            alt.Chart(serie)
            .mark_line()
            .encode(x="fecha:T", y="valor_venta:Q", tooltip=["fecha:T", "valor_venta:Q"])
            .properties(height=360, title="Ventas diarias")
        )
        st.altair_chart(line, use_container_width=True)

# =========================
# Tabla dinámica (opcional)
# =========================
st.subheader("Tabla dinámica")
agrupables = [c for c in ["vendedor", "ciudad", "producto", "categoria"] if c in df.columns]
metricas = [c for c in ["valor_venta", "utilidad", "cantidad", "precio", "comision"] if c in df.columns]

if agrupables and metricas:
    cols = st.multiselect("Agrupar por", agrupables, default=["vendedor"] if "vendedor" in agrupables else agrupables[:1])
    metrica = st.selectbox("Métrica", metricas, index=metricas.index("valor_venta") if "valor_venta" in metricas else 0)
    agg = st.selectbox("Agregación", ["sum", "mean", "count"], index=0)
    if cols:
        piv = df.groupby(cols, dropna=False)[metrica].agg(agg).reset_index().sort_values(metrica, ascending=False)
        st.dataframe(piv, use_container_width=True)
else:
    st.info("No hay columnas suficientes para tabla dinámica.")

# =========================
# Notas
# =========================
with st.expander("Notas de lógica", expanded=False):
    st.markdown(
        """
- Dataset **único y fijo**: `data/ventas_5k.csv`.
- El estado se fuerza a **Cerrado**; comisiones a 0 si no es Cerrado.
- Si falta `utilidad`, se calcula como **5%** de `valor_venta`.
- La app **no** lista ni carga CSV de `out/` ni otros en `data/`.
        """
    )
