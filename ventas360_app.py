# app.py
# -*- coding: utf-8 -*-
"""
Ventas 360 — Dashboard Streamlit (KPIs, filtros, Altair)

Descripción
-----------
Interfaz web para explorar KPIs y gráficos de ventas. Carga automáticamente:
1) Resultados generados por el CLI en `out/` (si existen).
2) Dataset base en `data/` como respaldo.

Características
---------------
- KPIs principales (totales, promedios, top categorías/productos/ciudades).
- Filtros por rango de fechas, ciudad, producto y estado (si aplica).
- Gráficos interactivos con Altair (series mensuales, comparativas, etc.).
- Tabla dinámica para explorar registros filtrados.

Asunciones de datos
-------------------
- Fechas entre 2021–2025.
- Montos en COP con 2 decimales.
- Estructura de carpetas del proyecto:
    data/  -> dataset base (p. ej., ventas_5k.csv)
    out/   -> salidas del CLI (CSV agregados + DOCX)
    *.py   -> ventas360_cli.py / ventas360_app.py

Uso rápido
----------
streamlit run ventas360_app.py

Requisitos
----------
- Python 3.10+
- pip install -r requirements.txt

Notas
-----
- Si `out/` está vacío, el dashboard intentará usar `data/`.
- Para resultados más ricos, ejecute primero el CLI para poblar `out/`.
- Los gráficos usan Altair; los datos se muestran en COP (formato local).
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt


st.set_page_config(page_title="Evaluación de Datos - Pandas", layout="wide")
st.title("Evaluación de Datos con Pandas")
st.caption("Carga automática de datasets en `out/` (y `data/` si existe) + KPIs, filtros, tabla dinámica y gráficos")

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "out"
DATA_DIR = BASE / "data"



# ---------- utilidades ----------
def discover_files():
    exts = [".csv", ".xlsx", ".xls", ".json", ".parquet"]
    found = []
    for d in [OUT_DIR, DATA_DIR]:
        if d.exists():
            for p in sorted(d.rglob("*")):
                if p.suffix.lower() in exts and p.is_file():
                    found.append(p)
    return found

@st.cache_data(show_spinner=False)
def load_df(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        # tolerante con encoding y separador
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, sep=";", encoding_errors="ignore")
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suf == ".json":
        return pd.read_json(path)
    if suf == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Extensión no soportada: {suf}")

def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    # intenta convertir columnas típicas de fecha
    for col in df.columns:
        if col.lower() in ("fecha", "date", "order_date"):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def apply_date_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica filtro de fechas robusto: solo activa el widget si hay fechas válidas."""
    if "fecha" not in df.columns or not pd.api.types.is_datetime64_any_dtype(df["fecha"]):
        return df
    valid = df["fecha"].dropna()
    if valid.empty:
        st.sidebar.info("Dataset sin fechas válidas; filtro de fechas desactivado.")
        return df
    min_d = valid.min().date()
    max_d = valid.max().date()
    # Intentar crear el widget con rango completo por defecto
    try:
        d1, d2 = st.sidebar.date_input("Rango de fechas", value=(min_d, max_d))
        # Algunas versiones pueden devolver un solo valor o un tuple extraño
        if isinstance(d1, (tuple, list)):
            d1, d2 = d1
    except Exception:
        # Fallback en caso de NaT/timetuple u otros
        d1, d2 = (min_d, max_d)
    # Filtro por fechas con .dt.date para evitar tz y NaT raros
    mask = (df["fecha"].dt.date >= d1) & (df["fecha"].dt.date <= d2)
    return df.loc[mask]

    # intenta convertir columnas típicas de fecha
    for col in df.columns:
        if col.lower() in ("fecha", "date", "order_date"):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # mapea columnas frecuentes a nombres consistentes (si existen)
    mapping = {
        "VALOR_VENTA": "valor_venta",
        "UTILIDAD": "utilidad",
        "CATEGORIA": "categoria",
        "CIUDAD": "ciudad",
        "VENDEDOR": "vendedor",
        "ESTADO": "estado",
        "PRODUCTO": "producto",
        "FECHA": "fecha",
        "COMISION": "comision",
}
    to_apply = {src: dst for src, dst in mapping.items() if src in df.columns}
    return df.rename(columns=to_apply)

def money_fmt(x):
    try:
        s = f"{float(x):,.2f}"
        s = s.replace(",", "_").replace(".", ",").replace("_", ".")
        return f"$ {s}"
    except Exception:
        return x

# ---------- carga automática ----------
files = discover_files()
if not files:
    st.error("No se encontraron datasets en `out/` ni en `data/`. Coloca tus archivos ahí.")
    st.stop()

# selector de dataset (carga uno por defecto)
st.sidebar.header("Datasets disponibles")
labels = [str(p.relative_to(BASE)) for p in files]
choice = st.sidebar.selectbox("Selecciona un dataset", labels, index=0)
chosen_path = files[labels.index(choice)]

df = load_df(chosen_path)
df = normalize_cols(df)
df = ensure_datetime(df)

st.success(f"Dataset cargado — **{chosen_path.name}**  ·  Dimensiones: {df.shape[0]:,} x {df.shape[1]:,}")

# ---------- filtros (sidebar) ----------
df = apply_date_filter(df)


for col in ("ciudad", "categoria", "estado", "vendedor", "producto"):
    if col in df.columns:
        vals = sorted(df[col].dropna().astype(str).unique().tolist())[:2000]
        sel = st.sidebar.multiselect(col.capitalize(), vals)
        if sel:
            df = df[df[col].astype(str).isin(sel)]

st.write(f"**Dimensiones tras filtros:** {df.shape[0]:,} filas × {df.shape[1]:,} columnas")

# ---------- KPIs ----------
k1, k2, k3, k4 = st.columns(4, gap="small")
with k1:
    total_venta = df.get("valor_venta", pd.Series(dtype=float)).sum()
    st.metric("Ventas (∑)", money_fmt(total_venta))
with k2:
    total_util = df.get("utilidad", pd.Series(dtype=float)).sum()
    st.metric("Utilidad (∑)", money_fmt(total_util))
with k3:
    if total_venta and total_venta != 0:
        st.metric("Margen %", f"{100*total_util/total_venta:,.1f}%")
    else:
        st.metric("Margen %", "—")
with k4:
    st.metric("Filas", f"{len(df):,}")

# ---------- vista previa estilizada ----------
st.subheader("Vista previa (estilizada)")
style_cols = [c for c in ["valor_venta", "utilidad", "comision"] if c in df.columns]
styled = df.head(1000).style
if style_cols:
    styled = styled.format({c: money_fmt for c in style_cols}).background_gradient(
        subset=style_cols, cmap="Greens"
    )
st.write(styled)

# ---------- tabla dinámica ----------
st.subheader("Tabla dinámica")
group_by = st.multiselect("Agrupar por", [c for c in df.columns if df[c].dtype == "object"])
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
value_col = st.selectbox("Métrica", num_cols, index=0 if num_cols else None)
agg_fn = st.selectbox("Agregación", ["sum", "mean", "median", "max", "min", "count"], index=0)

if group_by and value_col:
    pivot = df.groupby(group_by, dropna=False)[value_col].agg(agg_fn).reset_index() \
              .sort_values(value_col, ascending=False)
    st.dataframe(pivot, use_container_width=True)
    st.download_button("Descargar tabla dinámica (CSV)",
                       data=pivot.to_csv(index=False).encode("utf-8"),
                       file_name="pivot.csv", mime="text/csv")

# ---------- gráficos (Altair: ya viene con Streamlit) ----------
st.subheader("Gráficos")
if group_by and value_col:
    cat = group_by[0]
    topN = st.slider("Top N", 5, 30, 10)
    bars = (df.groupby(cat, dropna=False)[value_col].sum()
              .sort_values(ascending=False).head(topN).reset_index())
    chart = alt.Chart(bars).mark_bar().encode(
        x=alt.X(cat, sort='-y'),
        y=alt.Y(value_col, title=value_col),
        tooltip=[cat, value_col]
    ).properties(height=320)
    st.altair_chart(chart, use_container_width=True)

if "fecha" in df.columns and pd.api.types.is_datetime64_any_dtype(df["fecha"]) and value_col and df["fecha"].notna().any():
    monthly = (df.loc[df["fecha"].notna()].set_index("fecha")[value_col]
                 .resample("MS").sum().reset_index())
    line = alt.Chart(monthly).mark_line().encode(
        x="fecha:T", y=alt.Y(value_col, title=value_col), tooltip=["fecha:T", value_col]
    ).properties(height=320)
    st.altair_chart(line, use_container_width=True)

# ---------- export ----------
st.download_button("Descargar CSV filtrado", data=df.to_csv(index=False).encode("utf-8"),
                   file_name=f"{chosen_path.stem}_filtrado.csv", mime="text/csv")

