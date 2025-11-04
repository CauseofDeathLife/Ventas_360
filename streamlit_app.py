import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Ventas360 · Dashboard", page_icon="📊", layout="wide")

# ===================== Carga base =====================
@st.cache_data
def load_base(path="data/ventas_5k.csv"):
    df = pd.read_csv(path, low_memory=False)
    # normaliza nombres
    df.columns = [c.lower() for c in df.columns]
    # fecha
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    # valor_venta si no existe (precio * cantidad)
    if "valor_venta" not in df.columns and {"precio","cantidad"} <= set(df.columns):
        df["valor_venta"] = (df["precio"].astype(float) * df["cantidad"].astype(float)).round(2)
    return df

df = load_base()

# ===================== Sidebar =====================
st.sidebar.header("Filtros")

# 1) Rango de fechas (si aplica)
if "fecha" in df.columns and df["fecha"].notna().any():
    dmin, dmax = pd.to_datetime(df["fecha"].min()), pd.to_datetime(df["fecha"].max())
    rango = st.sidebar.date_input("Rango de fechas", (dmin.date(), dmax.date()))
else:
    rango = None

# helper para multiselect dependiente
def ms(df_ref, col, label):
    if col not in df_ref.columns:
        return []
    opciones = sorted(df_ref[col].dropna().unique())
    return st.sidebar.multiselect(label, opciones)

# 2) Filtros encadenados
df_step = df.copy()
if rango and "fecha" in df_step.columns:
    fi, ff = pd.to_datetime(rango[0]), pd.to_datetime(rango[1])
    df_step = df_step[df_step["fecha"].between(fi, ff)]

sel_ciudad    = ms(df_step, "ciudad",    "Ciudad")
df_step = df_step if not sel_ciudad else df_step[df_step["ciudad"].isin(sel_ciudad)]

sel_categoria = ms(df_step, "categoria", "Categoría")
df_step = df_step if not sel_categoria else df_step[df_step["categoria"].isin(sel_categoria)]

sel_estado    = ms(df_step, "estado",    "Estado")
df_step = df_step if not sel_estado else df_step[df_step["estado"].isin(sel_estado)]

# columna a usar para vendedor (preferir nombre si existe)
vend_col_filter = "vendedor" if "vendedor" in df.columns else ("vendedor_nombre" if "vendedor_nombre" in df.columns else "vendedor_id")

vendor_options = sorted(df_step[vend_col_filter].dropna().unique()) if vend_col_filter in df_step.columns else []
sel_vendedor  = st.sidebar.multiselect("Vendedor", vendor_options)

product_options = sorted(df_step["producto"].dropna().unique()) if "producto" in df_step.columns else []
sel_producto  = st.sidebar.multiselect("Producto", product_options)

# 3) Máscara final
mask = pd.Series(True, index=df.index)
if rango and "fecha" in df.columns:
    fi, ff = pd.to_datetime(rango[0]), pd.to_datetime(rango[1])
    mask &= df["fecha"].between(fi, ff)
if sel_ciudad:    mask &= df["ciudad"].isin(sel_ciudad)
if sel_categoria: mask &= df["categoria"].isin(sel_categoria)
if sel_estado:    mask &= df["estado"].isin(sel_estado)
if sel_vendedor:  mask &= df[vend_col_filter].isin(sel_vendedor)
if sel_producto:  mask &= df["producto"].isin(sel_producto)

dff = df.loc[mask].copy()

# ===================== KPIs =====================
st.title("Evaluación de Datos con Pandas")
st.caption("Fuente única: data/ventas_5k.csv · Filtros consistentes · Nombres ficticios generados desde el CLI")

venta_col = next((c for c in dff.columns if c in ["valor_venta","venta","total","monto","importe","precio_total","precio"]), None)
util_col  = next((c for c in dff.columns if "util" in c), None)

ventas_total = float(dff[venta_col].sum()) if venta_col else 0.0
utilidad_total = float(dff[util_col].sum()) if util_col else 0.0
margen_pct = (utilidad_total / ventas_total * 100) if venta_col and util_col and ventas_total else 0.0
unidades_total = int(dff["cantidad"].sum()) if "cantidad" in dff.columns else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Ventas (∑)", f"${ventas_total:,.2f}")
c2.metric("Utilidad (∑)", f"${utilidad_total:,.2f}")
c3.metric("Margen %", f"{margen_pct:,.1f}%")
c4.metric("Filas", f"{len(dff):,}")
c5.metric("Unidades (∑)", f"{unidades_total:,}")

# ===================== Vista previa =====================
# Mostrar 'precio' como 'precio_unitario' (sólo UI)
ui = dff.rename(columns={'precio': 'precio_unitario'})

st.subheader("Vista previa (estilizada)")
preview_cols = [c for c in [
    "fecha",
    ("cliente" if "cliente" in ui.columns else "cliente_nombre" if "cliente_nombre" in ui.columns else "cliente_id"),
    vend_col_filter, "ciudad", "producto", "categoria", "estado",
    "cantidad", "precio_unitario", "valor_venta"
] if c in ui.columns]
st.dataframe(ui[preview_cols].head(200) if preview_cols else ui.head(200), use_container_width=True)

# ===================== Top vendedores =====================
if venta_col and vend_col_filter in dff.columns:
    st.subheader("Top 10 vendedores por ventas")
    top_v = dff.groupby(vend_col_filter)[venta_col].sum().reset_index().sort_values(venta_col, ascending=False).head(10)
    st.altair_chart(
        alt.Chart(top_v).mark_bar().encode(
            x=alt.X(vend_col_filter, sort='-y'),
            y=venta_col,
            tooltip=[vend_col_filter, venta_col]
        ),
        use_container_width=True
    )

# ===================== Serie temporal =====================
if venta_col and "fecha" in dff.columns and dff["fecha"].notna().any():
    ts = dff.groupby(pd.Grouper(key="fecha", freq="M"))[venta_col].sum().reset_index()
    st.subheader("Ventas mensuales")
    st.altair_chart(
        alt.Chart(ts).mark_line(point=True).encode(x="fecha", y=venta_col, tooltip=["fecha", venta_col]),
        use_container_width=True
    )

# ===================== Tabla dinámica =====================
st.subheader("Tabla dinámica")
cat_cols = [c for c in dff.columns if dff[c].dtype == "object" and c not in ["fecha"]]
if cat_cols and venta_col:
    by = st.selectbox("Agrupar por", cat_cols, index=0)
    pivot = dff.groupby(by)[venta_col].sum().reset_index().sort_values(venta_col, ascending=False)
    st.dataframe(pivot, use_container_width=True)
else:
    st.info("No hay columnas categóricas o métrica de ventas para tabla dinámica.")
