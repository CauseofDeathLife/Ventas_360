#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ventas360 — CLI (generador + análisis)

Uso rápido:
  # Generar base con 5000 filas
  python ventas360_cli.py --generate --n 5000 --csv data/ventas_5k.csv

  # (Opcional) Agregados a 'out/'
  python ventas360_cli.py --analyze --csv data/ventas_5k.csv --out out
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
from datetime import date

import numpy as np
import pandas as pd

# =========================================================
# Config
# =========================================================
RANDOM_SEED = 42

CATEGORIAS = [
    "Tecnología", "Hogar", "Alimentos", "Deportes", "Ropa",
    "Belleza", "Jardín", "Mascotas", "Salud", "Juguetes",
]

PRODUCTOS_POR_CATEGORIA: Dict[str, List[str]] = {
    "Tecnología": ["Laptop", "Tablet", "Celular", "Monitor", "Teclado", "Mouse", "Televisor"],
    "Hogar": ["Sofá", "Mesa", "Silla", "Cortinas", "Lámpara"],
    "Alimentos": ["Arroz", "Atún", "Aceite", "Leche", "Galletas"],
    "Deportes": ["Balón", "Bicicleta", "Pesas", "Proteína"],
    "Ropa": ["Pantalón", "Camisa", "Zapatos", "Chaqueta"],
    "Belleza": ["Perfume", "Maquillaje", "Crema"],
    "Jardín": ["Pala", "Manguera", "Abono"],
    "Mascotas": ["Concentrado", "Arena", "Juguete"],
    "Salud": ["Alcohol", "Algodón", "Vitaminas"],
    "Juguetes": ["Rompecabezas", "Muñeco", "Carritos"],
}

# Mapa auxiliar producto -> categoría (para elegir rangos por categoría)
PRODUCTO_A_CATEGORIA = {p: cat for cat, prods in PRODUCTOS_POR_CATEGORIA.items() for p in prods}

CIUDADES = [
    "Bogotá", "Medellín", "Cali", "Barranquilla", "Bucaramanga",
    "Cartagena", "Pereira", "Manizales", "Cúcuta", "Santa Marta",
]

# 200 vendedores/ids y 5000 clientes/ids (los nombres se derivan del id)
VENDEDORES = np.array([f"Vendedor_{i:03d}" for i in range(1, 201)], dtype=object)
CLIENTES_ID = np.array([f"Cliente_{i:04d}" for i in range(1, 5001)], dtype=object)

# Listas para nombres ficticios reproducibles
NOMBRES_M = [
    "Juan","Carlos","Andrés","Pedro","Luis","Santiago","Felipe","Miguel","Jorge","Ricardo",
    "David","Camilo","Alejandro","Diego","Sebastián","Mauricio","Fernando","Cristian","Óscar","Rafael",
    "Hugo","Iván","Sergio","Daniel","Nicolás","Mateo","Álvaro","Tomás","Simón","Gabriel",
]
NOMBRES_F = [
    "María","Laura","Paula","Sara","Natalia","Carolina","Diana","Valeria","Isabela","Sofía",
    "Gabriela","Mónica","Tatiana","Daniela","Luisa","Camila","Juliana","Ana","Andrea","Lorena",
    "Karina","Verónica","Claudia","Vanessa","Adriana","Noelia","Mariana","Yolanda","Rocío","Viviana",
]
APELLIDOS = [
    "Gómez","Rodríguez","López","Martínez","Pérez","Sánchez","Ramírez","Torres","Quintero","Caro",
    "Vargas","Moreno","Rojas","Guerrero","Castro","Ortega","Cortés","Ibarra","Molina","Méndez",
    "Castaño","Salazar","Naranjo","Ospina","Arango","Bedoya","Zapata","Henao","Mejía","Gaviria",
]

import hashlib
def nombre_ficticio(key: str) -> str:
    """Nombre reproducible 'Nombre Apellido' basado en hash de la clave."""
    h = int(hashlib.sha256(str(key).encode()).hexdigest(), 16)
    pool = NOMBRES_M if (h % 2 == 0) else NOMBRES_F
    nombre = pool[h % len(pool)]
    apellido = APELLIDOS[(h // 97) % len(APELLIDOS)]
    return f"{nombre} {apellido}"

# Rangos de precio por categoría (min, max)
RANGO_PRECIO = {
    "Tecnología": (300_000, 5_000_000),
    "Hogar": (50_000, 3_000_000),
    "Alimentos": (5_000, 200_000),
    "Deportes": (30_000, 1_500_000),
    "Ropa": (20_000, 800_000),
    "Belleza": (15_000, 600_000),
    "Jardín": (10_000, 700_000),
    "Mascotas": (10_000, 600_000),
    "Salud": (10_000, 600_000),
    "Juguetes": (10_000, 500_000),
}

# =========================================================
# Precio unitario fijo por producto + variación leve por año
# =========================================================
def precios_unitarios_por_producto_y_anio(
    productos: np.ndarray,
    fechas,                      # array-like datetime
    seed: Optional[int] = None,
    base_year: int = 2021,
    inc_min: float = 0.01,       # 1% mínimo por año
    inc_max: float = 0.03        # 3% máximo por año
) -> np.ndarray:
    """
    Devuelve un array de PRECIO UNITARIO tal que:
      - para un mismo producto y año, el precio es constante;
      - por cada producto se aplica un incremento anual leve y constante:
            precio_y = base * (1 + r) ** (y - base_year)
    El precio base del producto (año base) se toma del rango de su categoría.
    """
    rng = np.random.default_rng(seed)
    productos = np.asarray(productos)
    years = pd.to_datetime(fechas).year

    prods = np.unique(productos)
    base_price: Dict[str, float] = {}
    rate: Dict[str, float] = {}
    for p in prods:
        cat = PRODUCTO_A_CATEGORIA.get(p)
        low, high = RANGO_PRECIO.get(cat, (8_000.0, 150_000.0))
        base_price[p] = float(np.round(rng.uniform(low, high), 2))
        rate[p]       = float(rng.uniform(inc_min, inc_max))

    def precio_i(i: int) -> float:
        p = productos[i]
        k = max(0, int(years[i] - base_year))  # 2021->0, 2022->1, ...
        return float(np.round(base_price[p] * ((1.0 + rate[p]) ** k), 2))

    return np.fromiter((precio_i(i) for i in range(len(productos))), dtype=float)

# =========================================================
# Generación base
# =========================================================
def generar_dataset(n: int, csv_path: Path, seed: int = RANDOM_SEED) -> Path:
    """Genera dataset sintético y lo guarda en csv_path."""
    rng = np.random.default_rng(seed)

    categorias = rng.choice(CATEGORIAS, size=n)
    productos = np.array([rng.choice(PRODUCTOS_POR_CATEGORIA[cat]) for cat in categorias], dtype=object)

    # Rango de fechas: 2021-01-01 .. HOY (sin futuros)
    start = np.datetime64("2021-01-01T00:00:00")
    today = date.today().isoformat()
    end = np.datetime64(f"{today}T23:59:00")
    delta_min = int(((end - start) / np.timedelta64(1, "m")))
    offset_min = rng.integers(0, delta_min + 1, size=n)
    fechas = start + offset_min.astype("timedelta64[m]")

    # Precio unitario fijo por producto/año (con leve incremento anual)
    precios = precios_unitarios_por_producto_y_anio(
        productos=productos, fechas=fechas, seed=seed,
        inc_min=0.01, inc_max=0.03, base_year=2021
    )

    cantidades = rng.integers(1, 6, size=n)  # 1..5
    ciudades = rng.choice(CIUDADES, size=n)
    vendedores = rng.choice(VENDEDORES, size=n)

    estados = rng.choice(["Cerrado", "Abierto", "Cancelado", "En Proceso"],
                         size=n, p=[0.6, 0.2, 0.1, 0.1])
    comisiones_pct = rng.uniform(0.03, 0.08, size=n)

    valor_venta = (precios * cantidades).round(2)
    comision = (valor_venta * comisiones_pct).round(2)

    clientes = rng.choice(CLIENTES_ID, size=n)

    df = pd.DataFrame({
        "CLIENTE_ID": clientes,
        "CLIENTE": np.array([nombre_ficticio(c) for c in clientes], dtype=object),
        "PRODUCTO": productos,
        "PRECIO": precios.round(2),        # precio unitario
        "CANTIDAD": cantidades,
        "CIUDAD": ciudades,
        "VENDEDOR_ID": vendedores,
        "VENDEDOR": np.array([nombre_ficticio(v) for v in vendedores], dtype=object),
        "FECHA": fechas.astype("datetime64[m]").astype(str),
        "ESTADO": estados,
        "VALOR_VENTA": valor_venta,
        "COMISION": comision,
        "CATEGORIA": categorias,
    })

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path

# =========================================================
# Carga normalizada + utilidades de análisis
# =========================================================
def cargar_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    mapping = {
        "VALOR_VENTA": "valor_venta",
        "UTILIDAD": "utilidad",
        "CATEGORIA": "categoria",
        "CIUDAD": "ciudad",
        "VENDEDOR": "vendedor",
        "VENDEDOR_ID": "vendedor_id",
        "CLIENTE": "cliente",
        "CLIENTE_ID": "cliente_id",
        "ESTADO": "estado",
        "PRODUCTO": "producto",
        "FECHA": "fecha",
        "PRECIO": "precio",
        "CANTIDAD": "cantidad",
        "COMISION": "comision",
        # aliases
        "VENDEDOR_NOMBRE": "vendedor",
        "CLIENTE_NOMBRE": "cliente",
    }
    df.rename(columns={k: v for k, v in mapping.items() if k in df.columns}, inplace=True)
    df.columns = [c.lower() for c in df.columns]

    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    # aseguramos valor_venta si no existe
    if "valor_venta" not in df.columns and {"precio","cantidad"} <= set(df.columns):
        df["valor_venta"] = (df["precio"].astype(float) * df["cantidad"].astype(float)).round(2)

    return df

def escribir_agregados(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalización segura
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df["mes"] = df["fecha"].dt.to_period("M").astype(str)
        df["anio"] = df["fecha"].dt.year

    for col in ["valor_venta", "precio", "cantidad", "comision"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    def dump(name: str, dframe: pd.DataFrame):
        dframe.to_csv(out_dir / f"{name}.csv", index=False)

    # ===== Agregados “clásicos” =====
    if {"ciudad","valor_venta"} <= set(df.columns):
        dump("ventas_por_ciudad",
             df.groupby("ciudad", as_index=False)["valor_venta"].sum().sort_values("valor_venta", ascending=False))

    if {"producto","valor_venta"} <= set(df.columns):
        dump("ventas_por_producto",
             df.groupby("producto", as_index=False)["valor_venta"].sum().sort_values("valor_venta", ascending=False))

    if {"estado","valor_venta"} <= set(df.columns):
        dump("ventas_por_estado",
             df.groupby("estado", as_index=False)["valor_venta"].sum().sort_values("valor_venta", ascending=False))

    if {"mes","valor_venta"} <= set(df.columns):
        dump("ventas_mensuales",
             df.groupby("mes", as_index=False)["valor_venta"].sum().sort_values("mes"))

    # ===== Nuevos agregados (para llegar a ~10+) =====
    if {"categoria","valor_venta"} <= set(df.columns):
        dump("ventas_por_categoria",
             df.groupby("categoria", as_index=False)["valor_venta"].sum().sort_values("valor_venta", ascending=False))

    vend_col = "vendedor" if "vendedor" in df.columns else ("vendedor_nombre" if "vendedor_nombre" in df.columns else "vendedor_id" if "vendedor_id" in df.columns else None)
    if vend_col and {"valor_venta", vend_col} <= set(df.columns):
        dump("ventas_por_vendedor",
             df.groupby(vend_col, as_index=False)["valor_venta"].sum().sort_values("valor_venta", ascending=False))

    if {"categoria","ciudad","valor_venta"} <= set(df.columns):
        dump("ventas_por_categoria_ciudad",
             df.groupby(["categoria","ciudad"], as_index=False)["valor_venta"].sum().sort_values("valor_venta", ascending=False))

    if {"producto","ciudad","valor_venta"} <= set(df.columns):
        dump("ventas_por_producto_ciudad",
             df.groupby(["producto","ciudad"], as_index=False)["valor_venta"].sum().sort_values("valor_venta", ascending=False))

    if {"mes","categoria","valor_venta"} <= set(df.columns):
        dump("ventas_mensuales_categoria",
             df.groupby(["mes","categoria"], as_index=False)["valor_venta"].sum().sort_values(["mes","valor_venta"]))

    if {"producto","cantidad"} <= set(df.columns):
        dump("unidades_por_producto",
             df.groupby("producto", as_index=False)["cantidad"].sum().sort_values("cantidad", ascending=False))

    if {"categoria","cantidad"} <= set(df.columns):
        dump("unidades_por_categoria",
             df.groupby("categoria", as_index=False)["cantidad"].sum().sort_values("cantidad", ascending=False))

    # Top 10 (productos y vendedores)
    if {"producto","valor_venta"} <= set(df.columns):
        top_prod = (df.groupby("producto", as_index=False)["valor_venta"].sum()
                      .sort_values("valor_venta", ascending=False).head(10))
        dump("top10_productos", top_prod)

    if vend_col and {"valor_venta", vend_col} <= set(df.columns):
        top_vend = (df.groupby(vend_col, as_index=False)["valor_venta"].sum()
                      .sort_values("valor_venta", ascending=False).head(10))
        dump("top10_vendedores", top_vend)

    # Resumen KPI
    ventas_total = float(df["valor_venta"].sum()) if "valor_venta" in df.columns else 0.0
    unidades_total = int(df["cantidad"].sum()) if "cantidad" in df.columns else 0
    resumen = pd.DataFrame([{
        "ventas_total": round(ventas_total, 2),
        "unidades_total": unidades_total,
        "filas": len(df)
    }])
    dump("kpis_resumen", resumen)


# =========================================================
# CLI
# =========================================================
import argparse

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ventas360 — Generador/Análisis")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generar dataset sintético")
    g.add_argument("--n", type=int, default=5000)
    g.add_argument("--csv", type=Path, default=Path("data/ventas_5k.csv"))

    a = sub.add_parser("analyze", help="Escribir agregados a 'out/'")
    a.add_argument("--csv", type=Path, default=Path("data/ventas_5k.csv"))
    a.add_argument("--out", type=Path, default=Path("out"))
    return p

def main(argv: Optional[List[str]] = None) -> None:
    p = build_parser()
    args = p.parse_args(argv)

    if args.cmd == "generate":
        path = generar_dataset(args.n, args.csv, seed=RANDOM_SEED)
        print(f"[OK] Generado: {path}")
    elif args.cmd == "analyze":
        df = cargar_csv(args.csv)
        escribir_agregados(df, args.out)
        print(f"[OK] Agregados en: {args.out.resolve()}")

if __name__ == "__main__":
    main()
