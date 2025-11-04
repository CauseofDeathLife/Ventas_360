# 💼 Ventas360 — Analítica

Aplicación en **Python + Streamlit + Pandas** para generar, analizar y visualizar datos de ventas simuladas. Forma parte de la suite analítica *360* junto a Asistencia360.

## 🚀 Demo en vivo
**Ventas360 Analytics:** [ventas360-analytics](https://ventas360-analytics.streamlit.app)

## 🧩 Instalación y ejecución local
```bash
git clone https://github.com/CauseofDeathLife/Ventas_360.git
cd Ventas_360
pip install -r requirements.txt
streamlit run ventas360_app.py
```

## 📊 Características
- Generación de datasets sintéticos con `ventas360_cli.py`.
- Dashboard interactivo con filtros por fecha, vendedor, ciudad, estado y producto.
- KPIs de ventas, comisiones y utilidad.
- Exportación de agregados a CSV/Excel.

## 📁 Estructura
```
Ventas_360/
├── data/                     # CSV generados
├── out/                      # Reportes/Agregados
├── ventas360_cli.py          # CLI: generate/analyze
├── ventas360_app.py          # App principal de Streamlit
├── requirements.txt
└── README.md
```

## 🧠 Uso de la CLI

**Generar dataset**
```bash
python ventas360_cli.py generate --n 5000 --csv data/ventas_5k.csv
```

**Analizar dataset**
```bash
python ventas360_cli.py analyze --csv data/ventas_5k.csv --out out
```

## ✅ Notas de lógica
- KPIs por defecto consideran **solo** ventas con `estado = "Cerrado"`.
- La **comisión** se contabiliza solo para ventas cerradas.
- Si no existe la columna `utilidad`, se calcula como **5%** de `valor_venta`.
- Generación reproducible mediante `RANDOM_SEED`.

## 👤 Autor
**Daniel Esteban Quintero Caro** — [GitHub: CauseofDeathLife](https://github.com/CauseofDeathLife)
