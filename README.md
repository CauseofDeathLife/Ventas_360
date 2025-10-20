
# Ventas 360 — Guía de uso y despliegue (Proyecto Final)

Este README **actualizado** explica cómo llevar el proyecto a otro computador y ejecutarlo **tal cual**: generación de datos y reportes con la **CLI** y visualización de KPIs con el **dashboard en Streamlit**.  
El proyecto ya incluye un dataset base (`data/ventas_5k.csv`) y salidas de ejemplo en `out/`.

> **Nota clave**  
> - La **generación sintética** produce fechas **entre 2021-01-01 y 2025-10-19** (sin futuros) y montos en **COP** con **dos decimales**.  
> - La app muestra valores en formato $ **COP**, acorde a tu versión final.  
> - **No se elimina ninguna interfaz**: el flujo y la UI actual se mantienen; este README solo documenta el uso.

---

## 📦 Estructura del proyecto

```
Ventas_360/
├─ data/
│  └─ ventas_5k.csv                # dataset base (5.000 filas)
├─ out/                            # exportaciones de ejemplo (se pueden regenerar)
│  ├─ cerradas_por_vendedor_y_ciudad.csv
│  ├─ dataset_con_utilidad.csv
│  ├─ Informe_Analitico_Ventas_Pandas.docx
│  ├─ media_valor_venta_por_mes.csv
│  ├─ productos_mas_de_3_ciudades.csv
│  ├─ promedio_comision_por_vendedor.csv
│  ├─ resumen.txt
│  ├─ top5_productos_por_registros.csv
│  ├─ utilidad_total_por_producto.csv
│  ├─ valor_total_por_categoria.csv
│  ├─ ventas_mensuales_por_ciudad.csv
│  ├─ ventas_por_ciudad_sum.csv
│  ├─ ventas_por_estado.csv
│  ├─ ventas_por_estado_y_ciudad.csv
│  └─ ventas_por_trimestre.csv
├─ ventas360_cli.py                # CLI: generar datos, analizar y exportar
├─ ventas360_app.py                # Dashboard Streamlit (KPIs, tablas y gráficos Altair)
├─ requirements.txt                # deps completas (CLI + Dashboard)
├─ requirements_streamlit.txt      # deps mínimas para solo Dashboard
└─ .streamlit/config.toml          # configuración de Streamlit (tema/puerto, etc.)
```

---

## 🧰 Requisitos

- **Python 3.10+** (recomendado)
- **Windows**, **macOS** o **Linux**
- (Opcional para Excel/Parquet): `openpyxl`, `pyarrow`

> Instala dependencias en **entorno virtual**. Hay dos opciones:
> - Completo (CLI + Dashboard): `requirements.txt`
> - Solo Dashboard: `requirements_streamlit.txt`

---

## 🚀 Instalación rápida (paso a paso)

### Windows (PowerShell)

```powershell
# 1) Entrar a la carpeta del proyecto
cd <ruta>\Ventas_360

# 2) Crear entorno virtual
py -m venv .venv

# 3) (Si da error de permisos) Habilitar scripts en PowerShell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 4) Activar el entorno
. .venv\Scripts\Activate.ps1

# 5) Instalar dependencias
pip install -r requirements.txt
# (alternativa si solo quieres el dashboard)
# pip install -r requirements_streamlit.txt
```

### macOS / Linux (bash/zsh)

```bash
# 1) Entrar a la carpeta del proyecto
cd /ruta/a/Ventas_360

# 2) Crear entorno virtual
python3 -m venv .venv

# 3) Activar el entorno
source .venv/bin/activate

# 4) Instalar dependencias
pip install -r requirements.txt
# (alternativa si solo quieres el dashboard)
# pip install -r requirements_streamlit.txt
```

> **Sugerencia**: actualiza instaladores si algo falla al compilar ruedas/binaries
> ```bash
> pip install --upgrade pip setuptools wheel
> ```

---

## 📊 Dashboard (Streamlit)

Ejecuta la aplicación web:

```bash
# dentro del entorno virtual y dentro de la carpeta del proyecto
streamlit run ventas360_app.py
```

- La app **carga automáticamente** datasets de `out/` (y también de `data/` si existen).
- Los montos se muestran con símbolo **$** y **dos decimales**, como **COP** (formato 1.234,56 → $ 1.234,56).
- Incluye filtros, KPIs, tabla dinámica y gráficos con **Altair**.

---

## 🧪 CLI: generar datos y exportar informes

La **CLI** procesa `data/ventas_5k.csv` (o el CSV que especifiques) y exporta resultados a `out/`.

### Comandos disponibles

```bash
# Generar dataset sintético (por defecto 5.000 filas)
python ventas360_cli.py --generate --n 5000 --csv data/ventas_5k.csv

# Analizar y exportar resultados (CSVs + resumen)
python ventas360_cli.py --analyze --csv data/ventas_5k.csv --out out

# (Opcional) Generar también el informe administrativo en Word
python ventas360_cli.py --analyze --csv data/ventas_5k.csv --out out --word

# Flujo completo en un solo paso: generar + analizar + informe Word
python ventas360_cli.py --generate --analyze --n 5000 --csv data/ventas_5k.csv --out out --word
```

**Qué se exporta en `out/`:**

- `dataset_con_utilidad.csv` – base con columnas calculadas (incluye *utilidad* del 5% a modo KPI).
- `resumen.txt` – resumen de métricas clave.
- `Informe_Analitico_Ventas_Pandas.docx` – reporte administrativo (si usas `--word`).  
- Agregaciones por categoría, estado, ciudad, vendedor, mes y trimestre:
  - `valor_total_por_categoria.csv`
  - `promedio_comision_por_vendedor.csv`
  - `cerradas_por_vendedor_y_ciudad.csv`
  - `ventas_por_estado.csv`, `ventas_por_estado_y_ciudad.csv`
  - `ventas_por_ciudad_sum.csv`
  - `media_valor_venta_por_mes.csv`
  - `ventas_mensuales_por_ciudad.csv`
  - `ventas_por_trimestre.csv`
  - `top5_productos_por_registros.csv`
  - `productos_mas_de_3_ciudades.csv`

> **Rango de fechas** en la generación sintética: **2021-01-01 → hoy** (no hay fechas futuras).  
> **Moneda**: montos en **COP** con dos decimales (p. ej. `PRECIO`, `VALOR_VENTA`, `COMISION`).

---

## 🧭 Guía para “probar rápido” en otro computador

1. Copia la carpeta `Ventas_360/` tal cual.
2. Crea y activa el **entorno virtual** (ver secciones de instalación).
3. `pip install -r requirements.txt`
4. (Opcional) Regenera salidas con la CLI:
   ```bash
   python ventas360_cli.py --generate --analyze --n 5000 --csv data/ventas_5k.csv --out out --word
   ```
5. Arranca el dashboard:
   ```bash
   streamlit run ventas360_app.py
   ```

---

## 🛠️ Solución de problemas comunes

- **“streamlit no se reconoce como un comando”**  
  Activa el entorno virtual correcto y asegúrate de haber instalado dependencias:
  ```powershell
  # Windows
  . .venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  streamlit run ventas360_app.py
  ```
  En VS Code, selecciona el intérprete de `.venv` (barra de estado inferior).

- **Error al instalar `pandas` en Windows**  
  Actualiza herramientas de build y ruedas:
  ```bash
  pip install --upgrade pip setuptools wheel
  ```
  Si persiste, instala una versión compatible de Python 3.10–3.12.

- **Caracteres extraños en CSV (tildes/ñ)**  
  Asegúrate de guardar tus CSV en **UTF-8**. Puedes intentar:
  ```python
  df.to_csv("archivo.csv", index=False, encoding="utf-8")
  ```

- **La app no muestra datasets**  
  Verifica que existan archivos en `out/` o `data/`. Vuelve a ejecutar la CLI para generarlos.

---

## 🧾 Notas de negocio (resumen)

- **Fechas** normalizadas a tipo `datetime`; filas sin fecha válida se excluyen de agregaciones temporales.
- Validación de valores no positivos en `VALOR_VENTA` y `COMISION`.
- **Utilidad**: se documenta como 5% del `VALOR_VENTA` para fines de KPI del tablero e informe.

---

## 📄 Licencia

MIT
