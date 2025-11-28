# GUÍA RÁPIDA - PRIMEROS PASOS CON BIGQUERY

*Para candidatos de la prueba de Data Analyst*

---

## 1. CONFIGURACIÓN INICIAL (5 minutos)

### Opción A: Google Colab (Recomendado - Sin instalación)
```python
# 1. Ve a https://colab.research.google.com
# 2. Crea un nuevo notebook
# 3. Ejecuta esto:

!pip install google-cloud-bigquery

from google.colab import auth
auth.authenticate_user()

from google.cloud import bigquery
client = bigquery.Client(project='tu-proyecto-gcp')

# Prueba conexión
query = "SELECT 1 as test"
client.query(query).to_dataframe()
```

### Opción B: JupyterLab Local
```bash
pip install google-cloud-bigquery pandas matplotlib seaborn

# Luego en Jupyter:
from google.cloud import bigquery
client = bigquery.Client()  # Usa credenciales por defecto
```

---

## 2. EXPLORAR UN DATASET PÚBLICO

```python
import pandas as pd
from google.cloud import bigquery

client = bigquery.Client()

# Ver primeras filas
query = """
SELECT *
FROM `bigquery-public-data.google_analytics_sample.ga_sessions`
LIMIT 10
"""
df = client.query(query).to_dataframe()
print(df)

# Información del dataset
print(df.info())
print(df.shape)
```

---

## 3. DATASETS PÚBLICOS DISPONIBLES

### Google Analytics Sample
```
bigquery-public-data.google_analytics_sample
```
- Tablas principales: `ga_sessions`, `ga_events`
- Útil para: Ejercicio 1 (SQL)

### COVID-19 ECDC
```
bigquery-public-data.covid19_ecdc
```
- Tabla: `cases_deaths_by_country`
- Columnas: date, country_code, confirmed_cases, deaths

### Calidad del Aire (OpenAQ)
```
bigquery-public-data.openaq.global_air_quality
```
- Tabla: `global_air_quality`
- Columnas: country, city, location, pollutant, value, timestamp

---

## 4. CONSULTAS BÁSICAS SQL EN BIGQUERY

### Estructura general
```sql
SELECT 
  column1,
  COUNT(*) as count,
  AVG(column2) as promedio
FROM `project.dataset.table`
WHERE condition
GROUP BY column1
ORDER BY count DESC
LIMIT 10
```

### Con CTEs (Recomendado para ejercicios complejos)
```sql
WITH base_data AS (
  SELECT 
    id,
    name,
    value
  FROM `bigquery-public-data.dataset.table`
  WHERE date >= '2023-01-01'
),
aggregated AS (
  SELECT 
    name,
    SUM(value) as total,
    COUNT(*) as count
  FROM base_data
  GROUP BY name
)
SELECT 
  name,
  total,
  count,
  total / count as promedio
FROM aggregated
ORDER BY total DESC
```

---

## 5. PYTHON: CONECTAR Y DESCARGAR DATOS

```python
from google.cloud import bigquery
import pandas as pd

# Conectar
client = bigquery.Client()

# Query
query = """
SELECT 
  country,
  COUNT(*) as total_sessions,
  SUM(totals.transactionRevenue) / 1e6 as revenue_usd
FROM `bigquery-public-data.google_analytics_sample.ga_sessions`
GROUP BY country
ORDER BY revenue_usd DESC
LIMIT 10
"""

# Descargar como DataFrame
df = client.query(query).to_dataframe()

# Explorar
print(df.head())
print(df.describe())
print(df.dtypes)

# Guardar si necesitas
df.to_csv('resultado.csv', index=False)
```

---

## 6. ANÁLISIS EXPLORATORIO (EDA) TEMPLATE

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Datos ya en df

# 1. INFO BÁSICA
print(f"Filas: {len(df)}, Columnas: {len(df.columns)}")
print(df.info())
print(df.describe())

# 2. VALORES NULOS
print(df.isnull().sum())

# 3. DUPLICADOS
print(f"Duplicados: {df.duplicated().sum()}")

# 4. DISTRIBUCIONES
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df['numeric_col1'], bins=30)
axes[0].set_title('Distribución de Variable 1')
axes[1].hist(df['numeric_col2'], bins=30)
axes[1].set_title('Distribución de Variable 2')
plt.show()

# 5. CORRELACIONES
print(df.corr())
sns.heatmap(df.corr(), annot=True)
plt.show()

# 6. TOP VALORES
print(df['categorical_col'].value_counts().head(10))
```

---

## 7. TRANSFORMACIÓN DE DATOS

```python
# Cambiar tipos
df['fecha'] = pd.to_datetime(df['fecha'])

# Crear nuevas columnas
df['mes'] = df['fecha'].dt.month
df['año'] = df['fecha'].dt.year

# Renombrar
df = df.rename(columns={'old_name': 'new_name'})

# Filtrar
df_filtrado = df[df['valor'] > 100]

# Groupby
resumen = df.groupby('categoria').agg({
    'valor': ['sum', 'mean', 'count']
}).round(2)
print(resumen)

# Media móvil
df_sorted = df.sort_values('fecha')
df_sorted['media_movil_7'] = df_sorted['valor'].rolling(window=7).mean()

# Eliminar duplicados
df_clean = df.drop_duplicates(subset=['id'])

# Valores nulos
df_clean = df.dropna(subset=['columna_importante'])
# O rellenar
df_filled = df.fillna(method='ffill')  # Forward fill
```

---

## 8. VISUALIZACIONES ÚTILES

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo
sns.set_style("whitegrid")

# 1. HISTOGRAMA
plt.figure(figsize=(10, 6))
plt.hist(df['valor'], bins=50, color='steelblue', edgecolor='black')
plt.title('Distribución de Valores')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.show()

# 2. SCATTER
plt.figure(figsize=(10, 6))
plt.scatter(df['x'], df['y'], alpha=0.5)
plt.title('Relación X vs Y')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# 3. BAR CHART
plt.figure(figsize=(12, 6))
df_top = df.nlargest(10, 'valor')
plt.bar(df_top['nombre'], df_top['valor'])
plt.xticks(rotation=45)
plt.title('Top 10 por Valor')
plt.show()

# 4. LÍNEA (Series temporal)
plt.figure(figsize=(12, 6))
df_sorted = df.sort_values('fecha')
plt.plot(df_sorted['fecha'], df_sorted['valor'], marker='o')
plt.title('Tendencia en el Tiempo')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.xticks(rotation=45)
plt.show()

# 5. BOX PLOT
plt.figure(figsize=(10, 6))
df.boxplot(column='valor', by='categoria')
plt.title('Distribución por Categoría')
plt.show()

# 6. HEATMAP DE CORRELACIÓN
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Matriz de Correlación')
plt.show()
```

---

## 9. ERRORES COMUNES Y SOLUCIONES

### Error: "Project ID is required"
```python
# Solución: Especificar proyecto
client = bigquery.Client(project='mi-proyecto-gcp')
```

### Error: "Time out"
```sql
-- Problema: Query muy pesada
-- Solución: Agregar LIMIT o usar materialised view
SELECT * FROM tabla LIMIT 1000

-- O optimizar la query con EXCEPT DISTINCT, etc
```

### Error: "Cannot find dataset"
```python
# Verificar la ruta completa:
# `proyecto.dataset.tabla`

# Ejemplo correcto:
FROM `bigquery-public-data.google_analytics_sample.ga_sessions`
```

### Valores NaN después de merge
```python
# Problema: LEFT JOIN descarta algunas filas
df_merged = df1.merge(df2, on='key', how='left')

# Solución: Revisar la key
print(df1['key'].isin(df2['key']).sum())
```

---

## 10. CHECKLIST ANTES DE ENTREGAR

- [ ] Código SQL ejecuta sin errores
- [ ] Código Python ejecuta sin errores
- [ ] Resultados tienen sentido (números razonables)
- [ ] He respondido TODAS las preguntas
- [ ] Los comentarios explican el por qué, no el qué
- [ ] Las visualizaciones tienen títulos y labels
- [ ] Revisualizó en una máquina limpia (sin variables previas)
- [ ] Documento está bien formateado y profesional
- [ ] He incluido observaciones/insights propios, no genéricos

---

## 11. RECURSOS ÚTILES

**Documentación:**
- https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax
- https://pandas.pydata.org/docs/
- https://matplotlib.org/stable/api/pyplot_api.html

**Ejemplos:**
- https://github.com/GoogleCloudPlatform/bigquery-samples
- https://www.kaggle.com/kernels (búsca "bigquery")

**Debugging:**
- Error? Copia el mensaje completo en Google
- BigQuery rechaza tu query? Copia la parte de la query en el editor BigQuery directamente
- Python error? Ejecuta en una celda limpia sin variables previas

---

## 12. TIPS FINALES

✅ **Haz esto:**
- Empieza simple, luego complejidad
- Documenta tus decisiones (¿POR QUÉ hiciste eso?)
- Verifica resultados (¿Tienen sentido los números?)
- Toma capturas de pantalla de las visualizaciones
- Haz backup de tu código (.py o .ipynb)

❌ **No hagas esto:**
- No hagas queries sin LIMIT si no sabes cuántos datos hay
- No uses IA para generar el código entero
- No copies ejemplos sin entenderlos
- No ignores errores, entiéndelos
- No entregues código "feo" sin comentarios

---

**¿Dudas?** Revisa la documentación oficial o pregunta antes de la prueba.

*¡Mucha suerte!* 📊
