# Demo funcional en Streamlit

Aplicación web que permite usar el modelo de detección de enfermedad hepática de dos
formas: **predicción individual** (un paciente, mediante formulario) y **procesamiento por
lotes** (un archivo con muchos pacientes, con descarga de resultados).

> Herramienta académica de apoyo. El resultado no constituye un diagnóstico médico y
> requiere validación clínica independiente.

---

## Enlace a la aplicación publicada

<!-- Pegue aquí la URL de Streamlit Community Cloud tras el despliegue -->
**URL pública:** _(pendiente de despliegue)_

---

## Ejecución local

Desde la raíz del repositorio:

```bash
uv sync --all-groups
uv run streamlit run app.py
```

La aplicación queda disponible en <http://localhost:8501>.

Para depurar con trazas completas (en producción están ocultas a propósito):

```bash
uv run streamlit run app.py --client.showErrorDetails=full
```

---

## Instrucciones de uso

### Pestaña 1 — Predicción individual

1. Opcionalmente elija un caso de ejemplo en **Cargar un caso de ejemplo** para rellenar el
   formulario con un perfil de referencia.
2. Complete las diez variables clínicas. Cada campo está acotado al rango clínico admitido
   por el proyecto, de modo que no es posible ingresar un valor imposible.
3. Pulse **Realizar predicción**.
4. La aplicación muestra el veredicto, la probabilidad estimada de enfermedad, la barra de
   riesgo, las variables derivadas que el modelo calcula internamente y, si corresponde, las
   advertencias aplicables.
5. Con **Descargar este resultado (CSV)** obtiene la fila de predicción en el mismo formato
   que el procesamiento por lotes.

### Pestaña 2 — Procesamiento por lotes

1. Descargue **la plantilla vacía** o **el archivo de ejemplo** con los botones superiores.
2. Suba su archivo con **Archivo con los pacientes**. Se admiten `.csv` y `.parquet`, hasta
   5 MB y 5 000 filas.
3. Revise la vista previa y el informe de validación: los errores en rojo impiden predecir y
   deben corregirse; las advertencias en amarillo no impiden el cálculo, pero señalan filas
   cuyo resultado es menos confiable.
4. Consulte las métricas agregadas y las dos gráficas (distribución de diagnósticos e
   histograma de probabilidades).
5. Pulse **Descargar predicciones (CSV)** para obtener el archivo completo de resultados.

### Pestaña 3 — Información del modelo

Ficha técnica del artefacto cargado: algoritmo, fecha de generación, métricas de evaluación,
matriz de confusión, gráficas diagnósticas del entrenamiento, variables de entrada y
limitaciones conocidas.

---

## Formato del archivo por lotes

Una fila por paciente y estas diez columnas obligatorias:

| Columna | Variable | Unidad | Valores admitidos |
| --- | --- | --- | --- |
| `age` | Edad | años | 1 a 120 |
| `gender` | Género | - | `Male` o `Female` |
| `total_bilirubin` | Bilirrubina total | mg/dL | 0 a 100 |
| `direct_bilirubin` | Bilirrubina directa | mg/dL | 0 a 50 |
| `alkaline_phosphotase` | Fosfatasa alcalina | UI/L | 10 a 3000 |
| `alamine_aminotransferase` | ALT | UI/L | 1 a 3000 |
| `aspartate_aminotransferase` | AST | UI/L | 1 a 6000 |
| `total_protiens` | Proteínas totales | g/dL | 1 a 15 |
| `albumin` | Albúmina | g/dL | 0.5 a 10 |
| `albumin_and_globulin_ratio` | Razón albúmina/globulina | - | 0 a 10 |

La aplicación es tolerante con el formato de entrada:

- Reconoce los encabezados tanto en `Total_Bilirubin` como en `total_bilirubin`.
- Detecta automáticamente los separadores `,`, `;`, tabulador y `|`.
- Acepta codificación UTF-8 (con o sin BOM) y latin-1.
- Interpreta la coma decimal (`0,85` se lee como `0.85`) y lo informa.
- Normaliza los alias de género más frecuentes (`M`, `masculino`, `hombre`, `mujer`, …).
- Descarta las columnas de relleno vacías y las filas totalmente vacías.
- Conserva la columna `patient_id` en la salida para mantener la trazabilidad.
- Descarta la columna de diagnóstico real (`Dataset` o `Diagnosis`) si viene incluida.

### Columnas de la salida

A las columnas de entrada se añaden:

| Columna | Contenido |
| --- | --- |
| `prediction` | `1` enfermo, `0` sano |
| `prediction_label` | `enfermo` o `sano` |
| `probability_disease` | Probabilidad estimada de enfermedad hepática |
| `probability_no_disease` | Probabilidad estimada de ausencia de enfermedad |
| `advertencias` | Variables que motivaron una advertencia en esa fila |

Es el mismo esquema que produce el Inference Pipeline por línea de comandos, de modo que
los resultados de la aplicación y los del CLI son intercambiables.

---

## Archivos de este directorio

| Archivo | Contenido |
| --- | --- |
| `plantilla_lote.csv` | Plantilla vacía con las columnas obligatorias y dos filas de ejemplo |
| `ejemplo_entrada_lote.csv` | Entrada de ejemplo: 24 pacientes reales del dataset original |
| `ejemplo_salida_predicciones.csv` | Salida correspondiente a la entrada de ejemplo |
| `ejemplo_entrada_con_errores.csv` | Entrada con defectos deliberados, para probar el manejo de errores |
| `demo_report.html` | Reporte autocontenido con la evidencia de funcionamiento |
| `demo_summary.json` | Resumen de la generación, con rutas relativas al proyecto |
| `inference_report.html` | Reporte del Inference Pipeline sobre la entrada de ejemplo |
| `inference_summary.json` | Resumen del Inference Pipeline |
| `images/prediction_distribution.png` | Distribución de las predicciones del lote de ejemplo |
| `images/capturas/` | Capturas de pantalla de la aplicación desplegada |

Todos se regeneran de forma reproducible con:

```bash
uv run python -m src.inference.demo_assets
```

---

## Evidencia de funcionamiento

### Manejo de entradas problemáticas

`demo_report.html` documenta la ejecución real de nueve casos límite contra el servicio.
En ninguno se propaga una excepción: el usuario siempre recibe un mensaje en español.

| Entrada | Comportamiento |
| --- | --- |
| Archivo válido | Genera predicciones |
| Separador de punto y coma | Se detecta y se procesa |
| Columna obligatoria ausente | Error controlado, indica qué columna falta |
| Valor no numérico | Error controlado, indica la fila y la columna |
| Género no reconocido | Predice y advierte de forma destacada |
| Archivo sin filas de datos | Error controlado |
| Archivo vacío | Error controlado |
| Extensión no soportada | Error controlado, indica los formatos admitidos |
| Contenido ilegible | Error controlado, explica el formato esperado |

`ejemplo_entrada_con_errores.csv` reúne seis filas, cada una con un defecto distinto
(género en español, género desconocido, edad fuera de rango, albúmina vacía, bilirrubina
total en cero y coma decimal), pensadas para reproducir estos casos desde la interfaz.

### Pruebas automatizadas

El requisito de que la interfaz no falle está verificado por pruebas que ejecutan la
aplicación real en modo headless con `streamlit.testing.v1.AppTest`: rellenan el formulario,
pulsan los botones y suben archivos, comprobando en cada caso que no se propaga ninguna
excepción y que aparece el mensaje esperado.

```bash
uv run pytest tests/inference -q
```

### Capturas de pantalla

<!-- Añada las capturas en images/capturas/ y vuelva a ejecutar demo_assets para incrustarlas -->

| Archivo | Qué debe mostrar |
| --- | --- |
| `images/capturas/01_prediccion_individual.png` | Formulario completo y resultado con probabilidad |
| `images/capturas/02_lote_resultado.png` | Métricas, gráficas y tabla de predicciones del lote |
| `images/capturas/03_lote_con_errores.png` | Informe de validación con errores y advertencias |
| `images/capturas/04_informacion_modelo.png` | Ficha técnica y métricas del modelo |

Tras guardarlas, vuelva a ejecutar `uv run python -m src.inference.demo_assets` para que
queden incrustadas en `demo_report.html` como data URI base64.

---

## Despliegue en Streamlit Community Cloud

1. **Verifique que estos archivos estén en la rama publicada**:
   - `app.py` (punto de entrada)
   - `src/inference/` y `src/model/preprocessing.py` (este último es imprescindible: joblib
     lo necesita para reconstruir los transformadores del pipeline)
   - `models/liver_patient_model/model.joblib`
   - `requirements.txt` y `.streamlit/config.toml`
2. **Ejecute las comprobaciones** antes de publicar:

   ```bash
   uv run pre-commit run --all-files
   uv run pytest --cov
   ```

3. **Haga commit y push** de la rama.
4. Entre a <https://share.streamlit.io> e inicie sesión con la cuenta de GitHub.
5. Pulse **New app** y seleccione el repositorio y la rama.
6. En **Main file path** indique `app.py`.
7. En **Advanced settings** seleccione **Python 3.12**.
8. Pulse **Deploy** y revise el log de construcción: debe instalar las dependencias desde
   `requirements.txt`.
9. Pruebe la aplicación publicada con los casos descritos arriba.
10. Copie la URL pública en este archivo y en el `README.md` del repositorio, y guarde las
    capturas de pantalla.

### Notas de despliegue

- `requirements.txt` fija las versiones de forma exacta a propósito. El modelo es un pickle
  de scikit-learn: una versión distinta a la que lo serializó puede impedir la carga o
  alterar las predicciones en silencio. Si actualiza alguna dependencia, vuelva a entrenar
  el modelo y vuelva a fijar las versiones.
- La aplicación no requiere secretos ni variables de entorno: el modelo viaja en el
  repositorio como artefacto local de 7 KB.
- La aplicación no escribe en disco en tiempo de ejecución; las descargas se generan en
  memoria. Esto es necesario porque el sistema de archivos de Streamlit Cloud es efímero.
- `.streamlit/config.toml` limita las subidas a 5 MB y oculta las trazas técnicas al usuario
  final.
