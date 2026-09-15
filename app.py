"""Demo funcional en Streamlit del modelo de detección de enfermedad hepática.

La aplicación cubre los dos modos de uso exigidos por el despliegue:

* **Predicción individual**: formulario con las variables clínicas de un paciente.
* **Procesamiento por lotes**: carga de un archivo con múltiples pacientes, visualización
  de los resultados y descarga de las predicciones.

Este archivo es sólo la capa de presentación. Toda la lógica (carga del modelo, lectura
defensiva del archivo, validación y predicción) vive en `src/inference/`, que se prueba con
pytest y se mide con coverage. Cada pestaña se renderiza dentro de su propio bloque
`try/except`, de modo que un fallo inesperado muestre un mensaje en español y nunca un
traceback, y no impida usar las demás pestañas.

Uso local:
    uv run streamlit run app.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import app_service as service  # noqa: E402
from src.inference import batch_validation as validation  # noqa: E402

PAGE_TITLE = "Predicción de enfermedad hepática"
PAGE_ICON = "🩺"
MODEL_IMAGES_DIR = PROJECT_ROOT / "models" / "liver_patient_model" / "images"
DEMO_DIR = PROJECT_ROOT / "models" / "liver_patient_model" / "demo"
EXAMPLE_BATCH_PATH = DEMO_DIR / "ejemplo_entrada_lote.csv"
FAULTY_BATCH_PATH = DEMO_DIR / "ejemplo_entrada_con_errores.csv"

WIDGET_PREFIX = "paciente_"
PRESET_KEY = "caso_ejemplo"
UPLOAD_KEY = "archivo_lote"
IMPUTE_KEY = "imputar_no_numericos"
ONLY_WARNED_KEY = "solo_filas_con_advertencias"

MAX_INLINE_ISSUES = 6
PREVIEW_ROWS = 20
DETAIL_MESSAGE_LIMIT = 500
HIGH_RISK_THRESHOLD = 0.5

DISCLAIMER = (
    "Herramienta académica de apoyo. El resultado no constituye un diagnóstico médico y "
    "requiere validación clínica independiente."
)

PRESETS: dict[str, dict[str, float | str]] = {
    "Perfil clínico intermedio": dict(service.SINGLE_RECORD_DEFAULTS),
    "Perfil sin alteración hepática": {
        "age": 26.0,
        "gender": "Female",
        "total_bilirubin": 0.9,
        "direct_bilirubin": 0.2,
        "alkaline_phosphotase": 154.0,
        "alamine_aminotransferase": 16.0,
        "aspartate_aminotransferase": 12.0,
        "total_protiens": 7.0,
        "albumin": 3.5,
        "albumin_and_globulin_ratio": 1.0,
    },
    "Perfil con alteración hepática": {
        "age": 62.0,
        "gender": "Male",
        "total_bilirubin": 10.9,
        "direct_bilirubin": 5.5,
        "alkaline_phosphotase": 699.0,
        "alamine_aminotransferase": 64.0,
        "aspartate_aminotransferase": 100.0,
        "total_protiens": 7.5,
        "albumin": 3.2,
        "albumin_and_globulin_ratio": 0.74,
    },
}


def widget_key(feature_name: str) -> str:
    """Construye la clave estable del widget asociado a una variable clínica."""
    return f"{WIDGET_PREFIX}{feature_name}"


def initialize_state() -> None:
    """Inicializa el estado del formulario con los valores por defecto."""
    for name, value in service.SINGLE_RECORD_DEFAULTS.items():
        st.session_state.setdefault(widget_key(name), value)
    st.session_state.setdefault(PRESET_KEY, next(iter(PRESETS)))


def apply_preset() -> None:
    """Vuelca en el formulario los valores del caso de ejemplo seleccionado."""
    preset = PRESETS.get(str(st.session_state.get(PRESET_KEY, "")))
    if not preset:
        return
    for name, value in preset.items():
        st.session_state[widget_key(name)] = value


def render_error(message: str, error: Exception | None = None) -> None:
    """Muestra un error en español y esconde el detalle técnico en un desplegable."""
    st.error(message, icon="🚫")
    if error is not None:
        with st.expander("Detalle técnico"):
            st.code(f"{type(error).__name__}: {str(error)[:DETAIL_MESSAGE_LIMIT]}")


def collect_form_values() -> dict[str, float | str]:
    """Lee del estado los valores actuales del formulario individual."""
    return {
        name: st.session_state[widget_key(name)]
        for name in service.REQUIRED_FEATURE_COLUMNS
        if widget_key(name) in st.session_state
    }


def render_single_form() -> bool:
    """Dibuja el formulario del paciente y devuelve True si se solicitó una predicción."""
    with st.form("formulario_paciente"):
        columns = st.columns(3)
        specs = service.feature_input_specs()

        with columns[0]:
            st.selectbox(
                validation.label_for("gender"),
                options=list(service.valid_gender_values()),
                format_func=lambda value: service.GENDER_DISPLAY.get(value, value),
                key=widget_key("gender"),
                help="Género biológico registrado en la historia clínica.",
            )

        for index, spec in enumerate(specs):
            with columns[(index + 1) % len(columns)]:
                st.number_input(
                    spec.label,
                    min_value=spec.minimum,
                    max_value=spec.maximum,
                    step=spec.step,
                    key=widget_key(spec.name),
                    help=spec.help_text,
                )

        submitted = st.form_submit_button("Realizar predicción", type="primary", width="stretch")
        return bool(submitted)


def render_single_result(result: service.SinglePrediction) -> None:
    """Muestra el veredicto, las probabilidades y los avisos de una predicción individual."""
    st.divider()
    if result.prediction == 1:
        st.error(f"Resultado del modelo: {result.label.lower()}", icon="⚠️")
    else:
        st.success(f"Resultado del modelo: {result.label.lower()}", icon="✅")

    metrics = st.columns(3)
    probability = result.probability_disease
    metrics[0].metric(
        "Probabilidad de enfermedad",
        f"{probability:.1%}" if probability is not None else "No disponible",
    )
    metrics[1].metric(
        "Probabilidad de ausencia",
        f"{result.probability_no_disease:.1%}"
        if result.probability_no_disease is not None
        else "No disponible",
    )
    metrics[2].metric("Clase predicha", result.label)

    if probability is not None:
        st.progress(min(max(probability, 0.0), 1.0), text="Riesgo estimado por el modelo")

    for message in result.warnings:
        st.warning(message, icon="⚠️")

    with st.expander("Variables clínicas derivadas que usa el modelo"):
        st.caption(
            "El pipeline calcula estos cocientes automáticamente a partir de los valores "
            "ingresados. Un valor no disponible se imputa con la mediana del entrenamiento."
        )
        for name, value in result.derived_features.items():
            st.write(
                f"**{name}**: {value:.4f}" if value is not None else f"**{name}**: no definido"
            )

    st.download_button(
        "Descargar este resultado (CSV)",
        data=service.predictions_to_csv_bytes(service.order_prediction_columns(result.frame)),
        file_name=f"prediccion_individual_{_timestamp()}.csv",
        mime="text/csv",
        key="descarga_individual",
    )
    st.caption(
        f"El modelo clasifica como enfermo cuando la probabilidad supera "
        f"{HIGH_RISK_THRESHOLD:.0%}. Fue ajustado priorizando la precisión sobre la "
        f"sensibilidad, de modo que puede pasar por alto casos positivos."
    )


def render_single_tab(bundle: service.ModelBundle | None, load_error: str | None) -> None:
    """Renderiza la pestaña de predicción individual."""
    st.subheader("Datos clínicos del paciente")
    st.caption(
        "Los límites de cada campo corresponden a los rangos clínicos admitidos por el "
        "proyecto, por lo que no es posible ingresar un valor imposible."
    )

    st.selectbox(
        "Cargar un caso de ejemplo",
        options=list(PRESETS),
        key=PRESET_KEY,
        on_change=apply_preset,
        help="Rellena el formulario con un perfil de referencia para probar la demo.",
    )

    submitted = render_single_form()

    if bundle is None:
        render_error(load_error or "El modelo no está disponible.")
        return
    if not submitted:
        return

    try:
        result = service.predict_single(bundle, collect_form_values())
    except service.AppInferenceError as error:
        render_error(str(error), error)
    except Exception as error:
        render_error("Ocurrió un problema inesperado al calcular la predicción individual.", error)
    else:
        render_single_result(result)


def render_batch_instructions() -> None:
    """Muestra las instrucciones y las descargas de plantilla y de archivo de ejemplo."""
    st.subheader("Predicción para varios pacientes")
    st.markdown(
        "Suba un archivo **CSV** (o **Parquet**) con una fila por paciente. Los nombres de "
        "las columnas se reconocen tanto en formato `Total_Bilirubin` como "
        "`total_bilirubin`, y se aceptan separadores `,` o `;`. Si incluye una columna "
        "`patient_id`, se conservará en el resultado para mantener la trazabilidad."
    )
    downloads = st.columns(2)
    downloads[0].download_button(
        "Descargar plantilla vacía",
        data=service.template_csv_bytes(),
        file_name="plantilla_lote.csv",
        mime="text/csv",
        width="stretch",
        key="descarga_plantilla",
    )
    if EXAMPLE_BATCH_PATH.exists():
        downloads[1].download_button(
            "Descargar archivo de ejemplo",
            data=EXAMPLE_BATCH_PATH.read_bytes(),
            file_name=EXAMPLE_BATCH_PATH.name,
            mime="text/csv",
            width="stretch",
            key="descarga_ejemplo",
        )


def render_validation_report(report: validation.ValidationReport) -> None:
    """Muestra los errores y las advertencias detectadas en el archivo cargado."""
    for issue in report.errors[:MAX_INLINE_ISSUES]:
        st.error(issue.message, icon="🚫")
    for issue in report.warnings[:MAX_INLINE_ISSUES]:
        st.warning(issue.message, icon="⚠️")

    total = len(report.errors) + len(report.warnings)
    if total > MAX_INLINE_ISSUES:
        with st.expander(f"Ver el informe completo de validación ({total} hallazgos)"):
            st.dataframe(report.to_frame(), hide_index=True, width="stretch")


def render_batch_summary(result: service.BatchResult) -> None:
    """Muestra las métricas agregadas y las gráficas del lote procesado."""
    metrics = st.columns(4)
    metrics[0].metric("Pacientes procesados", result.n_rows)
    metrics[1].metric("Con enfermedad hepática", result.n_positive)
    metrics[2].metric("Sin enfermedad hepática", result.n_negative)
    metrics[3].metric("Proporción positiva", f"{result.positive_ratio:.1%}")

    mean_probability = result.mean_probability
    if mean_probability is not None:
        st.caption(f"Probabilidad media estimada de enfermedad: {mean_probability:.1%}")

    charts = st.columns(2)
    with charts[0]:
        st.markdown("**Distribución de diagnósticos**")
        st.bar_chart(
            service.label_counts_frame(result.predictions),
            x="diagnóstico",
            y="pacientes",
            color="diagnóstico",
        )
    with charts[1]:
        st.markdown("**Probabilidad estimada de enfermedad**")
        histogram = service.probability_histogram_frame(result.predictions)
        if histogram.empty:
            st.info("El modelo cargado no proporciona probabilidades.")
        else:
            st.bar_chart(histogram, x="rango", y="pacientes")


def render_batch_predictions(result: service.BatchResult) -> None:
    """Muestra la tabla de predicciones y el botón de descarga."""
    ordered = service.order_prediction_columns(result.predictions)
    only_warned = st.checkbox(
        "Mostrar sólo las filas con advertencias", key=ONLY_WARNED_KEY, value=False
    )
    displayed = ordered
    if only_warned and service.WARNING_COLUMN in ordered.columns:
        displayed = ordered[ordered[service.WARNING_COLUMN].astype(bool)]
        if displayed.empty:
            st.info("Ninguna fila tiene advertencias.")

    st.dataframe(displayed, hide_index=True, width="stretch")
    st.download_button(
        "Descargar predicciones (CSV)",
        data=service.predictions_to_csv_bytes(ordered),
        file_name=f"predicciones_{_timestamp()}.csv",
        mime="text/csv",
        type="primary",
        key="descarga_lote",
    )


def render_batch_notices(result: service.BatchResult) -> None:
    """Informa sobre las transformaciones aplicadas automáticamente al archivo."""
    if result.ignored_columns:
        st.info(
            f"Se ignoraron columnas que el modelo no utiliza: {', '.join(result.ignored_columns)}.",
            icon="📋",
        )
    if result.dropped_empty_rows:
        st.info(f"Se descartaron {result.dropped_empty_rows} fila(s) vacía(s).", icon="📋")
    if result.duplicated_rows:
        st.info(f"El archivo contiene {result.duplicated_rows} fila(s) duplicada(s).", icon="📋")


def process_upload(bundle: service.ModelBundle, uploaded: object) -> None:
    """Lee, valida y predice el archivo cargado, mostrando cada etapa en la interfaz."""
    content = uploaded.getvalue()  # type: ignore[attr-defined]
    filename = uploaded.name  # type: ignore[attr-defined]
    impute = st.checkbox(
        "Convertir valores no numéricos en vacíos e imputarlos",
        key=IMPUTE_KEY,
        value=False,
        help=(
            "Por defecto un valor no numérico detiene el proceso. Active esta opción para "
            "tratarlo como dato faltante."
        ),
    )

    raw = service.read_uploaded_table(content, filename)
    st.caption(f"Archivo leído: {len(raw)} fila(s) y {len(raw.columns)} columna(s).")
    st.dataframe(raw.head(PREVIEW_ROWS), hide_index=True, width="stretch")
    if len(raw) > PREVIEW_ROWS:
        st.caption(f"Vista previa de {PREVIEW_ROWS} de {len(raw)} filas.")

    result = service.run_batch_prediction(bundle, raw, impute_non_numeric=impute)
    render_batch_notices(result)
    render_validation_report(result.report)

    if result.report.is_blocking:
        st.warning(
            "No se generaron predicciones porque el archivo tiene errores que deben "
            "corregirse primero.",
            icon="⚠️",
        )
        return

    st.divider()
    render_batch_summary(result)
    st.divider()
    render_batch_predictions(result)


def render_batch_tab(bundle: service.ModelBundle | None, load_error: str | None) -> None:
    """Renderiza la pestaña de procesamiento por lotes."""
    render_batch_instructions()

    uploaded = st.file_uploader(
        "Archivo con los pacientes",
        type=["csv", "parquet"],
        accept_multiple_files=False,
        key=UPLOAD_KEY,
        help=f"Máximo {validation.MAX_BATCH_ROWS} filas.",
    )

    if bundle is None:
        render_error(load_error or "El modelo no está disponible.")
        return

    if uploaded is None:
        st.info("Cargue un archivo para obtener las predicciones.", icon="📋")
        st.markdown("**Columnas obligatorias**")
        st.dataframe(service.required_columns_frame(), hide_index=True, width="stretch")
        return

    try:
        process_upload(bundle, uploaded)
    except service.AppInferenceError as error:
        render_error(str(error), error)
    except Exception as error:
        render_error("Ocurrió un problema inesperado al procesar el archivo.", error)


def render_model_tab(bundle: service.ModelBundle | None, load_error: str | None) -> None:
    """Renderiza la pestaña con la ficha técnica del modelo."""
    st.subheader("Modelo en producción")
    if bundle is None:
        render_error(load_error or "El modelo no está disponible.")
        return

    for message in bundle.load_warnings:
        st.warning(f"Advertencia al cargar el modelo: {message}", icon="⚠️")

    facts = st.columns(3)
    facts[0].metric("Algoritmo", bundle.model_type)
    facts[1].metric("Muestras de entrenamiento", bundle.metrics.get("train_samples", "-"))
    facts[2].metric("Muestras de prueba", bundle.metrics.get("test_samples", "-"))
    st.caption(f"Artefacto: `{bundle.source_path}` · Generado: {bundle.created_at or '-'}")

    st.markdown("**Métricas de evaluación sobre el conjunto de prueba**")
    metrics_frame = service.summarize_metrics(bundle)
    if metrics_frame.empty:
        st.info("El artefacto no incluye métricas de evaluación.")
    else:
        st.dataframe(metrics_frame, hide_index=True, width="stretch")

    confusion = service.confusion_matrix_frame(bundle)
    if not confusion.empty:
        st.markdown("**Matriz de confusión**")
        st.dataframe(confusion, width="stretch")

    st.markdown("**Variables de entrada**")
    st.dataframe(service.required_columns_frame(), hide_index=True, width="stretch")
    st.caption(
        "El pipeline añade automáticamente dos variables derivadas: el cociente entre "
        "bilirrubina directa y total, y el cociente AST/ALT (índice De Ritis)."
    )

    render_training_images()

    st.markdown("**Preprocesamiento y limitaciones**")
    st.markdown(
        "- Imputación por mediana (variables numéricas) y moda (género).\n"
        "- Transformación Yeo-Johnson y escalado robusto en las variables asimétricas.\n"
        "- Codificación one-hot del género.\n"
        f"- Umbral de decisión fijo en {HIGH_RISK_THRESHOLD:.0%}; el artefacto no almacena "
        "un umbral calibrado.\n"
        "- Entrenado sobre el Indian Liver Patient Dataset, un conjunto pequeño y "
        "desbalanceado. Antes de cualquier uso clínico harían falta validación externa, "
        "calibración, revisión médica y monitoreo de deriva."
    )


def render_training_images() -> None:
    """Muestra las gráficas diagnósticas del entrenamiento, si están disponibles."""
    available = [
        (title, MODEL_IMAGES_DIR / filename)
        for title, filename in (
            ("Curva ROC", "roc_curve.png"),
            ("Matriz de confusión", "confusion_matrix.png"),
            ("Curva de aprendizaje", "learning_curve.png"),
            ("Métricas train/CV/test", "train_cv_test_metrics.png"),
        )
        if (MODEL_IMAGES_DIR / filename).exists()
    ]
    if not available:
        return
    with st.expander("Gráficas diagnósticas del entrenamiento"):
        for title, path in available:
            st.markdown(f"**{title}**")
            st.image(str(path), width="stretch")


def render_sidebar(bundle: service.ModelBundle | None, load_error: str | None) -> None:
    """Dibuja la barra lateral con el estado del modelo."""
    with st.sidebar:
        st.header("Estado del sistema")
        if bundle is None:
            st.error("Modelo no disponible", icon="🚫")
            if load_error:
                st.caption(load_error)
        else:
            st.success("Modelo cargado", icon="✅")
            st.caption(f"Algoritmo: {bundle.model_type}")
            st.caption(f"Artefacto: {bundle.source_path}")
            if not bundle.supports_probabilities:
                st.warning("El modelo no expone probabilidades.", icon="⚠️")
        st.divider()
        st.caption(DISCLAIMER)


def _timestamp() -> str:
    """Devuelve una marca temporal UTC apta para nombrar archivos descargados."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


@st.cache_resource(show_spinner="Cargando el modelo entrenado...")
def load_bundle(model_path: str) -> service.ModelBundle:
    """Carga el modelo una sola vez por proceso de Streamlit.

    Args:
        model_path: Ruta del artefacto, usada además como clave de la caché.

    Returns:
        ModelBundle con el pipeline entrenado y sus metadatos.
    """
    return service.load_app_model(Path(model_path))


def resolve_bundle() -> tuple[service.ModelBundle | None, str | None]:
    """Intenta cargar el modelo y devuelve el bundle o el mensaje de error asociado."""
    path = service.resolve_app_model_path()
    try:
        return load_bundle(str(path)), None
    except service.AppInferenceError as error:
        return None, str(error)
    except Exception:
        return None, (
            "No fue posible cargar el modelo entrenado. Verifique el despliegue e "
            "intente nuevamente."
        )


def main() -> None:
    """Punto de entrada de la aplicación Streamlit."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    initialize_state()

    bundle, load_error = resolve_bundle()
    render_sidebar(bundle, load_error)

    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.caption(DISCLAIMER)

    individual_tab, batch_tab, model_tab = st.tabs(
        ["Predicción individual", "Procesamiento por lotes", "Información del modelo"]
    )

    with individual_tab:
        try:
            render_single_tab(bundle, load_error)
        except Exception as error:
            render_error("Ocurrió un problema inesperado en la predicción individual.", error)

    with batch_tab:
        try:
            render_batch_tab(bundle, load_error)
        except Exception as error:
            render_error("Ocurrió un problema inesperado en el procesamiento por lotes.", error)

    with model_tab:
        try:
            render_model_tab(bundle, load_error)
        except Exception as error:
            render_error("Ocurrió un problema inesperado al mostrar la ficha del modelo.", error)


main()
