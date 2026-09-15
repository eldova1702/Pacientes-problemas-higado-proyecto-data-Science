"""Servicio de inferencia que alimenta la demo Streamlit (predicción individual y por lotes).

Este módulo concentra toda la lógica que la interfaz necesita, de modo que `app.py` quede
reducido a presentación. Ninguna función de aquí importa Streamlit: así el servicio es
ejecutable y testeable con pytest puro, y la cobertura lo mide (la configuración de coverage
sólo observa `src/`).

Responsabilidades:
1. Cargar el artefacto entrenado y exponer sus metadatos de forma segura.
2. Construir el registro individual del formulario y predecirlo.
3. Leer de forma defensiva el archivo que sube el usuario (CSV o Parquet, con distintos
   separadores y codificaciones) y normalizarlo al contrato del pipeline.
4. Ejecutar el lote reutilizando el Inference Pipeline, de modo que las predicciones de la
   app tengan exactamente el mismo esquema que las del CLI.
5. Convertir resultados a bytes descargables y a tablas listas para graficar.

Toda excepción esperable se reconvierte a `AppInferenceError`, con un mensaje en español
apto para mostrarse tal cual al usuario final.
"""

from __future__ import annotations

import io
import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from src.inference.batch_validation import (
    GENDER_COLUMN,
    MAX_BATCH_ROWS,
    NUMERIC_FEATURE_COLUMNS,
    ValidationIssue,
    ValidationReport,
    clinical_ranges,
    coerce_feature_types,
    label_for,
    valid_gender_values,
    validate_batch_features,
)
from src.pipelines.inference_pipeline.inference_pipeline import (
    DEFAULT_FALLBACK_MODEL_PATH,
    DEFAULT_MODEL_PATH,
    METADATA_COLUMNS,
    PROJECT_ROOT,
    REQUIRED_FEATURE_COLUMNS,
    TARGET_COLUMNS,
    InferencePipelineError,
    extract_metadata,
    generate_predictions,
    load_model_artifact,
)
from src.pipelines.inference_pipeline.inference_pipeline import (
    _normalize_column_name as normalize_column_name,
)

#: Variable de entorno que permite apuntar la app a otro artefacto. Hace testeable el caso
#: «modelo ausente» sin tocar el repositorio.
APP_MODEL_PATH_ENV: str = "LIVER_APP_MODEL_PATH"

#: Paquete raíz del proyecto, usado para distinguir un fallo de empaquetado de un
#: artefacto dañado cuando joblib lanza ModuleNotFoundError.
PROJECT_PACKAGE: str = "src"
CORRUPTED_MODEL_MESSAGE: str = (
    "El archivo del modelo está dañado o es incompatible con las versiones "
    "instaladas. Vuelva a generarlo con el Training Pipeline."
)

SUPPORTED_SUFFIXES: tuple[str, ...] = (".csv", ".parquet")
CSV_ENCODING_CANDIDATES: tuple[str, ...] = ("utf-8-sig", "utf-8", "latin-1")
CSV_SEPARATOR_CANDIDATES: tuple[str | None, ...] = (None, ",", ";", "\t", "|")
DOWNLOAD_ENCODING: str = "utf-8-sig"

WARNING_COLUMN: str = "advertencias"
PREDICTION_COLUMNS: tuple[str, ...] = (
    "prediction",
    "prediction_label",
    "probability_disease",
    "probability_no_disease",
)

DISPLAY_LABELS: dict[int, str] = {1: "Enfermedad hepática", 0: "Sin enfermedad hepática"}
GENDER_DISPLAY: dict[str, str] = {"Male": "Masculino", "Female": "Femenino"}
DEFAULT_HISTOGRAM_BINS: int = 10
TEMPLATE_ROW_COUNT: int = 2

#: Alias frecuentes de género, en minúsculas y sin espacios. El modelo sólo entiende
#: `Male`/`Female`; cualquier otra cosa se codifica como vector de ceros sin avisar.
GENDER_ALIASES: dict[str, str] = {
    "m": "Male",
    "male": "Male",
    "masculino": "Male",
    "hombre": "Male",
    "varon": "Male",
    "varón": "Male",
    "h": "Male",
    "f": "Female",
    "female": "Female",
    "femenino": "Female",
    "mujer": "Female",
    "fem": "Female",
    "masc": "Male",
}

FEATURE_UNITS: dict[str, str] = {
    "age": "años",
    "total_bilirubin": "mg/dL",
    "direct_bilirubin": "mg/dL",
    "alkaline_phosphotase": "UI/L",
    "alamine_aminotransferase": "UI/L",
    "aspartate_aminotransferase": "UI/L",
    "total_protiens": "g/dL",
    "albumin": "g/dL",
    "albumin_and_globulin_ratio": "",
}

#: Valores iniciales del formulario, correspondientes a un perfil clínico intermedio.
SINGLE_RECORD_DEFAULTS: dict[str, float | str] = {
    "age": 45.0,
    "gender": "Male",
    "total_bilirubin": 1.0,
    "direct_bilirubin": 0.3,
    "alkaline_phosphotase": 210.0,
    "alamine_aminotransferase": 36.0,
    "aspartate_aminotransferase": 42.0,
    "total_protiens": 6.5,
    "albumin": 3.1,
    "albumin_and_globulin_ratio": 0.9,
}

FEATURE_STEPS: dict[str, float] = {
    "age": 1.0,
    "total_bilirubin": 0.1,
    "direct_bilirubin": 0.1,
    "alkaline_phosphotase": 1.0,
    "alamine_aminotransferase": 1.0,
    "aspartate_aminotransferase": 1.0,
    "total_protiens": 0.1,
    "albumin": 0.1,
    "albumin_and_globulin_ratio": 0.1,
}


class AppInferenceError(Exception):
    """Error con mensaje apto para el usuario final de la aplicación."""


@dataclass(frozen=True)
class FeatureInputSpec:
    """Especificación de un campo numérico del formulario individual."""

    name: str
    label: str
    minimum: float
    maximum: float
    default: float
    step: float
    unit: str

    @property
    def help_text(self) -> str:
        """Texto de ayuda con unidad y rango clínico admitido."""
        unit = f" en {self.unit}" if self.unit else ""
        return f"Valor{unit}. Rango clínico admitido: {self.minimum:g} a {self.maximum:g}."


@dataclass(frozen=True)
class ModelBundle:
    """Modelo entrenado junto con los metadatos que la interfaz necesita mostrar."""

    pipeline: BaseEstimator
    model_type: str
    metrics: dict[str, Any]
    feature_names: list[str]
    created_at: str
    source_path: str
    supports_probabilities: bool
    load_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SinglePrediction:
    """Resultado de predecir un único paciente desde el formulario."""

    prediction: int
    label: str
    probability_disease: float | None
    probability_no_disease: float | None
    derived_features: dict[str, float | None]
    warnings: tuple[str, ...] = ()
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class BatchResult:
    """Resultado de procesar un archivo con múltiples pacientes."""

    predictions: pd.DataFrame
    report: ValidationReport
    ignored_columns: tuple[str, ...] = ()
    dropped_empty_rows: int = 0
    duplicated_rows: int = 0

    @property
    def n_rows(self) -> int:
        """Cantidad de pacientes con predicción generada."""
        return len(self.predictions)

    @property
    def n_positive(self) -> int:
        """Cantidad de pacientes clasificados como enfermos."""
        if self.predictions.empty:
            return 0
        return int((self.predictions["prediction"] == 1).sum())

    @property
    def n_negative(self) -> int:
        """Cantidad de pacientes clasificados como sanos."""
        return self.n_rows - self.n_positive

    @property
    def positive_ratio(self) -> float:
        """Proporción de pacientes clasificados como enfermos."""
        return self.n_positive / self.n_rows if self.n_rows else 0.0

    @property
    def mean_probability(self) -> float | None:
        """Probabilidad media estimada de enfermedad, o None si no hay probabilidades."""
        if self.predictions.empty or "probability_disease" not in self.predictions.columns:
            return None
        values = self.predictions["probability_disease"].dropna()
        return float(values.mean()) if not values.empty else None


def feature_input_specs() -> list[FeatureInputSpec]:
    """Construye la especificación de los campos numéricos del formulario.

    Los límites salen de `clinical_ranges()`, es decir del esquema Pandera del proyecto, de
    modo que el widget es la primera barrera contra valores imposibles.

    Returns:
        Lista de especificaciones en el orden canónico de las variables.
    """
    ranges = clinical_ranges()
    specs: list[FeatureInputSpec] = []
    for name in NUMERIC_FEATURE_COLUMNS:
        minimum, maximum = ranges[name]
        specs.append(
            FeatureInputSpec(
                name=name,
                label=label_for(name),
                minimum=minimum,
                maximum=maximum,
                default=float(SINGLE_RECORD_DEFAULTS[name]),  # type: ignore[arg-type]
                step=FEATURE_STEPS[name],
                unit=FEATURE_UNITS[name],
            )
        )
    return specs


def resolve_app_model_path(model_path: Path | None = None) -> Path:
    """Determina qué artefacto debe cargar la aplicación.

    Prioridad: argumento explícito, variable de entorno `LIVER_APP_MODEL_PATH`, modelo
    canónico `models/liver_patient_model/model.joblib` y, como último recurso, el espejo
    `models/modelo_final.joblib`.

    Args:
        model_path: Ruta explícita que tiene prioridad sobre todo lo demás.

    Returns:
        Ruta del artefacto a cargar.
    """
    if model_path is not None:
        return Path(model_path)
    from_env = os.environ.get(APP_MODEL_PATH_ENV)
    if from_env:
        return Path(from_env)
    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH
    return DEFAULT_FALLBACK_MODEL_PATH


def portable_model_path(path: str | Path) -> str:
    """Devuelve la ruta relativa al proyecto cuando sea posible, para no exponer rutas locales."""
    resolved = Path(path)
    try:
        return resolved.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.name


def load_app_model(model_path: Path | None = None) -> ModelBundle:
    """Carga el artefacto entrenado y lo envuelve con sus metadatos.

    Cualquier fallo de deserialización se traduce a `AppInferenceError` con un mensaje
    accionable; las advertencias de versión de scikit-learn se recogen sin interrumpir la
    carga, porque el modelo sigue siendo utilizable.

    Args:
        model_path: Ruta explícita del artefacto. Si es None se resuelve automáticamente.

    Returns:
        ModelBundle con el pipeline entrenado y sus metadatos.

    Raises:
        AppInferenceError: Si el modelo no existe, está dañado o no es utilizable.
    """
    resolved = resolve_app_model_path(model_path)
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            artifact = load_model_artifact(resolved)
        load_warnings = _version_warnings(captured)
    except FileNotFoundError as error:
        message = (
            f"No se encontró el modelo entrenado en «{portable_model_path(resolved)}». "
            f"Verifique que el archivo esté incluido en el repositorio desplegado o "
            f"vuelva a generarlo con el Training Pipeline."
        )
        raise AppInferenceError(message) from error
    except ModuleNotFoundError as error:
        # Un archivo dañado también provoca ModuleNotFoundError, porque joblib interpreta
        # los bytes basura como el nombre de un módulo. Sólo es un problema de empaquetado
        # cuando el módulo ausente pertenece realmente al proyecto.
        if str(error.name or "").split(".")[0] == PROJECT_PACKAGE:
            message = (
                "No fue posible reconstruir los transformadores del modelo. Verifique que "
                "el paquete «src/» (en particular src/model/preprocessing.py) esté presente "
                "en el despliegue."
            )
        else:
            message = CORRUPTED_MODEL_MESSAGE
        raise AppInferenceError(message) from error
    except InferencePipelineError as error:
        raise AppInferenceError(str(error)) from error
    except Exception as error:
        raise AppInferenceError(CORRUPTED_MODEL_MESSAGE) from error

    return _build_bundle(artifact, resolved, load_warnings)


def _version_warnings(captured: Sequence[warnings.WarningMessage]) -> tuple[str, ...]:
    """Filtra las advertencias de incompatibilidad de versión emitidas al deserializar."""
    messages: list[str] = []
    for item in captured:
        if "version" in type(item.message).__name__.lower():
            messages.append(str(item.message).strip().splitlines()[0])
    return tuple(messages)


def _build_bundle(
    artifact: Mapping[str, Any], source: Path, load_warnings: tuple[str, ...]
) -> ModelBundle:
    """Construye el `ModelBundle` a partir del diccionario devuelto por joblib."""
    pipeline = artifact["pipeline"]
    metrics = artifact.get("metrics") or {}
    return ModelBundle(
        pipeline=pipeline,
        model_type=str(artifact.get("model_type", "desconocido")),
        metrics=dict(metrics),
        feature_names=list(artifact.get("feature_names") or REQUIRED_FEATURE_COLUMNS),
        created_at=str(artifact.get("created_at", "")),
        source_path=portable_model_path(source),
        supports_probabilities=hasattr(pipeline, "predict_proba"),
        load_warnings=load_warnings,
    )


def build_single_record_frame(values: Mapping[str, float | str]) -> pd.DataFrame:
    """Construye el DataFrame de una fila con el contrato exacto que espera el pipeline.

    Args:
        values: Valores del formulario, indexados por nombre snake_case de la variable.

    Returns:
        DataFrame de una fila con las columnas de `REQUIRED_FEATURE_COLUMNS` en orden.
    """
    record = {
        name: values.get(name, SINGLE_RECORD_DEFAULTS[name]) for name in REQUIRED_FEATURE_COLUMNS
    }
    frame = pd.DataFrame([record], columns=list(REQUIRED_FEATURE_COLUMNS))
    for column in NUMERIC_FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    frame[GENDER_COLUMN] = frame[GENDER_COLUMN].astype(str)
    return frame


def derived_ratios(frame: pd.DataFrame) -> dict[str, float | None]:
    """Calcula los ratios clínicos derivados de una fila, para mostrarlos en la interfaz.

    Reproduce la fórmula de `ClinicalFeatureBuilder` sin invocar el pipeline, únicamente con
    fines de transparencia hacia el usuario.

    Args:
        frame: DataFrame de una fila con las variables clínicas.

    Returns:
        Diccionario con `direct_to_total_bilirubin` y `ast_to_alt`, o None si son indefinidos.
    """
    row = frame.iloc[0]
    ratios: dict[str, float | None] = {}
    total_bilirubin = float(row["total_bilirubin"])
    alt = float(row["alamine_aminotransferase"])
    ratios["direct_to_total_bilirubin"] = (
        float(row["direct_bilirubin"]) / total_bilirubin if total_bilirubin else None
    )
    ratios["ast_to_alt"] = float(row["aspartate_aminotransferase"]) / alt if alt else None
    return ratios


def _single_record_warnings(frame: pd.DataFrame) -> tuple[str, ...]:
    """Detecta situaciones que degradan silenciosamente una predicción individual."""
    messages: list[str] = []
    row = frame.iloc[0]
    if float(row["total_bilirubin"]) == 0:
        messages.append(
            "La bilirrubina total es 0, por lo que el cociente bilirrubina directa/total "
            "queda indefinido y se imputa con la mediana del entrenamiento. La probabilidad "
            "estimada puede desviarse de forma significativa."
        )
    if str(row[GENDER_COLUMN]) not in valid_gender_values():
        messages.append(
            "El género indicado no es una categoría conocida por el modelo; la predicción "
            "es menos confiable."
        )
    return tuple(messages)


def predict_single(bundle: ModelBundle, values: Mapping[str, float | str]) -> SinglePrediction:
    """Predice el diagnóstico de un único paciente.

    Reutiliza `generate_predictions` del Inference Pipeline para que el resultado y sus
    columnas sean idénticos a los del procesamiento por lotes y a los del CLI.

    Args:
        bundle: Modelo cargado.
        values: Valores clínicos del formulario.

    Returns:
        SinglePrediction con la clase, las probabilidades y los avisos aplicables.

    Raises:
        AppInferenceError: Si el modelo no puede generar la predicción.
    """
    frame = build_single_record_frame(values)
    try:
        predicted = generate_predictions(bundle.pipeline, frame)
    except Exception as error:
        message = (
            "No fue posible calcular la predicción con los datos ingresados. Revise los "
            "valores del formulario e intente nuevamente."
        )
        raise AppInferenceError(message) from error

    row = predicted.iloc[0]
    prediction = int(row["prediction"])
    return SinglePrediction(
        prediction=prediction,
        label=DISPLAY_LABELS.get(prediction, str(prediction)),
        probability_disease=_optional_float(row.get("probability_disease")),
        probability_no_disease=_optional_float(row.get("probability_no_disease")),
        derived_features=derived_ratios(frame),
        warnings=_single_record_warnings(frame),
        frame=predicted,
    )


def _optional_float(value: Any) -> float | None:
    """Convierte un valor a float, devolviendo None si es nulo o no convertible."""
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_csv_bytes(content: bytes) -> pd.DataFrame:
    """Lee un CSV probando codificaciones y separadores hasta reconocer las columnas clínicas.

    Se puntúa cada combinación por la cantidad de variables clínicas reconocidas tras
    normalizar los nombres, lo que resuelve de una sola pasada los casos de separador `;`,
    tabulador, codificación latin-1 y marca de orden de bytes.

    Args:
        content: Bytes del archivo subido.

    Returns:
        El DataFrame de la mejor combinación encontrada.

    Raises:
        AppInferenceError: Si ninguna combinación logra interpretar el archivo.
    """
    best_frame: pd.DataFrame | None = None
    best_score = -1
    for encoding in CSV_ENCODING_CANDIDATES:
        for separator in CSV_SEPARATOR_CANDIDATES:
            frame = _try_read_csv(content, encoding, separator)
            if frame is None:
                continue
            score = sum(
                1
                for column in frame.columns
                if normalize_column_name(column) in REQUIRED_FEATURE_COLUMNS
            )
            if score > best_score:
                best_frame, best_score = frame, score
            if best_score == len(REQUIRED_FEATURE_COLUMNS):
                return best_frame if best_frame is not None else frame

    if best_frame is None:
        message = (
            "No fue posible interpretar el archivo como CSV. Revise el formato o descargue "
            "la plantilla, y guarde el archivo con codificación UTF-8."
        )
        raise AppInferenceError(message)
    return best_frame


def _try_read_csv(content: bytes, encoding: str, separator: str | None) -> pd.DataFrame | None:
    """Intenta una lectura concreta de CSV, devolviendo None si la combinación no sirve."""
    try:
        frame = pd.read_csv(
            io.BytesIO(content),
            sep=separator,
            engine="python",
            encoding=encoding,
            skip_blank_lines=True,
        )
    except Exception:
        return None
    if frame.columns.empty:
        return None
    return frame


def _drop_filler_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Descarta las columnas de relleno vacías típicas de los CSV exportados desde Excel."""
    keep = [
        column
        for column in frame.columns
        if str(column).strip() and not str(column).startswith("Unnamed:")
    ]
    return frame[keep]


def read_uploaded_table(content: bytes, filename: str) -> pd.DataFrame:
    """Lee el archivo subido por el usuario y devuelve su contenido crudo.

    Args:
        content: Bytes del archivo.
        filename: Nombre original, usado para determinar el formato.

    Returns:
        DataFrame con índice posicional, donde la posición 0 es la primera fila de datos.

    Raises:
        AppInferenceError: Si el archivo está vacío, tiene un formato no soportado o no
            puede interpretarse.
    """
    if not content:
        raise AppInferenceError("El archivo está vacío.")

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        admitted = ", ".join(SUPPORTED_SUFFIXES)
        message = (
            f"Formato no soportado: «{suffix or filename}». Suba un archivo {admitted}. "
            f"Desde Excel puede exportarlo con la opcion «Guardar como CSV UTF-8»."
        )
        raise AppInferenceError(message)

    frame = _read_parquet_bytes(content) if suffix == ".parquet" else _parse_csv_bytes(content)

    frame = _drop_filler_columns(frame)
    if frame.columns.empty:
        raise AppInferenceError("El archivo no contiene columnas con nombre.")
    return frame.reset_index(drop=True)


def _read_parquet_bytes(content: bytes) -> pd.DataFrame:
    """Lee un archivo Parquet desde memoria, traduciendo el fallo a un mensaje de usuario."""
    try:
        return pd.read_parquet(io.BytesIO(content))
    except Exception as error:
        message = "No fue posible leer el archivo Parquet. Verifique que no esté dañado."
        raise AppInferenceError(message) from error


def normalize_gender(values: pd.Series) -> pd.Series:
    """Traduce los alias frecuentes de género a las categorías que el modelo conoce.

    Args:
        values: Serie con los valores de género tal como vienen del archivo.

    Returns:
        Serie con los alias reconocidos convertidos a `Male`/`Female`; el resto se conserva
        intacto para que la validación pueda advertir sobre ello.
    """
    allowed = valid_gender_values()

    def _translate(value: Any) -> Any:
        if pd.isna(value):
            return value
        text = str(value).strip()
        if text in allowed:
            return text
        return GENDER_ALIASES.get(text.lower(), text)

    return values.map(_translate)


def _normalized_columns(frame: pd.DataFrame) -> list[str]:
    """Devuelve los nombres de columna normalizados a snake_case."""
    return [normalize_column_name(column) for column in frame.columns]


def _assert_no_duplicate_columns(names: Sequence[str]) -> None:
    """Interrumpe si dos columnas distintas normalizan al mismo nombre.

    Raises:
        AppInferenceError: Si hay nombres duplicados tras normalizar.
    """
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        listed = ", ".join(duplicated)
        message = (
            f"Hay columnas duplicadas tras normalizar los nombres: {listed}. Conserve una "
            f"sola columna por variable."
        )
        raise AppInferenceError(message)


def normalize_uploaded_frame(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...], int]:
    """Lleva el archivo crudo al contrato que espera el pipeline entrenado.

    Normaliza los nombres a snake_case, descarta filas totalmente vacías conservando la
    numeración original, separa los metadatos de trazabilidad, descarta la columna de
    diagnóstico real si viene incluida y normaliza los alias de género.

    Args:
        frame: DataFrame crudo devuelto por `read_uploaded_table`.

    Returns:
        Tupla con las características, los metadatos, las columnas ignoradas y el número de
        filas vacías descartadas.

    Raises:
        AppInferenceError: Si dos columnas normalizan al mismo nombre.
    """
    renamed = frame.copy()
    names = _normalized_columns(renamed)
    _assert_no_duplicate_columns(names)
    renamed.columns = pd.Index(names)

    before = len(renamed)
    renamed = renamed.dropna(how="all")
    dropped_empty = before - len(renamed)

    metadata = extract_metadata(renamed)
    ignored = tuple(
        column
        for column in renamed.columns
        if column not in REQUIRED_FEATURE_COLUMNS
        and column not in METADATA_COLUMNS
        and column not in TARGET_COLUMNS
    )
    present = [column for column in REQUIRED_FEATURE_COLUMNS if column in renamed.columns]
    features = renamed[present].copy()

    if GENDER_COLUMN in features.columns:
        features[GENDER_COLUMN] = normalize_gender(features[GENDER_COLUMN])

    return features, metadata, ignored, dropped_empty


def _warning_labels_by_row(report: ValidationReport) -> dict[int, list[str]]:
    """Agrupa las etiquetas de advertencia por número de fila."""
    per_row: dict[int, list[str]] = {}
    for issue in report.warnings:
        label = label_for(issue.column) if issue.column else "archivo"
        for row in issue.rows:
            per_row.setdefault(row, [])
            if label not in per_row[row]:
                per_row[row].append(label)
    return per_row


def _attach_row_warnings(
    predictions: pd.DataFrame, features: pd.DataFrame, report: ValidationReport
) -> pd.DataFrame:
    """Añade una columna con las advertencias que afectan a cada fila."""
    per_row = _warning_labels_by_row(report)
    annotated = predictions.copy()
    annotated[WARNING_COLUMN] = [
        "; ".join(per_row.get(int(index) + 1, [])) for index in features.index
    ]
    return annotated


def run_batch_prediction(
    bundle: ModelBundle,
    raw_frame: pd.DataFrame,
    *,
    max_rows: int = MAX_BATCH_ROWS,
    impute_non_numeric: bool = False,
) -> BatchResult:
    """Procesa un archivo completo de pacientes y devuelve predicciones e informe.

    Si la validación encuentra errores bloqueantes no se invoca al modelo y se devuelve un
    resultado sin predicciones, con el informe completo para que la interfaz lo muestre.

    Args:
        bundle: Modelo cargado.
        raw_frame: DataFrame crudo devuelto por `read_uploaded_table`.
        max_rows: Número máximo de filas admitidas.
        impute_non_numeric: Si es True, los valores no numéricos se imputan en vez de bloquear.

    Returns:
        BatchResult con las predicciones (si las hay) y el informe de validación.

    Raises:
        AppInferenceError: Si el modelo falla pese a haber superado la validación.
    """
    features, metadata, ignored, dropped_empty = normalize_uploaded_frame(raw_frame)
    features, coercion_issues = coerce_feature_types(
        features, impute_non_numeric=impute_non_numeric
    )
    report = validate_batch_features(features, max_rows=max_rows, extra_issues=coercion_issues)
    duplicated = int(features.duplicated().sum()) if not features.empty else 0

    if report.is_blocking:
        return BatchResult(
            predictions=pd.DataFrame(),
            report=report,
            ignored_columns=ignored,
            dropped_empty_rows=dropped_empty,
            duplicated_rows=duplicated,
        )

    try:
        predictions = generate_predictions(bundle.pipeline, features, metadata=metadata)
    except Exception as error:
        message = (
            "No fue posible generar las predicciones con el archivo suministrado. Revise "
            "que los valores clínicos sean correctos e intente nuevamente."
        )
        raise AppInferenceError(message) from error

    return BatchResult(
        predictions=_attach_row_warnings(predictions, features, report),
        report=report,
        ignored_columns=ignored,
        dropped_empty_rows=dropped_empty,
        duplicated_rows=duplicated,
    )


def order_prediction_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    """Reordena el resultado dejando primero las columnas de predicción."""
    leading = [column for column in PREDICTION_COLUMNS if column in predictions.columns]
    if WARNING_COLUMN in predictions.columns:
        leading.append(WARNING_COLUMN)
    remaining = [column for column in predictions.columns if column not in leading]
    return predictions[[*leading, *remaining]]


def predictions_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serializa un DataFrame a CSV descargable, preservando los acentos en Excel."""
    return str(frame.to_csv(index=False)).encode(DOWNLOAD_ENCODING)


def build_template_frame() -> pd.DataFrame:
    """Construye la plantilla de carga por lotes con filas de ejemplo.

    Returns:
        DataFrame con las columnas obligatorias y filas de ejemplo editables.
    """
    example = dict(SINGLE_RECORD_DEFAULTS)
    second = dict(SINGLE_RECORD_DEFAULTS)
    second.update(
        {
            "age": 58.0,
            "gender": "Female",
            "total_bilirubin": 3.9,
            "direct_bilirubin": 2.0,
            "alkaline_phosphotase": 480.0,
            "alamine_aminotransferase": 120.0,
            "aspartate_aminotransferase": 190.0,
            "total_protiens": 5.9,
            "albumin": 2.4,
            "albumin_and_globulin_ratio": 0.6,
        }
    )
    rows = [example, second][:TEMPLATE_ROW_COUNT]
    return pd.DataFrame(rows, columns=list(REQUIRED_FEATURE_COLUMNS))


def template_csv_bytes() -> bytes:
    """Devuelve la plantilla de carga por lotes lista para descargar."""
    return predictions_to_csv_bytes(build_template_frame())


def summarize_metrics(bundle: ModelBundle) -> pd.DataFrame:
    """Construye la tabla de métricas del modelo para la pestaña informativa.

    Args:
        bundle: Modelo cargado.

    Returns:
        DataFrame con las columnas `métrica` y `valor`, vacío si no hay métricas.
    """
    labels = {
        "accuracy": "Exactitud (accuracy)",
        "balanced_accuracy": "Exactitud balanceada",
        "precision": "Precisión (clase enfermo)",
        "recall": "Sensibilidad (recall)",
        "f1_score": "F1-score",
        "roc_auc": "ROC-AUC",
        "average_precision": "Precisión promedio",
        "train_samples": "Muestras de entrenamiento",
        "test_samples": "Muestras de prueba",
    }
    records = [
        {"métrica": label, "valor": bundle.metrics[key]}
        for key, label in labels.items()
        if key in bundle.metrics
    ]
    return pd.DataFrame(records, columns=["métrica", "valor"])


def confusion_matrix_frame(bundle: ModelBundle) -> pd.DataFrame:
    """Construye la matriz de confusión etiquetada en español, o una tabla vacía."""
    matrix = bundle.metrics.get("confusion_matrix")
    expected_size = 2
    if not matrix or len(matrix) != expected_size:
        return pd.DataFrame()
    return pd.DataFrame(
        matrix,
        index=["Real: sin enfermedad", "Real: enfermedad"],
        columns=["Predicho: sin enfermedad", "Predicho: enfermedad"],
    )


def label_counts_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    """Cuenta pacientes por diagnóstico predicho, apto para un gráfico de barras."""
    if predictions.empty or "prediction" not in predictions.columns:
        return pd.DataFrame(columns=["diagnóstico", "pacientes"])
    counts = predictions["prediction"].value_counts()
    records = [
        {"diagnóstico": DISPLAY_LABELS.get(int(value), str(value)), "pacientes": int(count)}
        for value, count in counts.items()
    ]
    return pd.DataFrame(records, columns=["diagnóstico", "pacientes"])


def probability_histogram_frame(
    predictions: pd.DataFrame, bins: int = DEFAULT_HISTOGRAM_BINS
) -> pd.DataFrame:
    """Agrupa la probabilidad estimada de enfermedad en intervalos para graficarla.

    Args:
        predictions: DataFrame con la columna `probability_disease`.
        bins: Cantidad de intervalos entre 0 y 1.

    Returns:
        DataFrame con las columnas `rango` y `pacientes`, vacío si no hay probabilidades.
    """
    empty = pd.DataFrame(columns=["rango", "pacientes"])
    if predictions.empty or "probability_disease" not in predictions.columns:
        return empty
    values = predictions["probability_disease"].dropna()
    if values.empty:
        return empty

    edges = np.linspace(0.0, 1.0, bins + 1)
    counts, _ = np.histogram(values.to_numpy(dtype="float64"), bins=edges)
    records = [
        {"rango": f"{edges[index]:.1f}-{edges[index + 1]:.1f}", "pacientes": int(counts[index])}
        for index in range(bins)
    ]
    return pd.DataFrame(records, columns=["rango", "pacientes"])


def required_columns_frame() -> pd.DataFrame:
    """Construye la tabla documental de columnas obligatorias para la interfaz."""
    ranges = clinical_ranges()
    records: list[dict[str, str]] = []
    for name in REQUIRED_FEATURE_COLUMNS:
        if name == GENDER_COLUMN:
            admitted = " o ".join(
                f"{value} ({GENDER_DISPLAY.get(value, value)})" for value in valid_gender_values()
            )
            unit = "-"
        else:
            minimum, maximum = ranges[name]
            admitted = f"{minimum:g} a {maximum:g}"
            unit = FEATURE_UNITS[name] or "-"
        records.append(
            {
                "columna": name,
                "variable": label_for(name),
                "unidad": unit,
                "valores admitidos": admitted,
            }
        )
    return pd.DataFrame(records, columns=["columna", "variable", "unidad", "valores admitidos"])


def issue_messages(issues: Sequence[ValidationIssue]) -> list[str]:
    """Extrae los mensajes de una secuencia de hallazgos de validación."""
    return [issue.message for issue in issues]
