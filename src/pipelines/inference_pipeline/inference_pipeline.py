"""Pipeline de inferencia (Inference Pipeline) para detección de enfermedad hepática.

Este módulo implementa la etapa de inferencia de la arquitectura FTI
(Feature, Training, Inference). Sus responsabilidades son:
1. Cargar el modelo entrenado y su pipeline de preprocesamiento desde el almacenamiento local.
2. Leer datos nuevos de pacientes desde un archivo (Parquet o CSV).
3. Aplicar las mismas transformaciones usadas durante el entrenamiento (el pipeline
   persistido incluye el estandarizador de columnas, el generador de ratios clínicos y el
   preprocesamiento numérico/categórico).
4. Generar predicciones y probabilidades, almacenarlas en disco y visualizarlas.
5. Ejecutarse de forma autónoma mediante interfaz de línea de comandos (CLI).

Uso:
    uv run python src/pipelines/inference_pipeline/inference_pipeline.py --input-data <archivo>
    uv run python -m src.pipelines.inference_pipeline --input-data <archivo>
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import logging
import sys
from pathlib import Path
from typing import Any, cast

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Permitir ejecución directa del script
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importar el módulo de preprocesamiento para que joblib resuelva los transformadores
from src.model import preprocessing as _preprocessing  # noqa: E402, F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("inference_pipeline")

DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "liver_patient_model"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "model.joblib"
DEFAULT_FALLBACK_MODEL_PATH = PROJECT_ROOT / "models" / "modelo_final.joblib"
DEFAULT_INPUT_DATA_PATH = (
    PROJECT_ROOT / "data" / "02_intermediate" / "pacientes_higado_exploracion.parquet"
)
DEFAULT_PREDICTIONS_DIR = DEFAULT_MODEL_DIR / "predictions"

METADATA_COLUMNS = {"patient_id", "event_time"}
TARGET_COLUMNS = {"diagnosis", "dataset"}
REQUIRED_FEATURE_COLUMNS = [
    "age",
    "gender",
    "total_bilirubin",
    "direct_bilirubin",
    "alkaline_phosphotase",
    "alamine_aminotransferase",
    "aspartate_aminotransferase",
    "total_protiens",
    "albumin",
    "albumin_and_globulin_ratio",
]
LABEL_NAMES: dict[int, str] = {1: "enfermo", 0: "sano"}


class InferencePipelineError(Exception):
    """Excepción lanzada cuando la inferencia no puede completarse correctamente."""


def _normalize_column_name(name: Any) -> str:
    """Normaliza un nombre de columna a snake_case minúsculo."""
    normalized = (
        pd.Index([str(name).strip()])
        .str.replace(r"[^a-zA-Z0-9]+", "_", regex=True)
        .str.strip("_")
        .str.lower()[0]
    )
    return str(normalized)


def _resolve_model_path(model_path: Path | None = None) -> Path:
    """Determina la ruta del modelo a cargar, usando fallbacks locales si es necesario."""
    if model_path is not None:
        return model_path
    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH
    return DEFAULT_FALLBACK_MODEL_PATH


def _portable_path(path: str | Path) -> str:
    """Convierte una ruta a formato relativo al proyecto cuando sea posible.

    Evita persistir rutas absolutas dependientes de la máquina en la evidencia.
    """
    resolved = Path(path)
    try:
        return resolved.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def build_portable_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Construye una copia del resumen con rutas portables (relativas al proyecto).

    Args:
        result: Resultado de `run_inference_pipeline` con rutas absolutas.

    Returns:
        Diccionario serializable con rutas relativas al proyecto cuando aplica.
    """
    payload = dict(result)
    for key in (
        "model_path",
        "input_data_path",
        "predictions_path",
        "summary_path",
        "report_path",
    ):
        value = payload.get(key)
        if isinstance(value, (str, Path)):
            payload[key] = _portable_path(value)
    figures = result.get("figures", [])
    payload["figures"] = [_portable_path(figure) for figure in figures]
    return payload


def load_model_artifact(model_path: Path | None = None) -> dict[str, Any]:
    """Carga el artefacto del modelo entrenado desde el almacenamiento del proyecto.

    Acepta tanto el artefacto completo guardado por el Training Pipeline
    (diccionario con la clave ``pipeline``) como un estimador serializado directamente.

    Args:
        model_path: Ruta al archivo `.joblib`. Si es None, se usa el modelo por defecto
            y se aplica fallback a `models/modelo_final.joblib`.

    Returns:
        Diccionario con la clave ``pipeline`` (estimador entrenado) y metadatos asociados.

    Raises:
        FileNotFoundError: Si el archivo del modelo no existe.
        InferencePipelineError: Si el artefacto no contiene un pipeline válido.
    """
    resolved_path = _resolve_model_path(model_path)
    if not resolved_path.exists():
        msg = f"No se encontró el modelo entrenado en: {resolved_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info(f"Cargando modelo entrenado desde: {resolved_path}")
    loaded = joblib.load(resolved_path)

    if isinstance(loaded, dict):
        if "pipeline" not in loaded:
            msg = (
                f"El artefacto en {resolved_path} no contiene la clave 'pipeline'. "
                f"Claves disponibles: {sorted(loaded.keys())}."
            )
            logger.error(msg)
            raise InferencePipelineError(msg)
        artifact: dict[str, Any] = dict(loaded)
        pipeline = artifact["pipeline"]
    else:
        artifact = {"pipeline": loaded, "model_type": type(loaded).__name__}
        pipeline = loaded

    if not hasattr(pipeline, "predict"):
        msg = f"El objeto cargado desde {resolved_path} no implementa predict()."
        logger.error(msg)
        raise InferencePipelineError(msg)

    logger.info(
        f"Modelo cargado correctamente (tipo: {artifact.get('model_type', 'desconocido')})."
    )
    return artifact


def _read_csv(data_path: Path) -> pd.DataFrame:
    """Lee un CSV omitiendo columnas vacías de relleno (p. ej. 'Unnamed: N')."""
    header = pd.read_csv(data_path, nrows=0)
    valid_columns = [
        column
        for column in header.columns
        if str(column).strip() and not str(column).startswith("Unnamed:")
    ]
    dropped = len(header.columns) - len(valid_columns)
    if dropped:
        logger.info(f"Columnas vacías de relleno omitidas al leer el CSV: {dropped}.")
    return pd.read_csv(data_path, usecols=valid_columns)


def load_input_data(data_path: Path) -> pd.DataFrame:
    """Lee datos nuevos de pacientes desde un archivo Parquet o CSV.

    Args:
        data_path: Ruta del archivo de entrada.

    Returns:
        DataFrame con los datos crudos.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        InferencePipelineError: Si la extensión no está soportada o el archivo está vacío.
    """
    if not data_path.exists():
        msg = f"No se encontró el archivo de datos de entrada: {data_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    suffix = data_path.suffix.lower()
    logger.info(f"Leyendo datos nuevos desde: {data_path}")
    if suffix == ".parquet":
        df = pd.read_parquet(data_path)
    elif suffix == ".csv":
        df = _read_csv(data_path)
    else:
        msg = f"Formato de archivo no soportado: '{suffix}'. Use .parquet o .csv."
        logger.error(msg)
        raise InferencePipelineError(msg)

    if df.empty:
        msg = f"El archivo de datos está vacío: {data_path}"
        logger.error(msg)
        raise InferencePipelineError(msg)

    logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas.")
    return df


def extract_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae las columnas de metadatos (p. ej. patient_id, event_time) presentes en el DataFrame.

    Permite conservar la trazabilidad de cada muestra asociándola con su diagnóstico final
    sin ingresar metadatos al pipeline de inferencia.

    Args:
        df: DataFrame crudo de entrada.

    Returns:
        DataFrame con las columnas de metadatos identificadas, o DataFrame vacío con el mismo índice.
    """
    metadata_cols = [
        column for column in df.columns if _normalize_column_name(column) in METADATA_COLUMNS
    ]
    if metadata_cols:
        return df[metadata_cols].copy()
    return pd.DataFrame(index=df.index)


def prepare_inference_features(df: pd.DataFrame) -> pd.DataFrame:
    """Valida y prepara las características crudas para el pipeline entrenado.

    Elimina columnas de metadatos y de target si están presentes, y verifica que
    existan las variables clínicas requeridas por el preprocesamiento.

    Args:
        df: DataFrame crudo proveniente del archivo de entrada.

    Returns:
        DataFrame con las características listas para el pipeline.

    Raises:
        InferencePipelineError: Si falta alguna variable clínica requerida.
    """
    features = df.copy()
    normalized_names = {_normalize_column_name(column) for column in features.columns}

    columns_to_drop = [
        column
        for column in features.columns
        if _normalize_column_name(column) in METADATA_COLUMNS | TARGET_COLUMNS
    ]
    if columns_to_drop:
        features = features.drop(columns=columns_to_drop)
        logger.info(f"Columnas de metadatos/target omitidas: {columns_to_drop}")

    missing = [column for column in REQUIRED_FEATURE_COLUMNS if column not in normalized_names]
    if missing:
        msg = (
            "Faltan variables clínicas requeridas por el modelo: "
            f"{missing}. Variables esperadas: {REQUIRED_FEATURE_COLUMNS}."
        )
        logger.error(msg)
        raise InferencePipelineError(msg)

    logger.info(
        f"Características preparadas: {features.shape[0]} filas, {features.shape[1]} columnas."
    )
    return features


def apply_training_transformations(
    pipeline: BaseEstimator,
    X: pd.DataFrame,
) -> Any | None:
    """Aplica el preprocesamiento entrenado a las características de entrada.

    El pipeline persistido contiene los pasos de transformación entrenados; aquí se
    ejecutan explícitamente (sin el clasificador) para evidenciar la transformación.

    Args:
        pipeline: Pipeline entrenado (preprocesamiento + clasificador).
        X: Características crudas preparadas.

    Returns:
        Matriz transformada, o None si el pipeline no expone pasos de preprocesamiento.
    """
    if isinstance(pipeline, Pipeline) and len(pipeline.steps) > 1:
        try:
            transformed = pipeline[:-1].transform(X)
        except Exception as exc:
            logger.warning(f"No fue posible aplicar las transformaciones del pipeline: {exc}")
            return None
        else:
            logger.info(
                f"Transformaciones de entrenamiento aplicadas. "
                f"Forma de la matriz transformada: {getattr(transformed, 'shape', 'desconocida')}."
            )
            return cast(Any, transformed)
    return None


def generate_predictions(
    pipeline: BaseEstimator,
    X: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Genera predicciones, etiquetas y probabilidades para los datos de entrada.

    Args:
        pipeline: Pipeline entrenado.
        X: Características preparadas.
        metadata: DataFrame opcional con metadatos identificadores (p. ej. patient_id, event_time)
            para asociar a cada predicción.

    Returns:
        DataFrame con los metadatos, características originales y columnas de predicción.

    Raises:
        InferencePipelineError: Si el pipeline no puede generar predicciones o probabilidades.
    """
    logger.info(f"Generando predicciones para {len(X)} muestras...")
    try:
        predictions = np.asarray(pipeline.predict(X))
        probabilities: np.ndarray | None = None
        classes = list(getattr(pipeline, "classes_", []))
        if hasattr(pipeline, "predict_proba"):
            probabilities = np.asarray(pipeline.predict_proba(X))
    except Exception as exc:
        msg = f"Error al generar predicciones o probabilidades con el modelo cargado: {exc}"
        logger.exception(msg)
        raise InferencePipelineError(msg) from exc

    result = X.copy()
    if metadata is not None and not metadata.empty:
        for column in reversed(metadata.columns):
            if column not in result.columns:
                result.insert(0, column, metadata[column].to_numpy())

    result["prediction"] = predictions
    result["prediction_label"] = [LABEL_NAMES.get(int(value), str(value)) for value in predictions]

    if probabilities is not None:
        try:
            positive_index = classes.index(1) if 1 in classes else -1
            negative_index = classes.index(0) if 0 in classes else -1
            result["probability_disease"] = (
                probabilities[:, positive_index] if positive_index != -1 else np.nan
            )
            result["probability_no_disease"] = (
                probabilities[:, negative_index] if negative_index != -1 else np.nan
            )
        except Exception as exc:
            msg = f"Error al indexar las probabilidades calculadas por el modelo: {exc}"
            logger.exception(msg)
            raise InferencePipelineError(msg) from exc
    else:
        logger.warning("El modelo no implementa predict_proba(); no se generan probabilidades.")
        result["probability_disease"] = np.nan
        result["probability_no_disease"] = np.nan

    positive_count = int(np.sum(predictions == 1))
    logger.info(
        f"Predicciones generadas: {positive_count} enfermos y "
        f"{len(predictions) - positive_count} sanos."
    )
    return result


def save_predictions(predictions: pd.DataFrame, output_path: Path) -> str:
    """Almacena las predicciones en un archivo Parquet o CSV.

    Args:
        predictions: DataFrame con las predicciones generadas.
        output_path: Ruta destino del archivo.

    Returns:
        Ruta del archivo guardado.

    Raises:
        InferencePipelineError: Si la extensión del archivo no está soportada.
    """
    suffix = output_path.suffix.lower()
    if suffix not in (".parquet", ".csv"):
        msg = f"Formato de salida no soportado: '{suffix}'. Use .parquet o .csv."
        logger.error(msg)
        raise InferencePipelineError(msg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".parquet":
        predictions.to_parquet(output_path, index=False)
    else:
        predictions.to_csv(output_path, index=False)
    logger.info(f"Predicciones almacenadas en: {output_path}")
    return str(output_path)


def plot_prediction_distribution(predictions: pd.DataFrame, output_path: Path) -> str | None:
    """Genera una visualización de la distribución de predicciones y probabilidades.

    Args:
        predictions: DataFrame con las predicciones generadas.
        output_path: Ruta destino de la imagen PNG.

    Returns:
        Ruta de la imagen generada, o None si falla la generación.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        counts = predictions["prediction_label"].value_counts()
        color_map = {"sano": "#10b981", "enfermo": "#ef4444"}
        axes[0].bar(
            counts.index.astype(str),
            counts.to_numpy(),
            color=[color_map.get(str(label), "#6b7280") for label in counts.index],
        )
        axes[0].set_title("Distribución de predicciones")
        axes[0].set_xlabel("Diagnóstico predicho")
        axes[0].set_ylabel("Cantidad de pacientes")
        axes[0].grid(axis="y", alpha=0.3)

        probabilities = predictions["probability_disease"].dropna()
        if not probabilities.empty:
            axes[1].hist(probabilities, bins=20, color="#3b82f6", alpha=0.85)
            axes[1].set_title("Probabilidad estimada de enfermedad")
            axes[1].set_xlabel("P(enfermedad)")
            axes[1].set_ylabel("Frecuencia")
            axes[1].grid(axis="y", alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, "Sin probabilidades disponibles", ha="center", va="center")
            axes[1].set_axis_off()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info(f"Visualización de predicciones guardada en: {output_path}")
    except Exception as exc:
        logger.warning(f"No fue posible generar la visualización de predicciones: {exc}")
        return None
    return str(output_path)


def generate_inference_html_report(
    result: dict[str, Any],
    predictions: pd.DataFrame,
    figure_path: str | Path | None,
    report_path: Path,
    preview_rows: int = 10,
) -> str:
    """Genera un reporte HTML autocontenido con el resumen y las predicciones.

    La figura se incrusta como data URI base64, por lo que el reporte no depende de
    archivos externos y puede abrirse directamente sin visualizaciones rotas.

    Args:
        result: Resumen de la ejecución de inferencia.
        predictions: DataFrame con las predicciones generadas.
        figure_path: Ruta de la imagen PNG a incrustar (opcional).
        report_path: Ruta destino del archivo .html.
        preview_rows: Número de filas de predicciones a mostrar en la vista previa.

    Returns:
        Ruta del archivo HTML generado.
    """
    encoded_image = ""
    if figure_path is not None and Path(figure_path).exists():
        encoded = base64.b64encode(Path(figure_path).read_bytes()).decode("ascii")
        encoded_image = f"data:image/png;base64,{encoded}"

    image_html = (
        f'<img src="{encoded_image}" alt="Distribución de predicciones" '
        "style='max-width: 100%; border: 1px solid #e5e7eb; border-radius: 6px;'/>"
        if encoded_image
        else "<p style='color: #6b7280;'>Visualización no disponible.</p>"
    )

    preview_columns = [
        column
        for column in ("prediction", "prediction_label", "probability_disease")
        if column in predictions.columns
    ]
    preview_html = predictions.head(preview_rows)[preview_columns].to_html(
        index=False, border=0, float_format=lambda value: f"{value:.4f}"
    )

    status = html.escape(str(result.get("status", "success")))
    model_type = html.escape(str(result.get("model_type", "desconocido")))
    rows = {
        "Muestras procesadas": result.get("n_samples", "-"),
        "Predicciones: enfermo": result.get("positive_predictions", "-"),
        "Predicciones: sano": result.get("negative_predictions", "-"),
        "Proporción positiva": result.get("positive_ratio", "-"),
        "Forma matriz transformada": result.get("transformed_shape", "-"),
    }
    rows_html = "".join(
        f"<tr><th style='text-align:left; padding:6px 10px;'>{html.escape(str(key))}</th>"
        f"<td style='padding:6px 10px;'>{html.escape(str(value))}</td></tr>"
        for key, value in rows.items()
    )

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8">
    <title>Inference Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 30px; color: #111827; }}
        .card {{ max-width: 900px; margin: 0 auto; background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #e5e7eb; padding-bottom: 16px; margin-bottom: 20px; }}
        h1 {{ margin: 0; font-size: 20px; }}
        h2 {{ font-size: 16px; color: #374151; margin-top: 24px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border-bottom: 1px solid #e5e7eb; }}
    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <div>
                <h1>Inference Report</h1>
                <small style="color: #6b7280;">Predicciones del modelo de detección hepática</small>
            </div>
            <span style="background: #10b981; color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold; text-transform: uppercase; font-size: 14px;">
                {status}
            </span>
        </div>
        <p style="color: #4b5563;">Modelo: <strong>{model_type}</strong></p>
        <h2>Resumen de la ejecución</h2>
        <table>{rows_html}</table>
        <h2>Muestra de predicciones</h2>
        {preview_html}
        <h2>Visualización</h2>
        {image_html}
    </div>
</body>
</html>
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html_content, encoding="utf-8")
    logger.info(f"Reporte de inferencia guardado en: {report_path}")
    return str(report_path)


def save_inference_summary(summary: dict[str, Any], output_path: Path) -> str:
    """Guarda un resumen JSON de la ejecución de inferencia.

    Args:
        summary: Diccionario con el resumen de la inferencia.
        output_path: Ruta destino del archivo JSON.

    Returns:
        Ruta del archivo guardado.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False, default=str)
    logger.info(f"Resumen de inferencia guardado en: {output_path}")
    return str(output_path)


def run_inference_pipeline(
    model_path: Path | None = None,
    input_data_path: Path | None = None,
    output_path: Path | None = None,
    output_dir: Path | None = None,
    generate_plots: bool = True,
) -> dict[str, Any]:
    """Ejecuta el ciclo completo de inferencia: carga, transformación y predicción.

    Args:
        model_path: Ruta del modelo entrenado (`.joblib`).
        input_data_path: Ruta de los datos nuevos (Parquet o CSV).
        output_path: Ruta del archivo de predicciones (Parquet o CSV).
        output_dir: Directorio para almacenar predicciones, resumen y figuras.
        generate_plots: Si es True, genera la visualización de predicciones.

    Returns:
        Diccionario con el resumen de la ejecución y las rutas de los artefactos.
    """
    logger.info("=== Iniciando Inference Pipeline ===")
    target_output_dir = output_dir or (
        output_path.parent if output_path else DEFAULT_PREDICTIONS_DIR
    )
    target_input = input_data_path or DEFAULT_INPUT_DATA_PATH
    target_output = output_path or target_output_dir / "predictions.parquet"

    # 1. Carga del modelo entrenado
    artifact = load_model_artifact(model_path=model_path)
    pipeline = cast(BaseEstimator, artifact["pipeline"])

    # 2. Lectura de datos nuevos
    raw_data = load_input_data(target_input)

    # 3. Preparación de características y preservación de metadatos de identificación
    metadata = extract_metadata(raw_data)
    features = prepare_inference_features(raw_data)
    transformed = apply_training_transformations(pipeline, features)
    transformed_shape = list(getattr(transformed, "shape", [])) or None

    # 4. Generación de predicciones
    predictions = generate_predictions(pipeline, features, metadata=metadata)

    # 5. Almacenamiento y visualización
    predictions_path = save_predictions(predictions, target_output)
    figures: list[str] = []
    if generate_plots:
        figure_path = plot_prediction_distribution(
            predictions, target_output_dir / "images" / "prediction_distribution.png"
        )
        if figure_path:
            figures.append(figure_path)

    positive_count = int((predictions["prediction"] == 1).sum())
    total = len(predictions)
    result: dict[str, Any] = {
        "status": "success",
        "model_path": str(_resolve_model_path(model_path)),
        "model_type": artifact.get("model_type", "desconocido"),
        "input_data_path": str(target_input),
        "n_samples": total,
        "positive_predictions": positive_count,
        "negative_predictions": total - positive_count,
        "positive_ratio": round(positive_count / max(total, 1), 4),
        "transformed_shape": transformed_shape,
        "predictions_path": predictions_path,
        "figures": figures,
    }

    # 6. Reporte HTML autocontenido (figura embebida como data URI base64)
    report_path = generate_inference_html_report(
        result=result,
        predictions=predictions,
        figure_path=figures[0] if figures else None,
        report_path=target_output_dir / "inference_report.html",
    )
    result["report_path"] = report_path

    # 7. Resumen JSON con rutas portables (relativas al proyecto cuando aplica)
    summary_path = save_inference_summary(
        build_portable_summary(result), target_output_dir / "inference_summary.json"
    )
    result["summary_path"] = summary_path

    logger.info(
        f"=== Inference Pipeline Finalizado (success) === "
        f"{total} predicciones ({positive_count} enfermos, {total - positive_count} sanos)."
    )
    return result


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Inference Pipeline para detección de enfermedad hepática.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Ruta al modelo entrenado (.joblib). Por defecto: models/liver_patient_model/model.joblib",
    )
    parser.add_argument(
        "--input-data",
        type=Path,
        default=None,
        help="Ruta a los datos nuevos (.parquet o .csv). Por defecto: datos intermedios del proyecto.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Ruta del archivo de predicciones (.parquet o .csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directorio para almacenar predicciones, resumen y figuras.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Desactiva la generación de visualizaciones.",
    )
    return parser.parse_args(args)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada principal para CLI."""
    args = parse_args(argv)

    try:
        result = run_inference_pipeline(
            model_path=args.model_path,
            input_data_path=args.input_data,
            output_path=args.output_path,
            output_dir=args.output_dir,
            generate_plots=not args.no_plots,
        )
        logger.info(
            f"Resultado del Inference Pipeline: {result['status']} -> {result['predictions_path']}"
        )
    except Exception:
        logger.exception("Error durante la ejecución del Inference Pipeline.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
