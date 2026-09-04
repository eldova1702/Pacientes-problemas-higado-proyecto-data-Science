"""Pipeline de ingeniería de características (Feature Pipeline) para el Feature Store.

Este módulo implementa el pipeline de características de la arquitectura FTI
(Feature, Training, Inference). Sus responsabilidades son:
1. Extraer los datos crudos o intermedios del dataset de pacientes con problemas hepáticos.
2. Transformar y generar características clínicas relevantes (ratios de bilirrubina y AST/ALT).
3. Cargar el resultado en el Feature Store (Hopsworks).
4. Ejecutarse de forma autónoma mediante interfaz de línea de comandos (CLI).

Uso:
    uv run python src/pipelines/feature_pipeline/feature_pipeline.py --dry-run
    uv run python -m src.pipelines.feature_pipeline --dry-run
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path
from typing import Any

# Permitir ejecución directa del script (uv run python src/pipelines/feature_pipeline/feature_pipeline.py)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.feature_store import (  # noqa: E402
    DEFAULT_HISTORICAL_TIMESTAMP,
    FEATURE_DESCRIPTIONS,
    FEATURE_NAME_MAPPING,
    get_hopsworks_project,
    get_or_create_patient_feature_group,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("feature_pipeline")

INTERMEDIATE_DATA_PATH = (
    PROJECT_ROOT / "data" / "02_intermediate" / "pacientes_higado_exploracion.parquet"
)
RAW_DATA_PATH = PROJECT_ROOT / "data" / "01_raw" / "Pacientes_porblemas_higado_india.csv"

VALID_RAW_COLUMNS = [
    "Age",
    "Gender",
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Protiens",
    "Albumin",
    "Albumin_and_Globulin_Ratio",
    "Dataset",
]


def extract_data(data_path: Path | None = None) -> pd.DataFrame:
    """Extrae los datos clínicos originales desde archivo parquet o CSV.

    Args:
        data_path: Ruta personalizada del archivo de datos (opcional). Si es None,
            intenta cargar primero desde la ruta intermedia parquet y luego desde raw CSV.

    Returns:
        DataFrame con las columnas clínicas de pacientes.

    Raises:
        FileNotFoundError: Si no se encuentra ningún archivo de datos válido.
    """
    if data_path is not None:
        if not data_path.exists():
            msg = f"La ruta de datos especificada no existe: {data_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info(f"Cargando datos desde ruta especificada: {data_path}")
        if data_path.suffix == ".parquet":
            return pd.read_parquet(data_path)

        header_cols = list(pd.read_csv(data_path, nrows=0).columns)
        if "Dataset" in header_cols:
            usecols = [c for c in VALID_RAW_COLUMNS if c in header_cols]
            df_csv = pd.read_csv(data_path, usecols=usecols)
            return df_csv.rename(columns={"Dataset": "Diagnosis"})
        return pd.read_csv(data_path)

    if INTERMEDIATE_DATA_PATH.exists():
        logger.info(f"Cargando datos intermedios desde {INTERMEDIATE_DATA_PATH}")
        return pd.read_parquet(INTERMEDIATE_DATA_PATH)

    if RAW_DATA_PATH.exists():
        logger.info(f"Cargando datos raw desde {RAW_DATA_PATH} con columnas válidas")
        df_raw = pd.read_csv(RAW_DATA_PATH, usecols=VALID_RAW_COLUMNS)
        return df_raw.rename(columns={"Dataset": "Diagnosis"})

    msg = (
        f"No se encontró el archivo de datos ni en {INTERMEDIATE_DATA_PATH} ni en {RAW_DATA_PATH}."
    )
    logger.error(msg)
    raise FileNotFoundError(msg)


def transform_features(
    df: pd.DataFrame,
    start_id: int = 1,
    base_timestamp: datetime.datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Transforma los datos y calcula características clínicas derivadas para ML.

    Realiza:
    1. Estandarización de nombres de columnas a snake_case en minúsculas.
    2. Creación de identificador único determinista (`patient_id`).
    3. Asignación de marca temporal en UTC (`event_time`).
    4. Cálculo de razones clínicas especializadas:
       - `direct_to_total_bilirubin`: Razón de bilirrubina directa / total.
       - `ast_to_alt`: Cociente De Ritis (AST / ALT).
    5. Tipado estricto preservando tipos nativos compatibles con Hopsworks.

    Args:
        df: DataFrame de entrada con mediciones clínicas.
        start_id: Identificador inicial para la clave primaria de pacientes.
        base_timestamp: Marca de tiempo base para los eventos (por defecto usa
            DEFAULT_HISTORICAL_TIMESTAMP para reproducibilidad).

    Returns:
        DataFrame con características limpias, derivadas y tipificadas.
    """
    df_transformed = df.copy()

    # Estandarizar nombres de columnas a snake_case
    rename_dict = {
        col: FEATURE_NAME_MAPPING[col]
        for col in df_transformed.columns
        if col in FEATURE_NAME_MAPPING
    }
    df_transformed = df_transformed.rename(columns=rename_dict)
    df_transformed.columns = [str(col).lower() for col in df_transformed.columns]

    # Clave primaria única determinista
    if "patient_id" not in df_transformed.columns:
        df_transformed["patient_id"] = range(start_id, start_id + len(df_transformed))
    df_transformed["patient_id"] = df_transformed["patient_id"].astype("int64")

    # Marca temporal de evento en UTC
    if "event_time" not in df_transformed.columns:
        ts = base_timestamp if base_timestamp is not None else DEFAULT_HISTORICAL_TIMESTAMP
        df_transformed["event_time"] = pd.to_datetime(ts)
    else:
        df_transformed["event_time"] = pd.to_datetime(df_transformed["event_time"])

    # Normalizar género como string o None si es nulo
    if "gender" in df_transformed.columns:
        df_transformed["gender"] = df_transformed["gender"].apply(
            lambda x: (
                str(x)
                if pd.notna(x) and str(x).strip().lower() not in ("nan", "none", "")
                else None
            )
        )

    # Conversión de mediciones numéricas base
    numeric_base_cols = [
        "age",
        "total_bilirubin",
        "direct_bilirubin",
        "alkaline_phosphotase",
        "alamine_aminotransferase",
        "aspartate_aminotransferase",
        "total_protiens",
        "albumin",
        "albumin_and_globulin_ratio",
    ]
    for col in numeric_base_cols:
        if col in df_transformed.columns:
            df_transformed[col] = pd.to_numeric(df_transformed[col], errors="coerce")

    # Ingeniería de características clínicas:
    # 1. Razón bilirrubina directa / bilirrubina total
    if "direct_bilirubin" in df_transformed.columns and "total_bilirubin" in df_transformed.columns:
        tb = df_transformed["total_bilirubin"].replace(0, np.nan)
        ratio_bili = df_transformed["direct_bilirubin"] / tb
        df_transformed["direct_to_total_bilirubin"] = ratio_bili.replace([np.inf, -np.inf], np.nan)

    # 2. Razón AST / ALT (Cociente De Ritis)
    if (
        "aspartate_aminotransferase" in df_transformed.columns
        and "alamine_aminotransferase" in df_transformed.columns
    ):
        alt = df_transformed["alamine_aminotransferase"].replace(0, np.nan)
        ratio_ast_alt = df_transformed["aspartate_aminotransferase"] / alt
        df_transformed["ast_to_alt"] = ratio_ast_alt.replace([np.inf, -np.inf], np.nan)

    # Diagnóstico (etiqueta target si está presente)
    if "diagnosis" in df_transformed.columns:
        df_transformed["diagnosis"] = pd.to_numeric(df_transformed["diagnosis"], errors="coerce")
        if not df_transformed["diagnosis"].isna().any():
            df_transformed["diagnosis"] = df_transformed["diagnosis"].astype("int64")

    # Orden prioritario de columnas
    priority_cols = ["patient_id", "event_time"]
    remaining_cols = [col for col in df_transformed.columns if col not in priority_cols]
    df_transformed = df_transformed[priority_cols + remaining_cols]

    return df_transformed


def load_features_to_store(  # noqa: PLR0913
    df: pd.DataFrame,
    fg_name: str = "pacientes_higado_fg",
    version: int = 2,
    *,
    api_key: str | None = None,
    project_name: str | None = None,
    time_travel_format: str = "HUDI",
    wait_for_job: bool = True,
) -> dict[str, Any]:
    """Carga el DataFrame transformado al Feature Store de Hopsworks.

    Args:
        df: DataFrame con características procesadas para almacenar.
        fg_name: Nombre del Feature Group de destino.
        version: Versión del Feature Group.
        api_key: Llave API de Hopsworks (opcional).
        project_name: Nombre del proyecto en Hopsworks (opcional).
        time_travel_format: Formato temporal ('HUDI' o 'DELTA').
        wait_for_job: Indica si se debe esperar a que finalice el job de ingestión.

    Returns:
        Diccionario con el resumen de la operación.
    """
    logger.info("Iniciando conexión con Hopsworks para registrar características...")
    project = get_hopsworks_project(api_key=api_key, project_name=project_name)
    fs = project.get_feature_store()

    feature_group = get_or_create_patient_feature_group(
        fs=fs,
        name=fg_name,
        version=version,
        time_travel_format=time_travel_format,
    )

    logger.info(f"Insertando {len(df)} registros en el Feature Group '{fg_name}' (v{version})...")
    feature_group.insert(df, write_options={"wait_for_job": wait_for_job})
    logger.info("Inserción en Feature Store completada con éxito.")

    # Actualización de descripciones de características
    metadata_warnings: list[str] = []
    for feat_name, desc in FEATURE_DESCRIPTIONS.items():
        if feat_name in df.columns:
            try:
                feature_group.update_feature_description(feat_name, desc)
            except Exception as err:
                logger.warning(
                    f"No se pudo actualizar descripción de la característica '{feat_name}': {err}"
                )
                metadata_warnings.append(feat_name)

    if metadata_warnings:
        logger.warning(
            f"Advertencias al registrar metadatos de características: {metadata_warnings}"
        )
    else:
        logger.info("Descripciones de características actualizadas en Hopsworks.")

    return {
        "status": "success",
        "records_inserted": len(df),
        "feature_group_name": fg_name,
        "version": version,
        "features": list(df.columns),
        "metadata_warnings": metadata_warnings,
    }


def run_feature_pipeline(  # noqa: PLR0913
    data_path: Path | None = None,
    fg_name: str = "pacientes_higado_fg",
    version: int = 2,
    *,
    dry_run: bool = False,
    wait_for_job: bool = True,
    api_key: str | None = None,
    project_name: str | None = None,
) -> dict[str, Any]:
    """Ejecuta el pipeline completo de características de forma autónoma.

    Args:
        data_path: Ruta al archivo de datos de entrada (opcional).
        fg_name: Nombre del Feature Group en Hopsworks.
        version: Versión del Feature Group.
        dry_run: Si es True, solo extrae y transforma los datos sin conectar a Hopsworks.
        wait_for_job: Si es True, espera a que finalice el job de inserción.
        api_key: Llave API de Hopsworks (opcional).
        project_name: Nombre del proyecto Hopsworks (opcional).

    Returns:
        Diccionario con el resultado de la ejecución.
    """
    logger.info("=== Iniciando Feature Pipeline ===")
    df_raw = extract_data(data_path=data_path)
    logger.info(f"Datos extraídos: {df_raw.shape[0]} filas, {df_raw.shape[1]} columnas.")

    logger.info("Transformando datos y calculando características clínicas...")
    df_features = transform_features(df_raw)
    logger.info(
        f"Transformación finalizada: {df_features.shape[0]} filas, {df_features.shape[1]} columnas."
    )
    logger.info(f"Columnas resultantes: {list(df_features.columns)}")

    if dry_run:
        logger.info("Modo --dry-run activado: Omitiendo carga en Feature Store remoto.")
        logger.info(f"Esquema de tipos generado:\n{df_features.dtypes}")
        logger.info(f"Resumen estadístico de características:\n{df_features.describe().T}")
        return {
            "status": "dry_run_success",
            "records_processed": len(df_features),
            "features": list(df_features.columns),
        }

    return load_features_to_store(
        df=df_features,
        fg_name=fg_name,
        version=version,
        api_key=api_key,
        project_name=project_name,
        wait_for_job=wait_for_job,
    )


def main() -> int:
    """Punto de entrada de línea de comandos para la ejecución autónoma."""
    parser = argparse.ArgumentParser(
        description="Pipeline autónomo de extracción, ingeniería de features y carga en Feature Store."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Ruta al archivo de datos (parquet o csv).",
    )
    parser.add_argument(
        "--fg-name",
        type=str,
        default="pacientes_higado_fg",
        help="Nombre del Feature Group en Hopsworks.",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=2,
        help="Versión del Feature Group (por defecto: 2).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ejecuta solo la extracción y transformación sin conectar a Hopsworks.",
    )
    parser.add_argument(
        "--wait",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Esperar a que el job de inserción en Hopsworks finalice (por defecto: --wait).",
    )

    args = parser.parse_args()

    try:
        result = run_feature_pipeline(
            data_path=args.data_path,
            fg_name=args.fg_name,
            version=args.version,
            dry_run=args.dry_run,
            wait_for_job=args.wait,
        )
        logger.info(f"Feature Pipeline finalizado exitosamente: {result}")
    except Exception:
        logger.exception("Error durante la ejecución del Feature Pipeline.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
