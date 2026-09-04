"""Pipeline para la carga histórica de características (Backfill) en Hopsworks Feature Store.

Uso:
    uv run python -m src.pipelines.feature_backfill
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.data.feature_store import (
    backfill_patient_features_to_store,
    prepare_patient_features_for_feature_store,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("feature_backfill_pipeline")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
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


def load_dataset(data_path: Path | None = None) -> pd.DataFrame:
    """Carga los datos de pacientes desde intermediate parquet o raw CSV.

    Args:
        data_path: Ruta personalizada del archivo de datos (opcional).

    Returns:
        DataFrame cargado con las columnas clínicas.
    """
    if data_path is not None:
        if not data_path.exists():
            msg = f"La ruta de datos especificada no existe: {data_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)
        logger.info(f"Cargando datos desde ruta especificada: {data_path}")
        if data_path.suffix == ".parquet":
            return pd.read_parquet(data_path)
        df_csv = pd.read_csv(data_path)
        if "Dataset" in df_csv.columns:
            valid_cols = [c for c in VALID_RAW_COLUMNS if c in df_csv.columns]
            df_csv = df_csv[valid_cols].rename(columns={"Dataset": "Diagnosis"})
        return df_csv

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


def main() -> int:
    """Función principal de ejecución del pipeline de backfill."""
    parser = argparse.ArgumentParser(
        description="Pipeline para el registro de características históricas en Hopsworks Feature Store."
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
        help="Ejecuta solo la preparación y validación local de datos sin conectar a Hopsworks.",
    )

    args = parser.parse_args()

    try:
        df = load_dataset(args.data_path)
        logger.info(f"Dataset cargado con éxito: {df.shape[0]} filas y {df.shape[1]} columnas.")

        if args.dry_run:
            logger.info("Modo --dry-run activado: Realizando preparación y validación local...")
            df_prepared = prepare_patient_features_for_feature_store(df)
            logger.info(f"Esquema generado para Feature Store:\n{df_prepared.dtypes}")
            logger.info(f"\nMuestra de datos preparados:\n{df_prepared.head(3)}")
            logger.info("Validación local completada exitosamente.")
            return 0

        logger.info("Iniciando proceso de backfill en Hopsworks Feature Store...")
        result = backfill_patient_features_to_store(
            df=df,
            feature_group_name=args.fg_name,
            version=args.version,
        )
        logger.info(f"Proceso de backfill finalizado: {result}")
    except Exception:
        logger.exception("Error durante la ejecución del pipeline")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
