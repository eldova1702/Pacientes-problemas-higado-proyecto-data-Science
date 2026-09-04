"""Módulo puente/alias para ejecutar o importar train_pipeline directamente desde src.pipelines."""

import sys
from pathlib import Path

# Permitir ejecución directa de este script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.training_pipeline.train_pipeline import (  # noqa: E402
    DEFAULT_FEATURE_GROUP_NAME,
    DEFAULT_FEATURE_GROUP_VERSION,
    DEFAULT_FEATURE_VIEW_NAME,
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_TYPE,
    build_training_pipeline,
    evaluate_model,
    fetch_training_data,
    main,
    run_training_pipeline,
    save_model_artifacts,
    split_training_data,
    train_model,
)

__all__ = [
    "DEFAULT_FEATURE_GROUP_NAME",
    "DEFAULT_FEATURE_GROUP_VERSION",
    "DEFAULT_FEATURE_VIEW_NAME",
    "DEFAULT_MODEL_DIR",
    "DEFAULT_MODEL_TYPE",
    "build_training_pipeline",
    "evaluate_model",
    "fetch_training_data",
    "main",
    "run_training_pipeline",
    "save_model_artifacts",
    "split_training_data",
    "train_model",
]

if __name__ == "__main__":
    sys.exit(main())
