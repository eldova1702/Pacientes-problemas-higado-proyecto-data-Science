"""Módulo puente/alias para ejecutar o importar inference_pipeline directamente desde src.pipelines."""

from __future__ import annotations

import sys
from pathlib import Path

# Permitir ejecución directa de este script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.inference_pipeline.inference_pipeline import (  # noqa: E402
    DEFAULT_INPUT_DATA_PATH,
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREDICTIONS_DIR,
    InferencePipelineError,
    apply_training_transformations,
    build_portable_summary,
    generate_inference_html_report,
    generate_predictions,
    load_input_data,
    load_model_artifact,
    main,
    plot_prediction_distribution,
    prepare_inference_features,
    run_inference_pipeline,
    save_inference_summary,
    save_predictions,
)

__all__ = [
    "DEFAULT_INPUT_DATA_PATH",
    "DEFAULT_MODEL_DIR",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_PREDICTIONS_DIR",
    "InferencePipelineError",
    "apply_training_transformations",
    "build_portable_summary",
    "generate_inference_html_report",
    "generate_predictions",
    "load_input_data",
    "load_model_artifact",
    "main",
    "plot_prediction_distribution",
    "prepare_inference_features",
    "run_inference_pipeline",
    "save_inference_summary",
    "save_predictions",
]

if __name__ == "__main__":
    sys.exit(main())
