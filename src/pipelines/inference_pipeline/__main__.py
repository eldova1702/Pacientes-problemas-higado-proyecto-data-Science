"""Punto de entrada para ejecución modular: python -m src.pipelines.inference_pipeline."""

import sys

from src.pipelines.inference_pipeline.inference_pipeline import main

if __name__ == "__main__":
    sys.exit(main())
