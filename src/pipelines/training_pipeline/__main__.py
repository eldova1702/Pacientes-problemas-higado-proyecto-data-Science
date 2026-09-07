"""Punto de entrada para ejecución modular: python -m src.pipelines.training_pipeline."""

import sys

from src.pipelines.training_pipeline.train_pipeline import main

if __name__ == "__main__":
    sys.exit(main())
