"""Punto de ejecución de módulo para feature_pipeline."""

from __future__ import annotations

import sys

from src.pipelines.feature_pipeline.feature_pipeline import main

if __name__ == "__main__":
    sys.exit(main())
