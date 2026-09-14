"""Pruebas unitarias para el módulo de validación del modelo (src/model/model_validation.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.model.model_validation import (
    PRIMARY_METRIC,
    ModelValidationError,
    assess_fit,
    compute_split_metrics,
    cross_validate_model,
    generate_html_report,
    plot_learning_curve,
    plot_train_cv_test_metrics,
    validate_model_performance,
)
from src.model.preprocessing import build_feature_pipeline
from src.pipelines.training_pipeline.train_pipeline import (
    build_training_pipeline,
    train_model,
)

MIN_EXPECTED_FIGURES: int = 2
EXPECTED_CV_FOLDS: int = 5


def _build_features(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Genera un DataFrame sintético con las columnas clínicas esperadas por el pipeline."""
    return pd.DataFrame(
        {
            "age": rng.normal(45, 12, n),
            "gender": rng.choice(["Male", "Female"], n, p=[0.72, 0.28]),
            "total_bilirubin": rng.exponential(1.6, n),
            "direct_bilirubin": rng.exponential(0.6, n),
            "alkaline_phosphotase": rng.uniform(120, 480, n),
            "alamine_aminotransferase": rng.uniform(15, 130, n),
            "aspartate_aminotransferase": rng.uniform(20, 160, n),
            "total_protiens": rng.uniform(5.0, 8.5, n),
            "albumin": rng.uniform(2.0, 4.8, n),
            "albumin_and_globulin_ratio": rng.uniform(0.6, 1.5, n),
        }
    )


@pytest.fixture
def signal_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Dataset con señal real en `total_bilirubin` para que el modelo generalice bien."""
    rng = np.random.default_rng(7)
    n = 320
    X = _build_features(n, rng)
    logit = -2.2 + 1.5 * X["total_bilirubin"].to_numpy()
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = pd.Series(rng.binomial(1, prob), name="diagnosis")
    return X, y


@pytest.fixture
def noisy_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Dataset sin señal (target aleatorio) para inducir overfitting con árboles profundos."""
    rng = np.random.default_rng(11)
    n = 160
    X = _build_features(n, rng)
    y = pd.Series(rng.choice([0, 1], n, p=[0.55, 0.45]), name="diagnosis")
    return X, y


def _cv_result(mean: float, std: float = 0.02) -> dict:
    """Construye un resultado de CV mínimo para probar `assess_fit`."""
    return {"metrics": {PRIMARY_METRIC: {"mean": mean, "std": std, "train_mean": mean}}}


def test_cross_validate_model_returns_metrics(
    signal_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """La validación cruzada devuelve métricas medias por fold dentro de rango."""
    X, y = signal_dataset
    pipeline = build_training_pipeline("logistic_regression", random_state=42)

    res = cross_validate_model(pipeline, X, y, cv_folds=EXPECTED_CV_FOLDS, random_state=42)

    assert res["cv_folds"] == EXPECTED_CV_FOLDS
    assert res["splitter"] == "StratifiedKFold"
    expected_metrics = {
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    }
    assert expected_metrics.issubset(res["metrics"].keys())

    for data in res["metrics"].values():
        assert 0.0 <= data["mean"] <= 1.0
        assert data["std"] >= 0.0
        assert len(data["fold_values"]) == EXPECTED_CV_FOLDS


def test_compute_split_metrics_keys(signal_dataset: tuple[pd.DataFrame, pd.Series]) -> None:
    """`compute_split_metrics` calcula las métricas clínicas esperadas."""
    X, y = signal_dataset
    pipeline = build_training_pipeline("logistic_regression", random_state=42)
    fitted = train_model(pipeline, X, y)

    metrics = compute_split_metrics(fitted, X, y)

    for key in ["accuracy", "balanced_accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0
    assert metrics["n_samples"] == len(y)


def test_assess_fit_good_fit() -> None:
    """Diagnostica good_fit cuando train, CV y test son similares."""
    res = assess_fit(
        train_metrics={PRIMARY_METRIC: 0.85},
        cv_results=_cv_result(0.83),
        test_metrics={PRIMARY_METRIC: 0.82},
    )
    assert res["diagnosis"] == "good_fit"
    assert res["status"] == "passed"


def test_assess_fit_overfitting_warning() -> None:
    """Diagnostica overfitting moderado cuando la brecha train-CV supera el umbral."""
    res = assess_fit(
        train_metrics={PRIMARY_METRIC: 0.95},
        cv_results=_cv_result(0.80),
        test_metrics={PRIMARY_METRIC: 0.78},
    )
    assert res["diagnosis"] == "overfitting"
    assert res["status"] == "warning"
    assert res["recommendations"]


def test_assess_fit_overfitting_severe_fails() -> None:
    """Marca fallo crítico cuando el overfitting supera el umbral severo."""
    res = assess_fit(
        train_metrics={PRIMARY_METRIC: 0.99},
        cv_results=_cv_result(0.60),
        test_metrics={PRIMARY_METRIC: 0.58},
    )
    assert res["diagnosis"] == "overfitting"
    assert res["status"] == "failed"


def test_assess_fit_underfitting_warning() -> None:
    """Diagnostica underfitting cuando el rendimiento es bajo en train y CV."""
    res = assess_fit(
        train_metrics={PRIMARY_METRIC: 0.55},
        cv_results=_cv_result(0.58),
        test_metrics={PRIMARY_METRIC: 0.56},
    )
    assert res["diagnosis"] == "underfitting"
    assert res["status"] == "warning"


def test_assess_fit_possible_overfitting() -> None:
    """Detecta caída de CV a test como posible sobreajuste."""
    res = assess_fit(
        train_metrics={PRIMARY_METRIC: 0.80},
        cv_results=_cv_result(0.78),
        test_metrics={PRIMARY_METRIC: 0.60},
    )
    assert res["diagnosis"] == "possible_overfitting"
    assert res["status"] == "warning"


def test_validate_model_performance_success(
    signal_dataset: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    """Ejecuta la validación completa y genera reportes y figuras."""
    X, y = signal_dataset
    pipeline = build_training_pipeline("logistic_regression", random_state=42)
    fitted = train_model(pipeline, X, y)

    output_dir = tmp_path / "model_validation"
    res = validate_model_performance(
        pipeline=fitted,
        X_train=X,
        y_train=y,
        X_test=X,
        y_test=y,
        cv_folds=5,
        random_state=42,
        output_dir=output_dir,
    )

    assert res["status"] in ("passed", "warning")
    assert res["errors"] == []
    assert res["summary"]["cv_folds"] == EXPECTED_CV_FOLDS
    assert "cross_validation" in res
    assert "fit_analysis" in res
    assert len(res["generated_figures"]) >= MIN_EXPECTED_FIGURES

    json_path = output_dir / "model_validation_report.json"
    html_path = output_dir / "model_validation_report.html"
    assert json_path.exists()
    assert html_path.exists()

    with open(json_path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert "cross_validation" in loaded
    assert loaded["status"] in ("passed", "warning")


def test_validate_model_performance_raise_on_error(
    noisy_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Lanza ModelValidationError ante overfitting severo con raise_on_error=True."""
    X, y = noisy_dataset
    overfit_pipeline = build_training_pipeline("logistic_regression", random_state=42)
    overfit_pipeline.set_params(classifier=DecisionTreeClassifier(max_depth=None, random_state=0))
    fitted = train_model(overfit_pipeline, X, y)

    with pytest.raises(ModelValidationError, match="Fallo en la validación del modelo"):
        validate_model_performance(
            pipeline=fitted,
            X_train=X,
            y_train=y,
            X_test=X,
            y_test=y,
            cv_folds=5,
            random_state=42,
            raise_on_error=True,
            generate_plots=False,
        )


def test_validate_model_performance_no_raise_reports_failure(
    noisy_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Con raise_on_error=False reporta el fallo sin lanzar excepción."""
    X, y = noisy_dataset
    overfit_pipeline = build_training_pipeline("logistic_regression", random_state=42)
    overfit_pipeline.set_params(classifier=DecisionTreeClassifier(max_depth=None, random_state=0))
    fitted = train_model(overfit_pipeline, X, y)

    res = validate_model_performance(
        pipeline=fitted,
        X_train=X,
        y_train=y,
        X_test=X,
        y_test=y,
        cv_folds=5,
        random_state=42,
        raise_on_error=False,
        generate_plots=False,
    )

    assert res["passed"] is False
    assert res["status"] == "failed"
    assert len(res["errors"]) > 0


def test_validate_model_performance_without_plots(
    signal_dataset: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    """Con generate_plots=False no crea figuras pero sí conserva los reportes."""
    X, y = signal_dataset
    pipeline = build_training_pipeline("logistic_regression", random_state=42)
    fitted = train_model(pipeline, X, y)
    output_dir = tmp_path / "no_plots"

    res = validate_model_performance(
        pipeline=fitted,
        X_train=X,
        y_train=y,
        X_test=X,
        y_test=y,
        output_dir=output_dir,
        generate_plots=False,
    )

    assert res["generated_figures"] == []
    assert (output_dir / "model_validation_report.json").exists()


def test_plot_learning_curve_generates_file(
    signal_dataset: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    """La curva de aprendizaje se guarda correctamente en disco."""
    X, y = signal_dataset
    pipeline = build_training_pipeline("logistic_regression", random_state=42)
    output_path = tmp_path / "images" / "learning_curve.png"

    result = plot_learning_curve(pipeline, X, y, output_path, cv_folds=5, random_state=42)

    assert result is not None
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_train_cv_test_metrics_generates_file(
    signal_dataset: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    """La gráfica comparativa train/CV/test se guarda correctamente."""
    X, y = signal_dataset
    pipeline = build_training_pipeline("logistic_regression", random_state=42)
    fitted = train_model(pipeline, X, y)
    cv_results = cross_validate_model(pipeline, X, y, cv_folds=5, random_state=42)

    output_path = tmp_path / "images" / "metrics.png"
    result = plot_train_cv_test_metrics(
        train_metrics=compute_split_metrics(fitted, X, y),
        cv_results=cv_results,
        test_metrics=compute_split_metrics(fitted, X, y),
        output_path=output_path,
    )

    assert result is not None
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_generate_html_report_escapes_markup(tmp_path: Path) -> None:
    """Verifica el escapado de HTML en el reporte de validación del modelo."""
    html_file = tmp_path / "xss_model.html"
    malicious_results = {
        "status": "warning",
        "cross_validation": {"metrics": {"<script>alert(1)</script>": {"mean": 0.5, "std": 0.1}}},
        "train_metrics": {"<script>alert(1)</script>": 0.5},
        "test_metrics": {"<script>alert(1)</script>": 0.5},
        "fit_analysis": {
            "status": "warning",
            "diagnosis": "overfitting",
            "message": "<b>inyección</b>",
            "observations": ["<i>obs</i>"],
            "recommendations": ["<u>rec</u>"],
        },
        "generated_figures": [],
    }
    generate_html_report(malicious_results, html_file)

    assert html_file.exists()
    content = html_file.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in content
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in content
    assert "<b>inyección</b>" not in content
    assert "&lt;b&gt;inyección&lt;/b&gt;" in content


def test_feature_pipeline_available() -> None:
    """Sanity check de que el preprocesamiento compartido se construye."""
    pipeline = build_feature_pipeline()
    assert pipeline is not None
