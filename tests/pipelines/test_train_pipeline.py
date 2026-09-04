"""Pruebas unitarias para el Training Pipeline (src/pipelines/training_pipeline/train_pipeline.py)."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest

from src.pipelines.training_pipeline.train_pipeline import (
    PROJECT_ROOT,
    build_training_pipeline,
    evaluate_model,
    fetch_training_data,
    get_classifier,
    get_or_create_feature_view,
    main,
    plot_and_save_figures,
    register_model_in_hopsworks,
    run_training_pipeline,
    save_model_artifacts,
    split_training_data,
    train_model,
)

MIN_EXPECTED_DIAGNOSTIC_PLOTS: int = 2


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    """Fixture con dataset sintético realista de pacientes con problemas hepáticos."""
    np.random.seed(42)
    n = 60
    return pd.DataFrame(
        {
            "patient_id": range(1, n + 1),
            "event_time": pd.to_datetime(["2026-08-31"] * n, utc=True),
            "age": np.random.uniform(20, 75, size=n),
            "gender": np.random.choice(["Male", "Female"], size=n),
            "total_bilirubin": np.random.uniform(0.5, 10.0, size=n),
            "direct_bilirubin": np.random.uniform(0.1, 4.0, size=n),
            "alkaline_phosphotase": np.random.uniform(100, 500, size=n),
            "alamine_aminotransferase": np.random.uniform(15, 120, size=n),
            "aspartate_aminotransferase": np.random.uniform(20, 150, size=n),
            "total_protiens": np.random.uniform(5.0, 8.5, size=n),
            "albumin": np.random.uniform(2.0, 4.8, size=n),
            "albumin_and_globulin_ratio": np.random.uniform(0.6, 1.5, size=n),
            "direct_to_total_bilirubin": np.random.uniform(0.2, 0.6, size=n),
            "ast_to_alt": np.random.uniform(0.8, 2.0, size=n),
            "diagnosis": np.random.choice([1.0, 2.0], size=n, p=[0.7, 0.3]),
        }
    )


def test_get_or_create_feature_view_existing() -> None:
    """Verifica que se recupere la Feature View existente si no lanza excepción."""
    mock_fs = MagicMock()
    mock_fg = MagicMock()
    mock_fv = MagicMock()
    mock_fs.get_feature_view.return_value = mock_fv

    fv = get_or_create_feature_view(
        fs=mock_fs,
        feature_group=mock_fg,
        name="test_fv",
        version=1,
    )

    assert fv == mock_fv
    mock_fs.get_feature_view.assert_called_once_with(name="test_fv", version=1)
    mock_fs.create_feature_view.assert_not_called()


def test_get_or_create_feature_view_new() -> None:
    """Verifica que se cree una nueva Feature View si get_feature_view lanza excepción."""
    mock_fs = MagicMock()
    mock_fg = MagicMock()
    mock_fv = MagicMock()
    mock_fs.get_feature_view.side_effect = Exception("Not found")
    mock_fs.create_feature_view.return_value = mock_fv

    fv = get_or_create_feature_view(
        fs=mock_fs,
        feature_group=mock_fg,
        name="test_fv",
        version=1,
        labels=["diagnosis"],
    )

    assert fv == mock_fv
    mock_fs.create_feature_view.assert_called_once()


def test_fetch_training_data_from_feature_store(sample_dataset: pd.DataFrame) -> None:
    """Verifica la lectura exitosa desde Hopsworks Feature Store."""
    mock_project = MagicMock()
    mock_fs = MagicMock()
    mock_fg = MagicMock()
    mock_fv = MagicMock()

    mock_project.get_feature_store.return_value = mock_fs
    mock_fs.get_feature_group.return_value = mock_fg
    mock_fs.get_feature_view.return_value = mock_fv
    mock_fg.read.return_value = sample_dataset

    with patch(
        "src.pipelines.training_pipeline.train_pipeline.get_hopsworks_project",
        return_value=mock_project,
    ):
        df, fv, proj = fetch_training_data(
            use_feature_store=True,
            feature_group_name="pacientes_higado_fg",
            feature_group_version=3,
        )

    assert len(df) == len(sample_dataset)
    assert fv == mock_fv
    assert proj == mock_project


def test_fetch_training_data_fallback_local(tmp_path: Path, sample_dataset: pd.DataFrame) -> None:
    """Verifica el fallback a datos locales cuando el Feature Store falla."""
    local_file = tmp_path / "patients_test.parquet"
    sample_dataset.to_parquet(local_file)

    with patch(
        "src.pipelines.training_pipeline.train_pipeline.get_hopsworks_project",
        side_effect=RuntimeError("Connection refused"),
    ):
        df, fv, proj = fetch_training_data(
            use_feature_store=True,
            local_data_path=local_file,
        )

    assert len(df) == len(sample_dataset)
    assert fv is None
    assert proj is None


def test_fetch_training_data_file_not_found(tmp_path: Path) -> None:
    """Lanza FileNotFoundError si ni el Feature Store ni el archivo local existen."""
    missing_file = tmp_path / "non_existent.parquet"
    with pytest.raises(FileNotFoundError, match="No se pudo cargar desde Feature Store"):
        fetch_training_data(
            use_feature_store=False,
            local_data_path=missing_file,
        )


def test_split_training_data_stratification(sample_dataset: pd.DataFrame) -> None:
    """Verifica la división estratificada en train y test."""
    X_train, X_test, y_train, y_test = split_training_data(
        df=sample_dataset,
        target_col="diagnosis",
        test_size=0.25,
        random_state=42,
    )

    assert len(X_train) + len(X_test) == len(sample_dataset)
    assert len(X_test) == int(len(sample_dataset) * 0.25)
    assert "patient_id" not in X_train.columns
    assert "event_time" not in X_train.columns
    assert "diagnosis" not in X_train.columns
    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})


def test_get_classifier_types() -> None:
    """Verifica la instanciación de clasificadores soportados y error ante no soportado."""
    lr = get_classifier("logistic_regression", random_state=42)
    assert hasattr(lr, "predict")

    rf = get_classifier("random_forest", random_state=42)
    assert hasattr(rf, "predict")

    with pytest.raises(ValueError, match="Tipo de modelo 'unsupported_model' no soportado"):
        get_classifier("unsupported_model")


def test_build_and_train_pipeline(sample_dataset: pd.DataFrame) -> None:
    """Verifica la construcción y el ajuste del pipeline completo."""
    X_train, X_test, y_train, y_test = split_training_data(sample_dataset, test_size=0.3)

    pipeline = build_training_pipeline(model_type="logistic_regression", random_state=42)
    trained = train_model(pipeline=pipeline, X_train=X_train, y_train=y_train)

    assert hasattr(trained, "classes_")
    assert hasattr(trained, "predict")

    preds = trained.predict(X_test)
    assert len(preds) == len(X_test) == len(y_test)


def test_evaluate_model(sample_dataset: pd.DataFrame) -> None:
    """Verifica que evaluate_model calcule todas las métricas clínicas."""
    X_train, X_test, y_train, y_test = split_training_data(sample_dataset, test_size=0.3)
    pipeline = build_training_pipeline("logistic_regression")
    trained = train_model(pipeline, X_train, y_train)

    metrics = evaluate_model(pipeline=trained, X_test=X_test, y_test=y_test)

    expected_keys = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "average_precision",
        "confusion_matrix",
        "test_samples",
    ]
    for key in expected_keys:
        assert key in metrics

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert isinstance(metrics["confusion_matrix"], list)


def test_plot_and_save_figures(tmp_path: Path, sample_dataset: pd.DataFrame) -> None:
    """Verifica la generación y almacenamiento de gráficas diagnósticas."""
    X_train, X_test, y_train, y_test = split_training_data(sample_dataset, test_size=0.3)
    pipeline = build_training_pipeline("logistic_regression")
    trained = train_model(pipeline, X_train, y_train)

    output_dir = tmp_path / "model_artifacts"
    figures = plot_and_save_figures(
        pipeline=trained,
        X_test=X_test,
        y_test=y_test,
        output_dir=output_dir,
    )

    assert len(figures) >= MIN_EXPECTED_DIAGNOSTIC_PLOTS
    for fig_path in figures:
        assert Path(fig_path).exists()
        assert Path(fig_path).stat().st_size > 0


def test_save_model_artifacts(tmp_path: Path, sample_dataset: pd.DataFrame) -> None:
    """Verifica que el modelo y métricas se guarden correctamente en disco."""
    X_train, X_test, y_train, y_test = split_training_data(sample_dataset, test_size=0.3)
    pipeline = build_training_pipeline("logistic_regression")
    trained = train_model(pipeline, X_train, y_train)
    metrics = evaluate_model(trained, X_test, y_test)

    output_dir = tmp_path / "saved_model"
    saved = save_model_artifacts(
        pipeline=trained,
        metrics=metrics,
        output_dir=output_dir,
        model_type="logistic_regression",
        feature_names=list(X_train.columns),
        save_main_model_symlink=False,
    )

    assert Path(saved["model_path"]).exists()
    assert Path(saved["metrics_path"]).exists()

    # Probar que el artefacto guardado se carga y predice
    loaded_artifact = joblib.load(saved["model_path"])
    assert "pipeline" in loaded_artifact
    loaded_pipe = loaded_artifact["pipeline"]
    preds = loaded_pipe.predict(X_test)
    assert len(preds) == len(X_test)

    # Probar lectura del JSON de métricas
    with open(saved["metrics_path"], encoding="utf-8") as f:
        metrics_read = json.load(f)
    assert metrics_read["accuracy"] == metrics["accuracy"]


def test_register_model_in_hopsworks(tmp_path: Path) -> None:
    """Verifica la invocación al API de Model Registry de Hopsworks."""
    mock_project = MagicMock()
    mock_mr = MagicMock()
    mock_model = MagicMock()
    mock_model.version = 1
    mock_fv = MagicMock()

    mock_project.get_model_registry.return_value = mock_mr
    mock_mr.python.create_model.return_value = mock_model

    metrics = {"accuracy": 0.85, "roc_auc": 0.90}
    hw_model = register_model_in_hopsworks(
        project=mock_project,
        feature_view=mock_fv,
        model_dir=tmp_path,
        metrics=metrics,
        model_name="test_model",
    )

    assert hw_model.version == 1
    mock_mr.python.create_model.assert_called_once()
    mock_model.save.assert_called_once_with(str(tmp_path))


def test_run_training_pipeline_dry_run(tmp_path: Path, sample_dataset: pd.DataFrame) -> None:
    """Verifica la ejecución completa del pipeline en modo simulación (--dry-run)."""
    local_data = tmp_path / "patients_local.parquet"
    sample_dataset.to_parquet(local_data)
    output_dir = tmp_path / "pipeline_output"

    result = run_training_pipeline(
        model_type="logistic_regression",
        local_data_path=local_data,
        output_dir=output_dir,
        dry_run=True,
    )

    assert result["status"] == "dry_run_success"
    assert result["model_type"] == "logistic_regression"
    assert "metrics" in result
    assert "saved_paths" in result
    assert Path(result["saved_paths"]["model_path"]).exists()


def test_run_training_pipeline_random_forest(tmp_path: Path, sample_dataset: pd.DataFrame) -> None:
    """Verifica el pipeline entrenando un Random Forest."""
    local_data = tmp_path / "patients_local.parquet"
    sample_dataset.to_parquet(local_data)
    output_dir = tmp_path / "rf_output"

    result = run_training_pipeline(
        model_type="random_forest",
        local_data_path=local_data,
        output_dir=output_dir,
        dry_run=True,
    )

    assert result["status"] == "dry_run_success"
    assert result["model_type"] == "random_forest"
    assert Path(result["saved_paths"]["model_path"]).exists()


def test_main_cli_dry_run(tmp_path: Path, sample_dataset: pd.DataFrame) -> None:
    """Verifica la ejecución desde interfaz de línea de comandos."""
    local_data = tmp_path / "patients_cli.parquet"
    sample_dataset.to_parquet(local_data)
    output_dir = tmp_path / "cli_output"

    argv = [
        "--dry-run",
        "--local-data",
        str(local_data),
        "--output-dir",
        str(output_dir),
        "--model-type",
        "logistic_regression",
    ]

    exit_code = main(argv)
    assert exit_code == 0
    assert (output_dir / "model.joblib").exists()
    assert (output_dir / "metrics.json").exists()


def test_main_cli_error() -> None:
    """Verifica que errores durante main retornen código 1."""
    argv = ["--local-data", "non_existent_file_path_123.parquet", "--dry-run"]
    exit_code = main(argv)
    assert exit_code == 1


def test_save_main_model_symlink(tmp_path: Path, sample_dataset: pd.DataFrame) -> None:
    """Verifica que el modelo principal se guarde cuando save_main_model_symlink es True."""
    X_train, X_test, y_train, y_test = split_training_data(sample_dataset, test_size=0.3)
    pipeline = build_training_pipeline("logistic_regression")
    trained = train_model(pipeline, X_train, y_train)
    metrics = evaluate_model(trained, X_test, y_test)

    output_dir = tmp_path / "custom_output"
    saved = save_model_artifacts(
        pipeline=trained,
        metrics=metrics,
        output_dir=output_dir,
        model_type="logistic_regression",
        feature_names=list(X_train.columns),
        save_main_model_symlink=True,
    )

    assert "main_model_path" in saved
    assert Path(saved["main_model_path"]).exists()


def test_main_module_execution(tmp_path: Path, sample_dataset: pd.DataFrame) -> None:
    """Verifica la ejecución modular de __main__.py y del alias train_pipeline.py."""
    local_data = tmp_path / "module_patients.parquet"
    sample_dataset.to_parquet(local_data)
    out_dir = tmp_path / "module_out"

    test_argv = [
        "train_pipeline",
        "--dry-run",
        "--local-data",
        str(local_data),
        "--output-dir",
        str(out_dir),
    ]

    with patch("sys.argv", test_argv), pytest.raises(SystemExit) as exc_info:
        runpy.run_module("src.pipelines.training_pipeline", run_name="__main__")
    assert exc_info.value.code == 0
    assert (out_dir / "model.joblib").exists()

    # Probar también el script puente train_pipeline.py
    alias_path = PROJECT_ROOT / "src" / "pipelines" / "train_pipeline.py"
    out_dir_alias = tmp_path / "alias_out"
    test_argv_alias = [
        "train_pipeline.py",
        "--dry-run",
        "--local-data",
        str(local_data),
        "--output-dir",
        str(out_dir_alias),
    ]
    with patch("sys.argv", test_argv_alias), pytest.raises(SystemExit) as exc_info_alias:
        runpy.run_path(str(alias_path), run_name="__main__")
    assert exc_info_alias.value.code == 0
    assert (out_dir_alias / "model.joblib").exists()
