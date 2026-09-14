"""Pruebas unitarias para el módulo de validación de partición train/test (src/data/split_validation.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.split_validation import (
    TrainTestSplitValidationError,
    check_dataset_sizes,
    check_feature_drift,
    check_index_leakage,
    check_label_distribution,
    check_new_categories,
    check_sample_leakage,
    validate_train_test_split,
)


@pytest.fixture
def clean_train_test_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Genera conjuntos sintéticos limpios y representativos para train y test."""
    np.random.seed(42)
    n_train, n_test = 200, 50

    # Train
    X_train = pd.DataFrame(
        {
            "age": np.random.normal(45, 10, n_train),
            "total_bilirubin": np.random.exponential(1.5, n_train),
            "gender": np.random.choice(["Male", "Female"], n_train, p=[0.7, 0.3]),
        },
        index=[f"train_{i}" for i in range(n_train)],
    )
    y_train = pd.Series(
        np.random.choice([0, 1], n_train, p=[0.3, 0.7]),
        index=X_train.index,
        name="diagnosis",
    )

    # Test
    X_test = pd.DataFrame(
        {
            "age": np.random.normal(45, 10, n_test),
            "total_bilirubin": np.random.exponential(1.5, n_test),
            "gender": np.random.choice(["Male", "Female"], n_test, p=[0.7, 0.3]),
        },
        index=[f"test_{i}" for i in range(n_test)],
    )
    y_test = pd.Series(
        np.random.choice([0, 1], n_test, p=[0.3, 0.7]),
        index=X_test.index,
        name="diagnosis",
    )

    return X_train, X_test, y_train, y_test


def test_check_index_leakage_clean(
    clean_train_test_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
) -> None:
    """Verifica que no haya fuga de índices cuando son disjuntos."""
    X_train, X_test, _, _ = clean_train_test_data
    res = check_index_leakage(X_train, X_test)

    assert res["status"] == "passed"
    assert res["overlap_count"] == 0
    assert res["leakage_ratio"] == 0.0


def test_check_index_leakage_detected(
    clean_train_test_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
) -> None:
    """Detecta fuga cuando existen índices compartidos entre train y test."""
    X_train, X_test, _, _ = clean_train_test_data
    # Forzar solapamiento de índices
    X_test_leaked = X_test.copy()
    X_test_leaked.index = list(X_train.index[: len(X_test)])

    res = check_index_leakage(X_train, X_test_leaked)
    assert res["status"] == "failed"
    assert res["overlap_count"] == len(X_test)
    assert res["leakage_ratio"] > 0.0


def test_check_sample_leakage_clean(
    clean_train_test_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
) -> None:
    """Verifica que no haya fuga de muestras cuando no hay filas idénticas."""
    X_train, X_test, _, _ = clean_train_test_data
    res = check_sample_leakage(X_train, X_test)

    assert res["status"] == "passed"
    assert res["duplicate_count"] == 0


def test_check_sample_leakage_detected(
    clean_train_test_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
) -> None:
    """Detecta mezcla de muestras cuando se duplican filas de test en train."""
    X_train, X_test, _, _ = clean_train_test_data
    # Inyectar muestras idénticas de test dentro de train
    X_train_contaminated = pd.concat([X_train, X_test.iloc[:15]])

    res = check_sample_leakage(X_train_contaminated, X_test)
    assert res["duplicate_count"] >= 15
    assert res["status"] in ("warning", "failed")


def test_check_dataset_sizes_valid(
    clean_train_test_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
) -> None:
    """Verifica proporciones de tamaño estándar (200 train, 50 test = 20%)."""
    X_train, X_test, _, _ = clean_train_test_data
    res = check_dataset_sizes(X_train, X_test, expected_test_ratio=0.20)

    assert res["status"] == "passed"
    assert res["train_size"] == 200
    assert res["test_size"] == 50
    assert res["actual_test_ratio"] == 0.20


def test_check_dataset_sizes_empty() -> None:
    """Falla si los conjuntos están vacíos."""
    res = check_dataset_sizes(pd.DataFrame(), pd.DataFrame())
    assert res["status"] == "failed"


def test_check_new_categories_clean(
    clean_train_test_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
) -> None:
    """Pasa cuando no hay categorías no vistas en test."""
    X_train, X_test, _, _ = clean_train_test_data
    res = check_new_categories(X_train, X_test)

    assert res["status"] == "passed"
    assert len(res["new_categories"]) == 0


def test_check_new_categories_detected(
    clean_train_test_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
) -> None:
    """Detecta categorías en test no presentes en train."""
    X_train, X_test, _, _ = clean_train_test_data
    X_test_new_cat = X_test.copy()
    X_test_new_cat.iloc[0, X_test_new_cat.columns.get_loc("gender")] = "Non-binary"

    res = check_new_categories(X_train, X_test_new_cat)
    assert res["status"] == "failed"
    assert "gender" in res["new_categories"]
    assert "Non-binary" in res["new_categories"]["gender"]


def test_check_label_distribution_clean(
    clean_train_test_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
) -> None:
    """Pasa cuando la distribución del target es similar y contiene ambas clases."""
    _, _, y_train, y_test = clean_train_test_data
    res = check_label_distribution(y_train, y_test)

    assert res["status"] == "passed"
    assert "proportion_difference" in res


def test_check_label_distribution_missing_class(
    clean_train_test_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
) -> None:
    """Falla si una clase clínica está totalmente ausente en el conjunto de prueba."""
    _, _, y_train, _ = clean_train_test_data
    y_test_no_zeros = pd.Series([1] * 50, name="diagnosis")

    res = check_label_distribution(y_train, y_test_no_zeros)
    assert res["status"] == "failed"
    assert "Falta al menos una clase" in res["message"] or "Clases ausentes" in res["message"]


def test_check_feature_drift_clean(
    clean_train_test_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
) -> None:
    """No detecta desvío cuando las muestras provienen de la misma distribución."""
    X_train, X_test, _, _ = clean_train_test_data
    res = check_feature_drift(X_train, X_test)

    assert res["status"] == "passed"
    assert res["drifted_columns_count"] == 0


def test_check_feature_drift_detected(
    clean_train_test_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
) -> None:
    """Detecta desvío cuando una variable en test sufre un cambio drástico de distribución."""
    X_train, X_test, _, _ = clean_train_test_data
    X_test_drifted = X_test.copy()
    X_test_drifted["total_bilirubin"] = X_test_drifted["total_bilirubin"] + 50.0

    res = check_feature_drift(X_train, X_test_drifted)
    assert "total_bilirubin" in res["drifted_columns"]
    assert res["features"]["total_bilirubin"]["drift_detected"] is True


def test_validate_train_test_split_full_success(
    clean_train_test_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    tmp_path: Path,
) -> None:
    """Verifica la ejecución completa exitosa de validate_train_test_split y generación de reportes."""
    X_train, X_test, y_train, y_test = clean_train_test_data
    output_dir = tmp_path / "split_reports"

    results = validate_train_test_split(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        output_dir=output_dir,
        raise_on_error=True,
    )

    assert results["passed"] is True
    assert results["status"] in ("passed", "warning")
    assert len(results["errors"]) == 0
    assert "index_leakage" in results["checks"]
    assert "sample_leakage" in results["checks"]
    assert "dataset_sizes" in results["checks"]
    assert "new_categories" in results["checks"]
    assert "label_drift" in results["checks"]
    assert "feature_drift" in results["checks"]

    # Verificar existencia de los reportes en disco
    json_path = output_dir / "train_test_validation_report.json"
    html_path = output_dir / "train_test_validation_report.html"
    assert json_path.exists()
    assert html_path.exists()

    with open(json_path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["passed"] is True


def test_validate_train_test_split_raise_on_error(
    clean_train_test_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
) -> None:
    """Lanza TrainTestSplitValidationError cuando raise_on_error es True y hay fallos críticos."""
    X_train, X_test, y_train, _ = clean_train_test_data
    # Crear fallo crítico: test sin clase 0
    y_test_invalid = pd.Series([1] * len(X_test), index=X_test.index)

    with pytest.raises(TrainTestSplitValidationError, match="Fallo en la validación"):
        validate_train_test_split(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test_invalid,
            raise_on_error=True,
        )


def test_validate_train_test_split_no_raise_on_error(
    clean_train_test_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
) -> None:
    """No lanza excepción cuando raise_on_error es False, pero marca status como failed."""
    X_train, X_test, y_train, _ = clean_train_test_data
    y_test_invalid = pd.Series([1] * len(X_test), index=X_test.index)

    results = validate_train_test_split(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test_invalid,
        raise_on_error=False,
    )

    assert results["passed"] is False
    assert results["status"] == "failed"
    assert len(results["errors"]) > 0
