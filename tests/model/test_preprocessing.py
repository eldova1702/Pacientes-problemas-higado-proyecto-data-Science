"""Pruebas unitarias para el módulo de preprocesamiento clínico (src/model/preprocessing.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model.preprocessing import (
    ClinicalFeatureBuilder,
    ColumnStandardizer,
    build_feature_pipeline,
    load_supervised_data,
    prepare_supervised_data,
)


@pytest.fixture
def sample_raw_dataframe() -> pd.DataFrame:
    """Fixture con un conjunto reducido de pacientes con nombres en TitleCase."""
    return pd.DataFrame(
        {
            "Age": [55, 42, 30, 60, 48],
            "Gender": ["Male", "Female", "Male", "Male", "Female"],
            "Total_Bilirubin": [1.2, 0.8, 2.5, 0.9, 1.4],
            "Direct_Bilirubin": [0.4, 0.2, 1.0, 0.3, 0.5],
            "Alkaline_Phosphotase": [210, 180, 290, 160, 220],
            "Alamine_Aminotransferase": [35, 28, 45, 22, 38],
            "Aspartate_Aminotransferase": [42, 32, 60, 25, 46],
            "Total_Protiens": [6.8, 7.1, 5.9, 6.5, 7.0],
            "Albumin": [3.2, 3.5, 2.8, 3.0, 3.4],
            "Albumin_and_Globulin_Ratio": [0.9, 1.0, 0.8, 0.9, 0.95],
            "Diagnosis": [1, 2, 1, 2, 1],
        }
    )


@pytest.fixture
def sample_snake_dataframe() -> pd.DataFrame:
    """Fixture con características en snake_case (provenientes de Feature Store)."""
    return pd.DataFrame(
        {
            "patient_id": [101, 102, 103, 104],
            "event_time": pd.to_datetime(
                ["2026-09-01", "2026-09-01", "2026-09-01", "2026-09-01"], utc=True
            ),
            "age": [50.0, 45.0, 62.0, 38.0],
            "gender": ["Male", "Female", "Male", "Female"],
            "total_bilirubin": [1.0, 0.7, 3.2, 1.1],
            "direct_bilirubin": [0.3, 0.2, 1.4, 0.4],
            "alkaline_phosphotase": [200.0, 150.0, 310.0, 190.0],
            "alamine_aminotransferase": [30.0, 22.0, 55.0, 28.0],
            "aspartate_aminotransferase": [40.0, 26.0, 75.0, 34.0],
            "total_protiens": [6.5, 7.2, 5.8, 6.7],
            "albumin": [3.1, 3.6, 2.7, 3.3],
            "albumin_and_globulin_ratio": [0.9, 1.1, 0.7, 0.95],
            "direct_to_total_bilirubin": [0.3, 0.2857, 0.4375, 0.3636],
            "ast_to_alt": [1.3333, 1.1818, 1.3636, 1.2142],
            "diagnosis": [1.0, 2.0, 1.0, 2.0],
        }
    )


def test_column_standardizer(sample_raw_dataframe: pd.DataFrame) -> None:
    """Verifica que ColumnStandardizer convierta todos los nombres a minúsculas."""
    standardizer = ColumnStandardizer()
    transformed = standardizer.fit_transform(sample_raw_dataframe)

    for col in transformed.columns:
        assert col == col.lower()
    assert "total_bilirubin" in transformed.columns
    assert "gender" in transformed.columns

    names_out = standardizer.get_feature_names_out()
    assert all(c == c.lower() for c in names_out)


def test_clinical_feature_builder_calculates_ratios(sample_raw_dataframe: pd.DataFrame) -> None:
    """Verifica el cálculo correcto de direct_to_total_bilirubin y ast_to_alt."""
    standardizer = ColumnStandardizer()
    df_std = standardizer.fit_transform(sample_raw_dataframe)

    builder = ClinicalFeatureBuilder()
    transformed = builder.fit_transform(df_std)

    assert "direct_to_total_bilirubin" in transformed.columns
    assert "ast_to_alt" in transformed.columns

    expected_ratio = df_std["direct_bilirubin"] / df_std["total_bilirubin"]
    np.testing.assert_allclose(transformed["direct_to_total_bilirubin"], expected_ratio)

    expected_ast_alt = df_std["aspartate_aminotransferase"] / df_std["alamine_aminotransferase"]
    np.testing.assert_allclose(transformed["ast_to_alt"], expected_ast_alt)

    feature_names = builder.get_feature_names_out(df_std.columns)
    assert "direct_to_total_bilirubin" in feature_names
    assert "ast_to_alt" in feature_names


def test_clinical_feature_builder_preserves_existing_ratios(
    sample_snake_dataframe: pd.DataFrame,
) -> None:
    """Verifica que no sobreescriba ni duplique ratios si ya están calculados."""
    builder = ClinicalFeatureBuilder()
    transformed = builder.fit_transform(sample_snake_dataframe)

    assert "direct_to_total_bilirubin" in transformed.columns
    assert "ast_to_alt" in transformed.columns
    assert transformed.shape[1] == sample_snake_dataframe.shape[1]


def test_clinical_feature_builder_zero_division() -> None:
    """Verifica que la división por cero resulte en NaN sin lanzar excepciones."""
    df_zero = pd.DataFrame(
        {
            "total_bilirubin": [0.0, 1.0],
            "direct_bilirubin": [0.2, 0.5],
            "alamine_aminotransferase": [0.0, 20.0],
            "aspartate_aminotransferase": [30.0, 25.0],
        }
    )
    builder = ClinicalFeatureBuilder()
    transformed = builder.fit_transform(df_zero)

    assert np.isnan(transformed.loc[0, "direct_to_total_bilirubin"])
    assert np.isnan(transformed.loc[0, "ast_to_alt"])
    assert not np.isnan(transformed.loc[1, "direct_to_total_bilirubin"])


def test_build_feature_pipeline_fit_transform(sample_raw_dataframe: pd.DataFrame) -> None:
    """Verifica que el pipeline de preprocesamiento transforme correctamente los datos."""
    features = sample_raw_dataframe.drop(columns=["Diagnosis"])
    pipeline = build_feature_pipeline()

    transformed = pipeline.fit_transform(features)
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == len(features)
    assert not np.isnan(transformed).any()


def test_build_feature_pipeline_titlecase_and_snakecase(
    sample_raw_dataframe: pd.DataFrame,
) -> None:
    """Verifica que el pipeline acepte tanto TitleCase como snake_case."""
    features_title = sample_raw_dataframe.drop(columns=["Diagnosis"])
    features_lower = features_title.copy()
    features_lower.columns = [c.lower() for c in features_lower.columns]

    pipeline = build_feature_pipeline()
    transformed_title = pipeline.fit_transform(features_title)
    transformed_lower = pipeline.transform(features_lower)

    np.testing.assert_allclose(transformed_title, transformed_lower)


def test_prepare_supervised_data_success(sample_raw_dataframe: pd.DataFrame) -> None:
    """Verifica el mapeo correcto del target clínico (1 -> 1, 2 -> 0)."""
    X, y = prepare_supervised_data(sample_raw_dataframe, target="Diagnosis")

    assert "Diagnosis" not in X.columns
    assert set(y.unique()).issubset({0, 1})
    assert len(y) == len(X)
    # Fila 0 tenía Diagnosis=1 -> mapea a 1
    assert y.iloc[0] == 1
    # Fila 1 tenía Diagnosis=2 -> mapea a 0
    assert y.iloc[1] == 0


def test_prepare_supervised_data_drops_metadata(sample_snake_dataframe: pd.DataFrame) -> None:
    """Verifica que se excluyan patient_id y event_time de las características."""
    X, y = prepare_supervised_data(sample_snake_dataframe, target="diagnosis")

    assert "patient_id" not in X.columns
    assert "event_time" not in X.columns
    assert "diagnosis" not in X.columns
    assert set(y.unique()).issubset({0, 1})


def test_prepare_supervised_data_missing_target(sample_raw_dataframe: pd.DataFrame) -> None:
    """Lanza KeyError si la columna target no existe."""
    with pytest.raises(KeyError, match="No se encontró la columna objetivo"):
        prepare_supervised_data(sample_raw_dataframe, target="non_existent_col")


def test_prepare_supervised_data_invalid_values() -> None:
    """Lanza ValueError si el target tiene valores fuera de [1, 2] o [0, 1]."""
    df_invalid = pd.DataFrame(
        {
            "Age": [40, 50],
            "Diagnosis": [99, 100],
        }
    )
    with pytest.raises(ValueError, match="El target contiene valores inválidos"):
        prepare_supervised_data(df_invalid, target="Diagnosis")


def test_load_supervised_data(tmp_path: Path, sample_raw_dataframe: pd.DataFrame) -> None:
    """Verifica la carga desde archivo Parquet y CSV temporal, incluyendo extensiones en mayúsculas."""
    parquet_path = tmp_path / "patients.parquet"
    sample_raw_dataframe.to_parquet(parquet_path)

    X_pq, y_pq = load_supervised_data(parquet_path)
    assert len(X_pq) == len(sample_raw_dataframe)
    assert len(y_pq) == len(sample_raw_dataframe)

    # Test con extensión en mayúsculas (.PARQUET)
    upper_pq_path = tmp_path / "patients.PARQUET"
    sample_raw_dataframe.to_parquet(upper_pq_path)
    X_upq, y_upq = load_supervised_data(upper_pq_path)
    assert len(X_upq) == len(sample_raw_dataframe)
    assert len(y_upq) == len(sample_raw_dataframe)

    csv_path = tmp_path / "patients.csv"
    sample_raw_dataframe.to_csv(csv_path, index=False)

    X_csv, y_csv = load_supervised_data(csv_path)
    assert len(X_csv) == len(sample_raw_dataframe)
    assert len(y_csv) == len(sample_raw_dataframe)


def test_column_standardizer_handles_spaces_and_special_characters() -> None:
    """Verifica que ColumnStandardizer convierta espacios y caracteres no alfanuméricos a snake_case."""
    df_messy = pd.DataFrame(
        {
            "Total Bilirubin": [1.0],
            "Direct-Bilirubin": [0.5],
            "AST / ALT Ratio": [1.2],
            "  Age  ": [45],
        }
    )
    standardizer = ColumnStandardizer()
    res = standardizer.fit_transform(df_messy)

    assert list(res.columns) == [
        "total_bilirubin",
        "direct_bilirubin",
        "ast_alt_ratio",
        "age",
    ]
    assert list(standardizer.get_feature_names_out()) == [
        "total_bilirubin",
        "direct_bilirubin",
        "ast_alt_ratio",
        "age",
    ]
