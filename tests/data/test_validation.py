"""Tests unitarios para el módulo de validación de datos src.data.validation."""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

from src.data.validation import (
    DataValidationError,
    get_patient_features_schema,
    validate_patient_features,
)


@pytest.fixture
def valid_features_dataframe() -> pd.DataFrame:
    """Fixture con un DataFrame válido de 3 filas con las 15 características clínicas."""
    now_utc = pd.to_datetime(["2026-09-04 12:00:00"] * 3, utc=True)
    return pd.DataFrame(
        {
            "patient_id": [1, 2, 3],
            "event_time": now_utc,
            "age": [50.0, 42.0, 65.0],
            "gender": ["Male", "Female", "Male"],
            "total_bilirubin": [1.5, 0.8, 2.3],
            "direct_bilirubin": [0.6, 0.2, 1.1],
            "alkaline_phosphotase": [180.0, 150.0, 220.0],
            "alamine_aminotransferase": [35.0, 20.0, 45.0],
            "aspartate_aminotransferase": [40.0, 25.0, 50.0],
            "total_protiens": [7.0, 6.8, 6.5],
            "albumin": [3.5, 3.4, 3.2],
            "albumin_and_globulin_ratio": [1.0, 1.0, 0.9],
            "diagnosis": [1.0, 2.0, 1.0],
            "direct_to_total_bilirubin": [0.4, 0.25, 0.478],
            "ast_to_alt": [1.14, 1.25, 1.11],
        }
    )


def test_validate_patient_features_success(valid_features_dataframe: pd.DataFrame) -> None:
    """Verifica que un DataFrame con datos clínicamente correctos sea validado exitosamente."""
    validated = validate_patient_features(valid_features_dataframe)
    assert validated.shape == valid_features_dataframe.shape
    assert list(validated.columns) == list(valid_features_dataframe.columns)


def test_validate_patient_features_out_of_range(valid_features_dataframe: pd.DataFrame) -> None:
    """Verifica que valores fisiológicamente imposibles disparen DataValidationError."""
    corrupted_df = valid_features_dataframe.copy()
    corrupted_df.loc[0, "age"] = -5.0
    corrupted_df.loc[1, "total_protiens"] = 25.0

    with pytest.raises(DataValidationError, match="Los datos no cumplen con las reglas de calidad"):
        validate_patient_features(corrupted_df)


def test_validate_patient_features_invalid_gender(valid_features_dataframe: pd.DataFrame) -> None:
    """Verifica que categorías no permitidas en género disparen error de validación."""
    corrupted_df = valid_features_dataframe.copy()
    corrupted_df.loc[0, "gender"] = "Unknown"

    with pytest.raises(DataValidationError, match="gender"):
        validate_patient_features(corrupted_df)


def test_validate_patient_features_invalid_diagnosis(
    valid_features_dataframe: pd.DataFrame,
) -> None:
    """Verifica que valores de diagnóstico ajenos a 1 o 2 sean rechazados."""
    corrupted_df = valid_features_dataframe.copy()
    corrupted_df.loc[0, "diagnosis"] = 99.0

    with pytest.raises(DataValidationError, match="diagnosis"):
        validate_patient_features(corrupted_df)


def test_validate_patient_features_duplicate_patient_id(
    valid_features_dataframe: pd.DataFrame,
) -> None:
    """Verifica que identificadores duplicados de pacientes violen la restricción de unicidad."""
    corrupted_df = valid_features_dataframe.copy()
    corrupted_df.loc[1, "patient_id"] = corrupted_df.loc[0, "patient_id"]

    with pytest.raises(DataValidationError, match="patient_id"):
        validate_patient_features(corrupted_df)


def test_validate_patient_features_null_patient_id(valid_features_dataframe: pd.DataFrame) -> None:
    """Verifica que patient_id no admita valores nulos."""
    corrupted_df = valid_features_dataframe.copy()
    corrupted_df["patient_id"] = [1, 2, None]

    with pytest.raises(DataValidationError, match="patient_id"):
        validate_patient_features(corrupted_df)


def test_validate_patient_features_naive_event_time(
    valid_features_dataframe: pd.DataFrame,
) -> None:
    """Verifica que marcas de tiempo sin zona horaria UTC fallen la validación."""
    corrupted_df = valid_features_dataframe.copy()
    corrupted_df["event_time"] = [datetime.datetime(2026, 9, 1, 12, 0, 0)] * 3

    with pytest.raises(DataValidationError, match="event_time"):
        validate_patient_features(corrupted_df)


def test_validate_patient_features_negative_clinical_ratio(
    valid_features_dataframe: pd.DataFrame,
) -> None:
    """Verifica que razones clínicas negativas sean rechazadas."""
    corrupted_df = valid_features_dataframe.copy()
    corrupted_df.loc[0, "direct_to_total_bilirubin"] = -0.5

    with pytest.raises(DataValidationError, match="direct_to_total_bilirubin"):
        validate_patient_features(corrupted_df)


def test_validate_patient_features_excessive_nulls() -> None:
    """Verifica que superar el porcentaje máximo de nulos (5%) en lotes grandes lance error."""
    now_utc = pd.to_datetime(["2026-09-04 12:00:00"] * 30, utc=True)
    # 30 filas con 10 nulos en age (33.3% > 5%)
    corrupted_df = pd.DataFrame(
        {
            "patient_id": range(1, 31),
            "event_time": now_utc,
            "age": [50.0] * 20 + [None] * 10,
            "gender": ["Male"] * 30,
            "total_bilirubin": [1.0] * 30,
            "direct_bilirubin": [0.3] * 30,
            "alkaline_phosphotase": [150.0] * 30,
            "alamine_aminotransferase": [25.0] * 30,
            "aspartate_aminotransferase": [30.0] * 30,
            "total_protiens": [7.0] * 30,
            "albumin": [3.5] * 30,
            "albumin_and_globulin_ratio": [1.0] * 30,
            "diagnosis": [1.0] * 30,
            "direct_to_total_bilirubin": [0.3] * 30,
            "ast_to_alt": [1.2] * 30,
        }
    )

    with pytest.raises(DataValidationError, match="age"):
        validate_patient_features(corrupted_df)


def test_validate_patient_features_empty_dataframe() -> None:
    """Verifica que un DataFrame vacío sea rechazado."""
    empty_df = pd.DataFrame(
        columns=[
            "patient_id",
            "event_time",
            "age",
            "gender",
            "total_bilirubin",
            "direct_bilirubin",
            "alkaline_phosphotase",
            "alamine_aminotransferase",
            "aspartate_aminotransferase",
            "total_protiens",
            "albumin",
            "albumin_and_globulin_ratio",
            "diagnosis",
            "direct_to_total_bilirubin",
            "ast_to_alt",
        ]
    )

    with pytest.raises(DataValidationError, match=r"[vV]ac[ií]o"):
        validate_patient_features(empty_df)


def test_validate_patient_features_missing_column(valid_features_dataframe: pd.DataFrame) -> None:
    """Verifica que la ausencia de columnas obligatorias dispare error de esquema."""
    missing_col_df = valid_features_dataframe.drop(columns=["ast_to_alt"])

    with pytest.raises(DataValidationError, match="ast_to_alt"):
        validate_patient_features(missing_col_df)


def test_get_patient_features_schema_custom_null_fraction() -> None:
    """Verifica la generación de esquema con umbral de nulos personalizado."""
    schema = get_patient_features_schema(max_null_fraction=0.10)
    assert "patient_id" in schema.columns
    assert "age" in schema.columns
