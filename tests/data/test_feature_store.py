"""Tests unitarios para el módulo src.data.feature_store."""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.feature_store import (
    DEFAULT_HISTORICAL_TIMESTAMP,
    backfill_patient_features_to_store,
    get_hopsworks_project,
    get_or_create_patient_feature_group,
    prepare_patient_features_for_feature_store,
)


@pytest.fixture
def sample_raw_dataframe() -> pd.DataFrame:
    """Fixture con un DataFrame simulado de pacientes en formato crudo/intermedio."""
    return pd.DataFrame(
        {
            "Age": [65, 62, 45],
            "Gender": ["Female", "Male", "Male"],
            "Total_Bilirubin": [0.7, 7.3, 1.2],
            "Direct_Bilirubin": [0.1, 4.1, 0.4],
            "Alkaline_Phosphotase": [187.0, 690.0, 200.0],
            "Alamine_Aminotransferase": [16.0, 64.0, 30.0],
            "Aspartate_Aminotransferase": [18.0, 98.0, 40.0],
            "Total_Protiens": [6.8, 5.5, 6.0],
            "Albumin": [3.3, 1.8, 3.0],
            "Albumin_and_Globulin_Ratio": [0.9, 0.74, 1.0],
            "Diagnosis": [1.0, 1.0, 2.0],
        }
    )


def test_prepare_patient_features_column_names(sample_raw_dataframe: pd.DataFrame) -> None:
    """Verifica que las columnas se transformen correctamente a snake_case en minúsculas."""
    prepared = prepare_patient_features_for_feature_store(sample_raw_dataframe)

    expected_cols = [
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
    ]
    assert list(prepared.columns) == expected_cols


def test_prepare_patient_features_primary_key(sample_raw_dataframe: pd.DataFrame) -> None:
    """Verifica la generación de la clave primaria única patient_id."""
    prepared = prepare_patient_features_for_feature_store(sample_raw_dataframe, start_id=101)

    assert "patient_id" in prepared.columns
    assert list(prepared["patient_id"]) == [101, 102, 103]


def test_prepare_patient_features_timestamp(sample_raw_dataframe: pd.DataFrame) -> None:
    """Verifica la generación de la marca temporal event_time con timestamp explícito."""
    fixed_time = datetime.datetime(2026, 8, 31, 12, 0, 0, tzinfo=datetime.UTC)
    prepared = prepare_patient_features_for_feature_store(
        sample_raw_dataframe, base_timestamp=fixed_time
    )

    assert "event_time" in prepared.columns
    assert len(prepared["event_time"].dropna()) == len(sample_raw_dataframe)
    assert prepared["event_time"].iloc[0] == fixed_time


def test_prepare_patient_features_default_timestamp(sample_raw_dataframe: pd.DataFrame) -> None:
    """Verifica que sin base_timestamp se use DEFAULT_HISTORICAL_TIMESTAMP de forma determinista."""
    prepared = prepare_patient_features_for_feature_store(sample_raw_dataframe)

    assert "event_time" in prepared.columns
    assert prepared["event_time"].iloc[0] == DEFAULT_HISTORICAL_TIMESTAMP


def test_prepare_patient_features_values(sample_raw_dataframe: pd.DataFrame) -> None:
    """Verifica que los valores procesados correspondan a las observaciones de entrada."""
    prepared = prepare_patient_features_for_feature_store(sample_raw_dataframe)

    assert list(prepared["age"]) == [65.0, 62.0, 45.0]
    assert list(prepared["gender"]) == ["Female", "Male", "Male"]
    assert list(prepared["diagnosis"]) == [1.0, 1.0, 2.0]


def test_get_hopsworks_project_missing_key() -> None:
    """Verifica que se lance ValueError si no se proporciona API key ni variable de entorno."""
    with (
        patch.dict("os.environ", {}, clear=True),
        patch("src.data.feature_store.load_dotenv"),
        patch.dict("sys.modules", {"hopsworks": MagicMock()}),
        pytest.raises(ValueError, match="HOPSWORKS_API_KEY"),
    ):
        get_hopsworks_project(api_key=None)


def test_get_or_create_patient_feature_group() -> None:
    """Verifica la invocación a fs.get_or_create_feature_group con los argumentos correctos."""
    mock_fs = MagicMock()
    mock_fg = MagicMock()
    mock_fs.get_or_create_feature_group.return_value = mock_fg

    fg = get_or_create_patient_feature_group(
        fs=mock_fs,
        name="test_pacientes_fg",
        version=2,
        primary_key=["patient_id"],
        event_time="event_time",
        time_travel_format="HUDI",
    )

    assert fg == mock_fg
    mock_fs.get_or_create_feature_group.assert_called_once_with(
        name="test_pacientes_fg",
        version=2,
        description="Grupo de características históricas de pacientes para detección de enfermedad hepática",
        primary_key=["patient_id"],
        event_time="event_time",
        online_enabled=False,
        time_travel_format="HUDI",
    )


def test_backfill_patient_features_to_store_flow(sample_raw_dataframe: pd.DataFrame) -> None:
    """Verifica el flujo completo de backfill usando mocks para Hopsworks."""
    mock_project = MagicMock()
    mock_project.name = "test_project"
    mock_fs = MagicMock()
    mock_fg = MagicMock()

    mock_project.get_feature_store.return_value = mock_fs
    mock_fs.get_or_create_feature_group.return_value = mock_fg

    expected_version = 2
    with patch(
        "src.data.feature_store.get_hopsworks_project", return_value=mock_project
    ) as mock_login:
        result = backfill_patient_features_to_store(
            df=sample_raw_dataframe,
            feature_group_name="pacientes_higado_fg",
            version=expected_version,
            api_key="fake_key",
            project_name="test_project",
            wait_for_job=True,
        )

        mock_login.assert_called_once_with(api_key="fake_key", project_name="test_project")
        mock_fg.insert.assert_called_once()
        assert result["status"] == "success"
        assert result["records_inserted"] == len(sample_raw_dataframe)
        assert result["feature_group_name"] == "pacientes_higado_fg"
        assert result["version"] == expected_version
        assert result["metadata_warnings"] == []


def test_backfill_patient_features_metadata_warning(sample_raw_dataframe: pd.DataFrame) -> None:
    """Verifica que se capturen advertencias si alguna descripción no se puede actualizar."""
    mock_project = MagicMock()
    mock_fs = MagicMock()
    mock_fg = MagicMock()
    mock_fg.update_feature_description.side_effect = RuntimeError("Error de metadatos")

    mock_project.get_feature_store.return_value = mock_fs
    mock_fs.get_or_create_feature_group.return_value = mock_fg

    with patch("src.data.feature_store.get_hopsworks_project", return_value=mock_project):
        result = backfill_patient_features_to_store(
            df=sample_raw_dataframe,
            feature_group_name="pacientes_higado_fg",
            version=2,
            api_key="fake_key",
            project_name="test_project",
        )
        assert len(result["metadata_warnings"]) > 0
