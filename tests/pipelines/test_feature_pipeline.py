"""Tests unitarios para el Feature Pipeline src.pipelines.feature_pipeline."""

from __future__ import annotations

import datetime
import runpy
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.pipelines.feature_pipeline.feature_pipeline import (
    extract_data,
    load_features_to_store,
    main,
    run_feature_pipeline,
    transform_features,
)


@pytest.fixture
def sample_clinical_data() -> pd.DataFrame:
    """Fixture con mediciones clínicas para pruebas del pipeline."""
    return pd.DataFrame(
        {
            "Age": [50, 42, 60],
            "Gender": ["Male", "Female", "Male"],
            "Total_Bilirubin": [2.0, 1.0, 0.8],
            "Direct_Bilirubin": [1.0, 0.4, 0.2],
            "Alkaline_Phosphotase": [200.0, 150.0, 180.0],
            "Alamine_Aminotransferase": [40.0, 20.0, 30.0],
            "Aspartate_Aminotransferase": [60.0, 25.0, 30.0],
            "Total_Protiens": [7.0, 6.8, 6.5],
            "Albumin": [3.5, 3.4, 3.2],
            "Albumin_and_Globulin_Ratio": [1.0, 1.0, 0.9],
            "Diagnosis": [1.0, 2.0, 1.0],
        }
    )


# ==============================================================================
# Tests de Extracción (extract_data)
# ==============================================================================


def test_extract_data_custom_parquet(tmp_path: Path, sample_clinical_data: pd.DataFrame) -> None:
    """Verifica la extracción de datos desde archivo parquet personalizado."""
    parquet_file = tmp_path / "test_pacientes.parquet"
    sample_clinical_data.to_parquet(parquet_file)

    df_extracted = extract_data(parquet_file)
    assert df_extracted.shape == sample_clinical_data.shape
    assert list(df_extracted.columns) == list(sample_clinical_data.columns)


def test_extract_data_custom_csv(tmp_path: Path, sample_clinical_data: pd.DataFrame) -> None:
    """Verifica la extracción de datos desde archivo CSV personalizado."""
    csv_file = tmp_path / "test_pacientes.csv"
    sample_clinical_data.to_csv(csv_file, index=False)

    df_extracted = extract_data(csv_file)
    assert df_extracted.shape == sample_clinical_data.shape


def test_extract_data_raw_csv_with_dataset_col(tmp_path: Path) -> None:
    """Verifica que al leer CSV crudo con Dataset se aplique usecols y renombre a Diagnosis."""
    csv_file = tmp_path / "raw_pacientes.csv"
    raw_df = pd.DataFrame(
        {
            "Age": [55],
            "Gender": ["Male"],
            "Total_Bilirubin": [1.2],
            "Direct_Bilirubin": [0.4],
            "Alkaline_Phosphotase": [190.0],
            "Alamine_Aminotransferase": [35.0],
            "Aspartate_Aminotransferase": [45.0],
            "Total_Protiens": [6.9],
            "Albumin": [3.3],
            "Albumin_and_Globulin_Ratio": [0.9],
            "Dataset": [1],
            "Unwanted_Extra_Col": [None],
        }
    )
    raw_df.to_csv(csv_file, index=False)

    df_extracted = extract_data(csv_file)
    assert "Diagnosis" in df_extracted.columns
    assert "Dataset" not in df_extracted.columns
    assert "Unwanted_Extra_Col" not in df_extracted.columns


def test_extract_data_custom_path_not_found(tmp_path: Path) -> None:
    """Verifica que se lance FileNotFoundError si la ruta especificada no existe."""
    invalid_path = tmp_path / "missing_file.parquet"
    with pytest.raises(FileNotFoundError, match="La ruta de datos especificada no existe"):
        extract_data(invalid_path)


def test_extract_data_default_intermediate(
    tmp_path: Path, sample_clinical_data: pd.DataFrame
) -> None:
    """Verifica la carga por defecto desde INTERMEDIATE_DATA_PATH si existe."""
    inter_path = tmp_path / "inter.parquet"
    sample_clinical_data.to_parquet(inter_path)

    with patch(
        "src.pipelines.feature_pipeline.feature_pipeline.INTERMEDIATE_DATA_PATH", inter_path
    ):
        df_extracted = extract_data()
        assert df_extracted.shape == sample_clinical_data.shape


def test_extract_data_default_raw(tmp_path: Path) -> None:
    """Verifica la carga desde RAW_DATA_PATH si no existe intermediate."""
    raw_path = tmp_path / "raw.csv"
    raw_df = pd.DataFrame(
        {
            "Age": [50],
            "Gender": ["Female"],
            "Total_Bilirubin": [1.0],
            "Direct_Bilirubin": [0.2],
            "Alkaline_Phosphotase": [160.0],
            "Alamine_Aminotransferase": [22.0],
            "Aspartate_Aminotransferase": [28.0],
            "Total_Protiens": [7.1],
            "Albumin": [3.6],
            "Albumin_and_Globulin_Ratio": [1.0],
            "Dataset": [2],
        }
    )
    raw_df.to_csv(raw_path, index=False)

    non_existent = tmp_path / "none.parquet"
    with (
        patch(
            "src.pipelines.feature_pipeline.feature_pipeline.INTERMEDIATE_DATA_PATH", non_existent
        ),
        patch("src.pipelines.feature_pipeline.feature_pipeline.RAW_DATA_PATH", raw_path),
    ):
        df_extracted = extract_data()
        assert "Diagnosis" in df_extracted.columns


def test_extract_data_none_found(tmp_path: Path) -> None:
    """Verifica error si ni intermediate ni raw existen."""
    non_existent1 = tmp_path / "none1.parquet"
    non_existent2 = tmp_path / "none2.csv"
    with (
        patch(
            "src.pipelines.feature_pipeline.feature_pipeline.INTERMEDIATE_DATA_PATH",
            non_existent1,
        ),
        patch("src.pipelines.feature_pipeline.feature_pipeline.RAW_DATA_PATH", non_existent2),
        pytest.raises(FileNotFoundError, match="No se encontró el archivo de datos"),
    ):
        extract_data()


# ==============================================================================
# Tests de Transformación e Ingeniería de Features (transform_features)
# ==============================================================================


def test_transform_features_columns(sample_clinical_data: pd.DataFrame) -> None:
    """Verifica la presencia y orden de todas las columnas requeridas y derivadas."""
    transformed = transform_features(sample_clinical_data)

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
        "direct_to_total_bilirubin",
        "ast_to_alt",
    ]
    assert list(transformed.columns) == expected_cols


def test_transform_features_clinical_ratios(sample_clinical_data: pd.DataFrame) -> None:
    """Verifica la correcta computación matemática de los ratios clínicos."""
    transformed = transform_features(sample_clinical_data)

    # Fila 0: direct=1.0, total=2.0 -> ratio=0.5
    assert np.isclose(transformed.loc[0, "direct_to_total_bilirubin"], 0.5)

    # Fila 0: ast=60.0, alt=40.0 -> De Ritis ratio = 1.5
    assert np.isclose(transformed.loc[0, "ast_to_alt"], 1.5)

    # Fila 1: direct=0.4, total=1.0 -> ratio=0.4
    assert np.isclose(transformed.loc[1, "direct_to_total_bilirubin"], 0.4)

    # Fila 1: ast=25.0, alt=20.0 -> De Ritis ratio = 1.25
    assert np.isclose(transformed.loc[1, "ast_to_alt"], 1.25)


def test_transform_features_division_by_zero_handling() -> None:
    """Verifica que la división por cero en ratios clínicos produzca NaN y no inf ni error."""
    zero_df = pd.DataFrame(
        {
            "Total_Bilirubin": [0.0],
            "Direct_Bilirubin": [1.0],
            "Alamine_Aminotransferase": [0.0],
            "Aspartate_Aminotransferase": [50.0],
        }
    )
    transformed = transform_features(zero_df)

    assert pd.isna(transformed.loc[0, "direct_to_total_bilirubin"])
    assert pd.isna(transformed.loc[0, "ast_to_alt"])


def test_transform_features_dtypes(sample_clinical_data: pd.DataFrame) -> None:
    """Verifica que los tipos de datos sean compatibles y nativos."""
    transformed = transform_features(sample_clinical_data)

    assert transformed["patient_id"].dtype == "int64"
    assert pd.api.types.is_datetime64_any_dtype(transformed["event_time"])
    assert pd.api.types.is_string_dtype(transformed["gender"])
    assert pd.api.types.is_float_dtype(transformed["direct_to_total_bilirubin"])
    assert pd.api.types.is_float_dtype(transformed["ast_to_alt"])
    assert transformed["diagnosis"].dtype == "int64"


def test_transform_features_custom_patient_id_and_timestamp(
    sample_clinical_data: pd.DataFrame,
) -> None:
    """Verifica asignación de start_id y base_timestamp explícitos."""
    fixed_ts = datetime.datetime(2026, 9, 1, 10, 0, 0, tzinfo=datetime.UTC)
    transformed = transform_features(sample_clinical_data, start_id=500, base_timestamp=fixed_ts)

    assert list(transformed["patient_id"]) == [500, 501, 502]
    assert transformed["event_time"].iloc[0] == fixed_ts


def test_transform_features_preserves_existing_keys(sample_clinical_data: pd.DataFrame) -> None:
    """Verifica que si patient_id y event_time ya existen, se respeten."""
    existing_df = sample_clinical_data.copy()
    existing_df["patient_id"] = [10, 20, 30]
    existing_ts = pd.Timestamp("2025-01-01 00:00:00+00:00")
    existing_df["event_time"] = existing_ts

    transformed = transform_features(existing_df)
    assert list(transformed["patient_id"]) == [10, 20, 30]
    assert transformed["event_time"].iloc[0] == existing_ts


# ==============================================================================
# Tests de Carga en Feature Store (load_features_to_store)
# ==============================================================================


def test_load_features_to_store_flow(sample_clinical_data: pd.DataFrame) -> None:
    """Verifica la orquestación de la carga en Hopsworks con mocks."""
    transformed = transform_features(sample_clinical_data)

    mock_project = MagicMock()
    mock_project.name = "test_project"
    mock_fs = MagicMock()
    mock_fg = MagicMock()

    mock_project.get_feature_store.return_value = mock_fs
    mock_fs.get_or_create_feature_group.return_value = mock_fg

    with patch(
        "src.pipelines.feature_pipeline.feature_pipeline.get_hopsworks_project",
        return_value=mock_project,
    ) as mock_login:
        result = load_features_to_store(
            df=transformed,
            fg_name="pacientes_higado_fg",
            version=2,
            api_key="test_api_key",
            project_name="test_project",
            wait_for_job=True,
        )

        mock_login.assert_called_once_with(api_key="test_api_key", project_name="test_project")
        mock_fg.insert.assert_called_once_with(transformed, write_options={"wait_for_job": True})
        expected_records = len(sample_clinical_data)
        expected_version = 2
        assert result["status"] == "success"
        assert result["records_inserted"] == expected_records
        assert result["feature_group_name"] == "pacientes_higado_fg"
        assert result["version"] == expected_version
        assert result["metadata_warnings"] == []


def test_load_features_to_store_metadata_warnings(sample_clinical_data: pd.DataFrame) -> None:
    """Verifica que se registren advertencias de metadatos si update_feature_description falla."""
    transformed = transform_features(sample_clinical_data)

    mock_project = MagicMock()
    mock_fs = MagicMock()
    mock_fg = MagicMock()
    mock_fg.update_feature_description.side_effect = RuntimeError("Hopsworks metadata error")

    mock_project.get_feature_store.return_value = mock_fs
    mock_fs.get_or_create_feature_group.return_value = mock_fg

    with patch(
        "src.pipelines.feature_pipeline.feature_pipeline.get_hopsworks_project",
        return_value=mock_project,
    ):
        result = load_features_to_store(df=transformed, fg_name="test_fg", version=1)
        assert result["status"] == "success"
        assert len(result["metadata_warnings"]) > 0


# ==============================================================================
# Tests de Ejecución Completa (run_feature_pipeline y main CLI)
# ==============================================================================


def test_run_feature_pipeline_dry_run(tmp_path: Path, sample_clinical_data: pd.DataFrame) -> None:
    """Verifica la ejecución en modo dry-run sin interacción remota."""
    parquet_path = tmp_path / "patients.parquet"
    sample_clinical_data.to_parquet(parquet_path)

    expected_records = len(sample_clinical_data)
    result = run_feature_pipeline(data_path=parquet_path, dry_run=True)
    assert result["status"] == "dry_run_success"
    assert result["records_processed"] == expected_records
    assert "direct_to_total_bilirubin" in result["features"]
    assert "ast_to_alt" in result["features"]


def test_run_feature_pipeline_full_flow(tmp_path: Path, sample_clinical_data: pd.DataFrame) -> None:
    """Verifica la ejecución completa cuando dry_run=False."""
    parquet_path = tmp_path / "patients.parquet"
    sample_clinical_data.to_parquet(parquet_path)

    expected_records = len(sample_clinical_data)
    with patch(
        "src.pipelines.feature_pipeline.feature_pipeline.load_features_to_store",
        return_value={"status": "success", "records_inserted": expected_records},
    ) as mock_load:
        result = run_feature_pipeline(data_path=parquet_path, dry_run=False)
        mock_load.assert_called_once()
        assert result["status"] == "success"


def test_main_cli_dry_run(tmp_path: Path, sample_clinical_data: pd.DataFrame) -> None:
    """Verifica la interfaz CLI de main() con argumento --dry-run."""
    csv_path = tmp_path / "patients.csv"
    sample_clinical_data.to_csv(csv_path, index=False)

    test_args = ["feature_pipeline.py", "--data-path", str(csv_path), "--dry-run"]
    with patch("sys.argv", test_args):
        exit_code = main()
        assert exit_code == 0


def test_main_cli_error() -> None:
    """Verifica que main() capture excepciones y retorne código de error 1."""
    test_args = ["feature_pipeline.py", "--data-path", "invalid_non_existent.csv"]
    with patch("sys.argv", test_args):
        exit_code = main()
        assert exit_code == 1


def test_main_module_execution(tmp_path: Path, sample_clinical_data: pd.DataFrame) -> None:
    """Verifica la ejecución de src.pipelines.feature_pipeline.__main__ como módulo."""
    csv_path = tmp_path / "module_patients.csv"
    sample_clinical_data.to_csv(csv_path, index=False)
    test_args = ["__main__.py", "--data-path", str(csv_path), "--dry-run"]
    with patch("sys.argv", test_args), pytest.raises(SystemExit) as exc_info:
        runpy.run_module("src.pipelines.feature_pipeline", run_name="__main__")
    assert exc_info.value.code == 0
