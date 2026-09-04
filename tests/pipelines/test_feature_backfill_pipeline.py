"""Tests unitarios para el pipeline de backfill src.pipelines.feature_backfill."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.pipelines.feature_backfill import load_dataset, main


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Fixture con un DataFrame simulado para el pipeline."""
    return pd.DataFrame(
        {
            "Age": [50, 40],
            "Gender": ["Male", "Female"],
            "Total_Bilirubin": [1.0, 0.8],
            "Direct_Bilirubin": [0.3, 0.2],
            "Alkaline_Phosphotase": [150.0, 180.0],
            "Alamine_Aminotransferase": [25.0, 30.0],
            "Aspartate_Aminotransferase": [28.0, 32.0],
            "Total_Protiens": [7.0, 6.5],
            "Albumin": [3.5, 3.2],
            "Albumin_and_Globulin_Ratio": [1.0, 0.9],
            "Diagnosis": [1.0, 2.0],
        }
    )


def test_load_dataset_custom_parquet(tmp_path: Path, sample_dataframe: pd.DataFrame) -> None:
    """Verifica la carga de datos desde un archivo parquet personalizado."""
    parquet_path = tmp_path / "test_data.parquet"
    sample_dataframe.to_parquet(parquet_path)

    df_loaded = load_dataset(parquet_path)
    assert df_loaded.shape == sample_dataframe.shape
    assert list(df_loaded.columns) == list(sample_dataframe.columns)


def test_load_dataset_custom_csv(tmp_path: Path, sample_dataframe: pd.DataFrame) -> None:
    """Verifica la carga de datos desde un archivo CSV personalizado."""
    csv_path = tmp_path / "test_data.csv"
    sample_dataframe.to_csv(csv_path, index=False)

    df_loaded = load_dataset(csv_path)
    assert df_loaded.shape == sample_dataframe.shape


def test_load_dataset_custom_raw_csv_with_dataset_col(tmp_path: Path) -> None:
    """Verifica la carga y renombre automático de Dataset a Diagnosis en CSV raw."""
    csv_path = tmp_path / "raw_data.csv"
    raw_df = pd.DataFrame(
        {
            "Age": [50],
            "Gender": ["Male"],
            "Total_Bilirubin": [1.0],
            "Direct_Bilirubin": [0.3],
            "Alkaline_Phosphotase": [150.0],
            "Alamine_Aminotransferase": [25.0],
            "Aspartate_Aminotransferase": [28.0],
            "Total_Protiens": [7.0],
            "Albumin": [3.5],
            "Albumin_and_Globulin_Ratio": [1.0],
            "Dataset": [1],
            "Extra_Empty_Col": ["null"],
        }
    )
    raw_df.to_csv(csv_path, index=False)

    df_loaded = load_dataset(csv_path)
    assert "Diagnosis" in df_loaded.columns
    assert "Dataset" not in df_loaded.columns
    assert "Extra_Empty_Col" not in df_loaded.columns


def test_load_dataset_custom_path_not_found(tmp_path: Path) -> None:
    """Verifica que se lance FileNotFoundError si la ruta personalizada no existe."""
    non_existent = tmp_path / "non_existent.parquet"
    with pytest.raises(FileNotFoundError, match="La ruta de datos especificada no existe"):
        load_dataset(non_existent)


def test_load_dataset_file_not_found() -> None:
    """Verifica que se lance FileNotFoundError si no se encuentra ningún archivo."""
    with (
        patch("src.pipelines.feature_backfill.INTERMEDIATE_DATA_PATH") as mock_inter,
        patch("src.pipelines.feature_backfill.RAW_DATA_PATH") as mock_raw,
    ):
        mock_inter.exists.return_value = False
        mock_raw.exists.return_value = False

        with pytest.raises(FileNotFoundError, match="No se encontró el archivo de datos"):
            load_dataset()


def test_main_dry_run(sample_dataframe: pd.DataFrame) -> None:
    """Verifica la ejecución de main en modo --dry-run."""
    with (
        patch("src.pipelines.feature_backfill.load_dataset", return_value=sample_dataframe),
        patch("sys.argv", ["feature_backfill", "--dry-run"]),
    ):
        exit_code = main()
        assert exit_code == 0


def test_main_success(sample_dataframe: pd.DataFrame) -> None:
    """Verifica la ejecución exitosa de main llamando al backfill con argumentos explícitos."""
    mock_result = {
        "status": "success",
        "feature_group_name": "pacientes_higado_fg",
        "version": 2,
        "records_inserted": len(sample_dataframe),
    }

    with (
        patch("src.pipelines.feature_backfill.load_dataset", return_value=sample_dataframe),
        patch(
            "src.pipelines.feature_backfill.backfill_patient_features_to_store",
            return_value=mock_result,
        ) as mock_backfill,
        patch(
            "sys.argv", ["feature_backfill", "--fg-name", "pacientes_higado_fg", "--version", "2"]
        ),
    ):
        exit_code = main()
        assert exit_code == 0
        mock_backfill.assert_called_once_with(
            df=sample_dataframe,
            feature_group_name="pacientes_higado_fg",
            version=2,
        )


def test_main_default_version(sample_dataframe: pd.DataFrame) -> None:
    """Verifica que main use version=2 por defecto cuando no se especifica --version."""
    mock_result = {
        "status": "success",
        "feature_group_name": "pacientes_higado_fg",
        "version": 2,
        "records_inserted": len(sample_dataframe),
    }

    with (
        patch("src.pipelines.feature_backfill.load_dataset", return_value=sample_dataframe),
        patch(
            "src.pipelines.feature_backfill.backfill_patient_features_to_store",
            return_value=mock_result,
        ) as mock_backfill,
        patch("sys.argv", ["feature_backfill"]),
    ):
        exit_code = main()
        assert exit_code == 0
        mock_backfill.assert_called_once_with(
            df=sample_dataframe,
            feature_group_name="pacientes_higado_fg",
            version=2,
        )


def test_main_exception() -> None:
    """Verifica que main capture excepciones y devuelva código de salida 1."""
    with (
        patch(
            "src.pipelines.feature_backfill.load_dataset",
            side_effect=RuntimeError("Error simulado"),
        ),
        patch("sys.argv", ["feature_backfill"]),
    ):
        exit_code = main()
        assert exit_code == 1
