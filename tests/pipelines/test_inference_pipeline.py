"""Pruebas unitarias para el Inference Pipeline (src/pipelines/inference_pipeline/)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.model.preprocessing import TARGET_COLUMN, build_feature_pipeline, prepare_supervised_data
from src.pipelines.inference_pipeline.inference_pipeline import (
    DEFAULT_MODEL_PATH,
    PROJECT_ROOT,
    InferencePipelineError,
    apply_training_transformations,
    build_portable_summary,
    extract_metadata,
    generate_inference_html_report,
    generate_predictions,
    load_input_data,
    load_model_artifact,
    main,
    plot_prediction_distribution,
    prepare_inference_features,
    run_inference_pipeline,
    save_inference_summary,
    save_predictions,
)

N_SYNTHETIC_ROWS = 20
EXPECTED_PREDICTION_COLUMNS = {
    "prediction",
    "prediction_label",
    "probability_disease",
    "probability_no_disease",
}


@pytest.fixture
def synthetic_raw_dataframe() -> pd.DataFrame:
    """Datos clínicos sintéticos en formato crudo (TitleCase) con metadatos y target."""
    rng = np.random.default_rng(42)
    labels = np.array([1, 2] * (N_SYNTHETIC_ROWS // 2))
    return pd.DataFrame(
        {
            "patient_id": range(1000, 1000 + N_SYNTHETIC_ROWS),
            "event_time": pd.to_datetime(["2026-09-01"] * N_SYNTHETIC_ROWS, utc=True),
            "Age": rng.uniform(20, 75, N_SYNTHETIC_ROWS).round(1),
            "Gender": rng.choice(["Male", "Female"], N_SYNTHETIC_ROWS),
            "Total_Bilirubin": rng.uniform(0.5, 8.0, N_SYNTHETIC_ROWS).round(2),
            "Direct_Bilirubin": rng.uniform(0.1, 3.0, N_SYNTHETIC_ROWS).round(2),
            "Alkaline_Phosphotase": rng.uniform(120, 500, N_SYNTHETIC_ROWS).round(1),
            "Alamine_Aminotransferase": rng.uniform(15, 120, N_SYNTHETIC_ROWS).round(1),
            "Aspartate_Aminotransferase": rng.uniform(20, 150, N_SYNTHETIC_ROWS).round(1),
            "Total_Protiens": rng.uniform(5.0, 8.5, N_SYNTHETIC_ROWS).round(2),
            "Albumin": rng.uniform(2.0, 4.8, N_SYNTHETIC_ROWS).round(2),
            "Albumin_and_Globulin_Ratio": rng.uniform(0.6, 1.5, N_SYNTHETIC_ROWS).round(2),
            "Diagnosis": labels,
        }
    )


@pytest.fixture
def dummy_pipeline(synthetic_raw_dataframe: pd.DataFrame) -> Pipeline:
    """Pipeline con preprocesamiento real y un DummyClassifier como clasificador."""
    X, y = prepare_supervised_data(synthetic_raw_dataframe, target=TARGET_COLUMN)
    pipeline = Pipeline(
        steps=[
            ("preprocessing", build_feature_pipeline()),
            ("classifier", DummyClassifier(strategy="prior", random_state=42)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture
def dummy_model_path(tmp_path: Path, dummy_pipeline: Pipeline) -> Path:
    """Artefacto de modelo dummy guardado en disco con el formato del Training Pipeline."""
    model_path = tmp_path / "dummy_model.joblib"
    joblib.dump(
        {
            "pipeline": dummy_pipeline,
            "model_type": "dummy_classifier",
            "feature_names": list(dummy_pipeline.feature_names_in_),
            "target": "diagnosis",
            "target_mapping": {1: 1, 2: 0},
        },
        model_path,
    )
    return model_path


def test_load_model_artifact_valid(dummy_model_path: Path) -> None:
    """Carga un artefacto completo y extrae el pipeline y sus metadatos."""
    artifact = load_model_artifact(dummy_model_path)

    assert "pipeline" in artifact
    assert artifact["model_type"] == "dummy_classifier"
    assert hasattr(artifact["pipeline"], "predict")


def test_load_model_artifact_raw_estimator(tmp_path: Path) -> None:
    """Envuelve un estimador serializado directamente en un artefacto válido."""
    model_path = tmp_path / "raw_estimator.joblib"
    joblib.dump(DummyClassifier(strategy="prior"), model_path)

    artifact = load_model_artifact(model_path)

    assert artifact["model_type"] == "DummyClassifier"
    assert hasattr(artifact["pipeline"], "predict")


def test_load_model_artifact_missing_file(tmp_path: Path) -> None:
    """Lanza FileNotFoundError si el modelo no existe."""
    with pytest.raises(FileNotFoundError, match="No se encontró el modelo"):
        load_model_artifact(tmp_path / "missing.joblib")


def test_load_model_artifact_invalid_payload(tmp_path: Path) -> None:
    """Lanza InferencePipelineError si el artefacto no contiene un pipeline."""
    model_path = tmp_path / "invalid.joblib"
    joblib.dump({"foo": "bar"}, model_path)

    with pytest.raises(InferencePipelineError, match="no contiene la clave 'pipeline'"):
        load_model_artifact(model_path)


def test_load_input_data_parquet_and_csv(
    tmp_path: Path, synthetic_raw_dataframe: pd.DataFrame
) -> None:
    """Lee correctamente archivos Parquet y CSV."""
    parquet_path = tmp_path / "patients.parquet"
    csv_path = tmp_path / "patients.csv"
    synthetic_raw_dataframe.to_parquet(parquet_path)
    synthetic_raw_dataframe.to_csv(csv_path, index=False)

    df_parquet = load_input_data(parquet_path)
    df_csv = load_input_data(csv_path)

    assert len(df_parquet) == N_SYNTHETIC_ROWS
    assert len(df_csv) == N_SYNTHETIC_ROWS
    assert "Age" in df_csv.columns


def test_load_input_data_missing_file(tmp_path: Path) -> None:
    """Lanza FileNotFoundError si el archivo de datos no existe."""
    with pytest.raises(FileNotFoundError, match="No se encontró el archivo de datos"):
        load_input_data(tmp_path / "missing.csv")


def test_load_input_data_unsupported_extension(tmp_path: Path) -> None:
    """Lanza InferencePipelineError ante una extensión no soportada."""
    data_path = tmp_path / "patients.txt"
    data_path.write_text("contenido", encoding="utf-8")

    with pytest.raises(InferencePipelineError, match="Formato de archivo no soportado"):
        load_input_data(data_path)


def test_load_input_data_csv_ignores_trailing_empty_columns(
    tmp_path: Path, synthetic_raw_dataframe: pd.DataFrame
) -> None:
    """Omite las columnas vacías de relleno típicas del CSV crudo."""
    csv_path = tmp_path / "raw_with_padding.csv"
    lines = synthetic_raw_dataframe.to_csv(index=False).rstrip("\n").splitlines()
    padded = [f"{lines[0]},,,,"] + [f"{line},,,," for line in lines[1:]]
    csv_path.write_text("\n".join(padded) + "\n", encoding="utf-8")

    df = load_input_data(csv_path)

    assert not any(str(column).startswith("Unnamed:") for column in df.columns)
    assert "Age" in df.columns
    assert len(df) == N_SYNTHETIC_ROWS


def test_prepare_inference_features_drops_metadata_and_target(
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """Omite columnas de metadatos y target preservando las variables clínicas."""
    features = prepare_inference_features(synthetic_raw_dataframe)

    assert "patient_id" not in features.columns
    assert "event_time" not in features.columns
    assert "Diagnosis" not in features.columns
    assert "Age" in features.columns
    assert len(features) == N_SYNTHETIC_ROWS


def test_prepare_inference_features_missing_required(
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """Lanza InferencePipelineError si falta una variable clínica requerida."""
    incomplete = synthetic_raw_dataframe.drop(columns=["Age"])

    with pytest.raises(InferencePipelineError, match="Faltan variables clínicas requeridas"):
        prepare_inference_features(incomplete)


def test_apply_training_transformations(
    dummy_pipeline: Pipeline, synthetic_raw_dataframe: pd.DataFrame
) -> None:
    """Aplica el preprocesamiento entrenado y devuelve la matriz transformada."""
    features = prepare_inference_features(synthetic_raw_dataframe)

    transformed = apply_training_transformations(dummy_pipeline, features)

    assert transformed is not None
    assert transformed.shape[0] == N_SYNTHETIC_ROWS


def test_apply_training_transformations_without_preprocessing(
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """Devuelve None cuando el estimador no tiene pasos de preprocesamiento."""
    bare_estimator = DummyClassifier(strategy="prior")
    features = prepare_inference_features(synthetic_raw_dataframe)

    assert apply_training_transformations(bare_estimator, features) is None


def test_generate_predictions_columns(
    dummy_pipeline: Pipeline, synthetic_raw_dataframe: pd.DataFrame
) -> None:
    """Genera etiquetas y probabilidades con el formato esperado."""
    features = prepare_inference_features(synthetic_raw_dataframe)

    predictions = generate_predictions(dummy_pipeline, features)

    assert len(predictions) == N_SYNTHETIC_ROWS
    assert EXPECTED_PREDICTION_COLUMNS.issubset(predictions.columns)
    assert set(predictions["prediction_label"]).issubset({"enfermo", "sano"})
    assert predictions["probability_disease"].between(0.0, 1.0).all()


def test_save_predictions_parquet_and_csv(
    tmp_path: Path, dummy_pipeline: Pipeline, synthetic_raw_dataframe: pd.DataFrame
) -> None:
    """Almacena predicciones en Parquet y CSV correctamente."""
    predictions = generate_predictions(
        dummy_pipeline, prepare_inference_features(synthetic_raw_dataframe)
    )

    parquet_path = save_predictions(predictions, tmp_path / "out" / "predictions.parquet")
    csv_path = save_predictions(predictions, tmp_path / "out" / "predictions.csv")

    assert Path(parquet_path).exists()
    assert Path(csv_path).exists()
    assert len(pd.read_parquet(parquet_path)) == N_SYNTHETIC_ROWS
    assert len(pd.read_csv(csv_path)) == N_SYNTHETIC_ROWS


def test_save_predictions_invalid_extension(
    tmp_path: Path, dummy_pipeline: Pipeline, synthetic_raw_dataframe: pd.DataFrame
) -> None:
    """Lanza InferencePipelineError ante una extensión de salida no soportada."""
    predictions = generate_predictions(
        dummy_pipeline, prepare_inference_features(synthetic_raw_dataframe)
    )

    with pytest.raises(InferencePipelineError, match="Formato de salida no soportado"):
        save_predictions(predictions, tmp_path / "predictions.json")


def test_plot_prediction_distribution(
    tmp_path: Path, dummy_pipeline: Pipeline, synthetic_raw_dataframe: pd.DataFrame
) -> None:
    """Genera la visualización de predicciones en disco."""
    predictions = generate_predictions(
        dummy_pipeline, prepare_inference_features(synthetic_raw_dataframe)
    )
    figure_path = tmp_path / "images" / "prediction_distribution.png"

    result = plot_prediction_distribution(predictions, figure_path)

    assert result is not None
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0


def test_save_inference_summary(tmp_path: Path) -> None:
    """Guarda el resumen de inferencia en formato JSON."""
    summary = {"status": "success", "n_samples": N_SYNTHETIC_ROWS}
    summary_path = save_inference_summary(summary, tmp_path / "summary.json")

    assert Path(summary_path).exists()
    with open(summary_path, encoding="utf-8") as file:
        assert json.load(file)["n_samples"] == N_SYNTHETIC_ROWS


def test_build_portable_summary_uses_relative_paths(tmp_path: Path) -> None:
    """Convierte rutas del proyecto a relativas y conserva las externas."""
    summary = build_portable_summary(
        {
            "model_path": str(PROJECT_ROOT / "models" / "liver_patient_model" / "model.joblib"),
            "input_data_path": str(tmp_path / "external.parquet"),
            "predictions_path": str(
                PROJECT_ROOT
                / "models"
                / "liver_patient_model"
                / "predictions"
                / "predictions.parquet"
            ),
            "figures": [str(PROJECT_ROOT / "models" / "liver_patient_model" / "figure.png")],
        }
    )

    assert summary["model_path"] == "models/liver_patient_model/model.joblib"
    assert summary["predictions_path"].startswith("models/liver_patient_model/predictions/")
    assert summary["figures"] == ["models/liver_patient_model/figure.png"]
    assert Path(summary["input_data_path"]).is_absolute()


def test_generate_inference_html_report_embeds_figure(
    tmp_path: Path, dummy_pipeline: Pipeline, synthetic_raw_dataframe: pd.DataFrame
) -> None:
    """El reporte HTML incrusta la figura como data URI y no referencia archivos externos."""
    predictions = generate_predictions(
        dummy_pipeline, prepare_inference_features(synthetic_raw_dataframe)
    )
    figure_path = plot_prediction_distribution(predictions, tmp_path / "images" / "dist.png")
    report_path = tmp_path / "inference_report.html"

    result = generate_inference_html_report(
        result={"status": "success", "model_type": "dummy", "n_samples": len(predictions)},
        predictions=predictions,
        figure_path=figure_path,
        report_path=report_path,
    )

    assert Path(result).exists()
    content = report_path.read_text(encoding="utf-8")
    assert "data:image/png;base64" in content
    assert 'src="images/' not in content
    assert "prediction_label" in content


def test_generate_inference_html_report_escapes_markup(
    tmp_path: Path, dummy_pipeline: Pipeline, synthetic_raw_dataframe: pd.DataFrame
) -> None:
    """El reporte escapa contenido HTML proveniente de los metadatos."""
    predictions = generate_predictions(
        dummy_pipeline, prepare_inference_features(synthetic_raw_dataframe)
    )
    report_path = tmp_path / "escaped_report.html"

    generate_inference_html_report(
        result={
            "status": "success",
            "model_type": "<script>alert(1)</script>",
            "n_samples": len(predictions),
        },
        predictions=predictions,
        figure_path=None,
        report_path=report_path,
    )

    content = report_path.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in content
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in content
    assert "Visualización no disponible" in content


def test_run_inference_pipeline_end_to_end(
    tmp_path: Path,
    dummy_model_path: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """Ejecuta el pipeline completo y genera predicciones, resumen y figura."""
    input_path = tmp_path / "new_patients.parquet"
    synthetic_raw_dataframe.to_parquet(input_path)
    output_dir = tmp_path / "inference_output"

    result = run_inference_pipeline(
        model_path=dummy_model_path,
        input_data_path=input_path,
        output_dir=output_dir,
    )

    assert result["status"] == "success"
    assert result["n_samples"] == N_SYNTHETIC_ROWS
    assert result["positive_predictions"] + result["negative_predictions"] == N_SYNTHETIC_ROWS
    assert result["transformed_shape"][0] == N_SYNTHETIC_ROWS
    assert len(result["figures"]) == 1

    assert Path(result["predictions_path"]).exists()
    assert Path(result["summary_path"]).exists()
    assert Path(result["report_path"]).exists()
    assert (output_dir / "images" / "prediction_distribution.png").exists()
    assert (output_dir / "inference_report.html").exists()

    content = (output_dir / "inference_report.html").read_text(encoding="utf-8")
    assert "data:image/png;base64" in content
    assert 'src="images/' not in content

    with open(result["summary_path"], encoding="utf-8") as file:
        summary = json.load(file)
    assert "report_path" in summary
    assert Path(summary["report_path"]).exists()


def test_run_inference_pipeline_without_plots(
    tmp_path: Path,
    dummy_model_path: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """Con generate_plots=False no se generan figuras."""
    input_path = tmp_path / "new_patients.parquet"
    synthetic_raw_dataframe.to_parquet(input_path)

    result = run_inference_pipeline(
        model_path=dummy_model_path,
        input_data_path=input_path,
        output_dir=tmp_path / "no_plots",
        generate_plots=False,
    )

    assert result["figures"] == []
    assert Path(result["report_path"]).exists()
    assert "Visualización no disponible" in Path(result["report_path"]).read_text(encoding="utf-8")


def test_main_cli_success(
    tmp_path: Path,
    dummy_model_path: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """La CLI ejecuta la inferencia y retorna código 0."""
    input_path = tmp_path / "cli_patients.parquet"
    synthetic_raw_dataframe.to_parquet(input_path)
    output_dir = tmp_path / "cli_output"

    exit_code = main(
        [
            "--model-path",
            str(dummy_model_path),
            "--input-data",
            str(input_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "predictions.parquet").exists()
    assert (output_dir / "inference_summary.json").exists()
    assert (output_dir / "inference_report.html").exists()


def test_main_cli_error(tmp_path: Path, dummy_model_path: Path) -> None:
    """La CLI retorna código 1 ante un archivo de entrada inexistente."""
    exit_code = main(
        [
            "--model-path",
            str(dummy_model_path),
            "--input-data",
            str(tmp_path / "missing.parquet"),
        ]
    )

    assert exit_code == 1


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="Modelo entrenado no disponible")
def test_real_trained_model_inference(synthetic_raw_dataframe: pd.DataFrame) -> None:
    """Verifica carga, transformación y predicción con el modelo real entrenado."""
    artifact = load_model_artifact(DEFAULT_MODEL_PATH)
    pipeline = artifact["pipeline"]
    features = prepare_inference_features(synthetic_raw_dataframe)

    transformed = apply_training_transformations(pipeline, features)
    predictions = generate_predictions(pipeline, features)

    assert transformed is not None
    assert transformed.shape[0] == len(features)
    assert len(predictions) == len(features)
    assert set(predictions["prediction"].unique()).issubset({0, 1})


def test_extract_metadata_and_preservation_in_predictions(
    dummy_pipeline: Pipeline, synthetic_raw_dataframe: pd.DataFrame
) -> None:
    """Verifica que los metadatos de identificación se extraigan y se asocien a las predicciones."""
    metadata = extract_metadata(synthetic_raw_dataframe)
    assert "patient_id" in metadata.columns
    assert "event_time" in metadata.columns

    features = prepare_inference_features(synthetic_raw_dataframe)
    predictions = generate_predictions(dummy_pipeline, features, metadata=metadata)

    assert "patient_id" in predictions.columns
    assert "event_time" in predictions.columns
    assert (
        predictions["patient_id"].to_numpy() == synthetic_raw_dataframe["patient_id"].to_numpy()
    ).all()


def test_generate_predictions_raises_on_predict_proba_failure(
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """Verifica que un fallo dentro de predict_proba lance InferencePipelineError."""

    class BrokenProbabilitiesEstimator(DummyClassifier):
        def predict_proba(self, X: Any) -> Any:
            raise RuntimeError("Fallo forzado en predict_proba")

    features = prepare_inference_features(synthetic_raw_dataframe)
    model = BrokenProbabilitiesEstimator(strategy="prior")
    model.fit(features, np.ones(len(features)))

    with pytest.raises(
        InferencePipelineError, match="Error al generar predicciones o probabilidades"
    ):
        generate_predictions(model, features)


def test_plot_prediction_distribution_color_consistency(
    tmp_path: Path, dummy_pipeline: Pipeline, synthetic_raw_dataframe: pd.DataFrame
) -> None:
    """Verifica que la generación de gráficos use asignación semántica de colores."""
    features = prepare_inference_features(synthetic_raw_dataframe)
    predictions = generate_predictions(dummy_pipeline, features)
    figure_path = tmp_path / "images" / "dist.png"

    result = plot_prediction_distribution(predictions, figure_path)
    assert result is not None
    assert figure_path.exists()


def test_run_inference_pipeline_output_path_derives_output_dir(
    tmp_path: Path,
    dummy_model_path: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """Verifica que pasar solo output_path guarde todos los artefactos en su directorio contenedor."""
    input_path = tmp_path / "patients.parquet"
    synthetic_raw_dataframe.to_parquet(input_path)
    custom_output = tmp_path / "custom_dir" / "my_preds.parquet"

    result = run_inference_pipeline(
        model_path=dummy_model_path,
        input_data_path=input_path,
        output_path=custom_output,
    )

    assert Path(result["predictions_path"]) == custom_output
    assert (custom_output.parent / "inference_summary.json").exists()
    assert (custom_output.parent / "inference_report.html").exists()
    assert (custom_output.parent / "images" / "prediction_distribution.png").exists()
