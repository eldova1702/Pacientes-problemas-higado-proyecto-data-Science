"""Pruebas unitarias para el servicio de inferencia de la demo (src/inference/app_service.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.inference.app_service import (
    APP_MODEL_PATH_ENV,
    AppInferenceError,
    ModelBundle,
    build_single_record_frame,
    build_template_frame,
    confusion_matrix_frame,
    derived_ratios,
    feature_input_specs,
    label_counts_frame,
    load_app_model,
    normalize_gender,
    normalize_uploaded_frame,
    order_prediction_columns,
    predict_single,
    predictions_to_csv_bytes,
    probability_histogram_frame,
    read_uploaded_table,
    required_columns_frame,
    resolve_app_model_path,
    run_batch_prediction,
    summarize_metrics,
    template_csv_bytes,
)
from src.inference.batch_validation import GENDER_COLUMN, NUMERIC_FEATURE_COLUMNS
from src.model.preprocessing import build_feature_pipeline
from src.pipelines.inference_pipeline.inference_pipeline import (
    DEFAULT_MODEL_PATH,
    REQUIRED_FEATURE_COLUMNS,
)

N_SYNTHETIC_ROWS = 12
EXPECTED_FEATURE_COUNT = 10
EXPECTED_HISTOGRAM_BINS = 10
CONFUSION_MATRIX_SIZE = 2
SAMPLE_PATIENT_ID = 1001

HEADER_TITLECASE = (
    "Age,Gender,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,"
    "Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,"
    "Albumin_and_Globulin_Ratio"
)
VALID_ROW = "65,Female,0.7,0.1,187,16,18,6.8,3.3,0.9"

BASE_VALUES: dict[str, Any] = {
    "age": 45.0,
    "gender": "Male",
    "total_bilirubin": 1.0,
    "direct_bilirubin": 0.3,
    "alkaline_phosphotase": 210.0,
    "alamine_aminotransferase": 36.0,
    "aspartate_aminotransferase": 42.0,
    "total_protiens": 6.5,
    "albumin": 3.1,
    "albumin_and_globulin_ratio": 0.9,
}


class EstimatorWithoutProbabilities:
    """Estimador mínimo que clasifica pero no expone `predict_proba`."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Devuelve la clase negativa para todas las filas recibidas."""
        return np.zeros(len(X), dtype="int64")


def csv_bytes(*rows: str, header: str = HEADER_TITLECASE, separator: str = ",") -> bytes:
    """Construye el contenido de un CSV sintético con el separador indicado."""
    lines = [header.replace(",", separator), *(row.replace(",", separator) for row in rows)]
    return ("\n".join(lines) + "\n").encode()


@pytest.fixture
def synthetic_features() -> pd.DataFrame:
    """Características clínicas sintéticas y deterministas."""
    generator = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "Age": generator.uniform(20, 75, N_SYNTHETIC_ROWS).round(1),
            "Gender": generator.choice(["Male", "Female"], N_SYNTHETIC_ROWS),
            "Total_Bilirubin": generator.uniform(0.5, 8.0, N_SYNTHETIC_ROWS).round(2),
            "Direct_Bilirubin": generator.uniform(0.1, 3.0, N_SYNTHETIC_ROWS).round(2),
            "Alkaline_Phosphotase": generator.uniform(120, 500, N_SYNTHETIC_ROWS).round(1),
            "Alamine_Aminotransferase": generator.uniform(15, 120, N_SYNTHETIC_ROWS).round(1),
            "Aspartate_Aminotransferase": generator.uniform(20, 150, N_SYNTHETIC_ROWS).round(1),
            "Total_Protiens": generator.uniform(5.0, 8.5, N_SYNTHETIC_ROWS).round(2),
            "Albumin": generator.uniform(2.0, 5.0, N_SYNTHETIC_ROWS).round(2),
            "Albumin_and_Globulin_Ratio": generator.uniform(0.4, 1.8, N_SYNTHETIC_ROWS).round(2),
        }
    )


@pytest.fixture
def dummy_bundle(synthetic_features: pd.DataFrame) -> ModelBundle:
    """Modelo ficticio entrenado sobre datos sintéticos, sin depender del artefacto real."""
    labels = np.array([1, 0] * (N_SYNTHETIC_ROWS // 2))
    pipeline = Pipeline(
        steps=[
            ("preprocessing", build_feature_pipeline()),
            ("classifier", DummyClassifier(strategy="stratified", random_state=42)),
        ]
    )
    pipeline.fit(synthetic_features, labels)
    return ModelBundle(
        pipeline=pipeline,
        model_type="dummy",
        metrics={"accuracy": 0.5, "confusion_matrix": [[1, 1], [1, 1]]},
        feature_names=list(REQUIRED_FEATURE_COLUMNS),
        created_at="2026-09-14T00:00:00+00:00",
        source_path="tests/dummy.joblib",
        supports_probabilities=True,
    )


@pytest.fixture
def real_bundle() -> ModelBundle:
    """Artefacto entrenado real del proyecto, omitido si todavía no se ha generado."""
    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip("Modelo entrenado no disponible")
    return load_app_model(DEFAULT_MODEL_PATH)


def test_load_app_model_returns_bundle_with_metrics(real_bundle: ModelBundle) -> None:
    """El artefacto real se carga con su pipeline, su tipo y sus métricas."""
    assert hasattr(real_bundle.pipeline, "predict")
    assert real_bundle.model_type
    assert real_bundle.supports_probabilities
    assert real_bundle.source_path.endswith("model.joblib")


def test_load_app_model_missing_file_raises_app_error(tmp_path: Path) -> None:
    """Un artefacto inexistente produce un mensaje accionable, no un FileNotFoundError."""
    with pytest.raises(AppInferenceError, match="No se encontró el modelo entrenado"):
        load_app_model(tmp_path / "inexistente.joblib")


def test_load_app_model_corrupted_file_raises_app_error(tmp_path: Path) -> None:
    """Un archivo dañado se reporta como incompatible en lugar de propagar la excepción."""
    corrupted = tmp_path / "modelo.joblib"
    corrupted.write_bytes(b"contenido que no es un joblib")
    with pytest.raises(AppInferenceError, match="dañado o es incompatible"):
        load_app_model(corrupted)


def test_load_app_model_artifact_without_pipeline_raises_app_error(tmp_path: Path) -> None:
    """Un diccionario sin la clave `pipeline` no es un modelo utilizable."""
    path = tmp_path / "modelo.joblib"
    joblib.dump({"metrics": {}}, path)
    with pytest.raises(AppInferenceError, match="pipeline"):
        load_app_model(path)


def test_load_app_model_detects_model_without_predict_proba(tmp_path: Path) -> None:
    """Un estimador sin probabilidades se carga igual, marcando la limitación.

    Se usa un estimador propio en lugar de quitarle el método a una clase de scikit-learn,
    porque mutar una clase compartida contaminaría el resto de la suite.
    """
    path = tmp_path / "modelo.joblib"
    joblib.dump({"pipeline": EstimatorWithoutProbabilities(), "model_type": "sin_proba"}, path)
    bundle = load_app_model(path)
    assert not bundle.supports_probabilities
    assert bundle.model_type == "sin_proba"


def test_resolve_app_model_path_honours_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """La variable de entorno permite apuntar la app a otro artefacto."""
    target = tmp_path / "otro.joblib"
    monkeypatch.setenv(APP_MODEL_PATH_ENV, str(target))
    assert resolve_app_model_path() == target


def test_resolve_app_model_path_prefers_explicit_argument(tmp_path: Path) -> None:
    """El argumento explícito tiene prioridad sobre cualquier valor por defecto."""
    target = tmp_path / "explicito.joblib"
    assert resolve_app_model_path(target) == target


def test_build_single_record_frame_uses_required_feature_columns() -> None:
    """El registro individual respeta el contrato exacto de columnas del pipeline."""
    frame = build_single_record_frame(BASE_VALUES)
    assert list(frame.columns) == list(REQUIRED_FEATURE_COLUMNS)
    assert len(frame) == 1
    for column in NUMERIC_FEATURE_COLUMNS:
        assert frame[column].dtype == "float64"


def test_build_single_record_frame_completes_missing_values() -> None:
    """Las variables ausentes se rellenan con los valores por defecto del formulario."""
    frame = build_single_record_frame({"age": 30.0})
    assert frame.loc[0, "age"] == pytest.approx(30.0)
    assert frame.loc[0, GENDER_COLUMN] == "Male"


def test_predict_single_returns_probabilities_in_unit_interval(
    real_bundle: ModelBundle,
) -> None:
    """La predicción individual devuelve probabilidades complementarias y coherentes."""
    result = predict_single(real_bundle, BASE_VALUES)
    assert result.prediction in (0, 1)
    assert result.probability_disease is not None
    assert result.probability_no_disease is not None
    assert 0.0 <= result.probability_disease <= 1.0
    assert result.probability_disease + result.probability_no_disease == pytest.approx(1.0)


def test_predict_single_matches_pipeline_prediction(real_bundle: ModelBundle) -> None:
    """El servicio no altera la decisión del modelo entrenado."""
    frame = build_single_record_frame(BASE_VALUES)
    expected = int(real_bundle.pipeline.predict(frame)[0])
    assert predict_single(real_bundle, BASE_VALUES).prediction == expected


def test_predict_single_warns_on_zero_total_bilirubin(real_bundle: ModelBundle) -> None:
    """Una bilirrubina total de cero deja indefinido un ratio y debe advertirse."""
    result = predict_single(real_bundle, {**BASE_VALUES, "total_bilirubin": 0.0})
    assert result.warnings
    assert any("indefinido" in message for message in result.warnings)
    assert result.derived_features["direct_to_total_bilirubin"] is None


def test_derived_ratios_reproduce_clinical_formulas() -> None:
    """Los cocientes mostrados coinciden con los que calcula el pipeline."""
    ratios = derived_ratios(build_single_record_frame(BASE_VALUES))
    expected_bilirubin = 0.3
    assert ratios["direct_to_total_bilirubin"] == pytest.approx(expected_bilirubin)
    assert ratios["ast_to_alt"] == pytest.approx(42.0 / 36.0)


def test_read_uploaded_table_parses_comma_separated_csv() -> None:
    """El caso más común, CSV separado por comas, se lee sin ajustes."""
    frame = read_uploaded_table(csv_bytes(VALID_ROW), "pacientes.csv")
    assert len(frame) == 1
    assert "Age" in frame.columns


def test_read_uploaded_table_parses_semicolon_separated_csv() -> None:
    """Un CSV exportado con punto y coma se detecta automáticamente."""
    frame = read_uploaded_table(csv_bytes(VALID_ROW, separator=";"), "pacientes.csv")
    assert len(frame) == 1
    assert "Gender" in frame.columns


def test_read_uploaded_table_parses_latin1_encoding() -> None:
    """Un archivo en latin-1 con acentos se decodifica sin fallar."""
    content = f"{HEADER_TITLECASE},Notas\n{VALID_ROW},Hepatitis crónica\n".encode("latin-1")
    frame = read_uploaded_table(content, "pacientes.csv")
    assert len(frame) == 1
    assert "Notas" in frame.columns


def test_read_uploaded_table_ignores_unnamed_filler_columns() -> None:
    """Las columnas de relleno del CSV original no llegan al pipeline."""
    content = f"{HEADER_TITLECASE},,,\n{VALID_ROW},,,\n".encode()
    frame = read_uploaded_table(content, "pacientes.csv")
    assert not any(str(column).startswith("Unnamed:") for column in frame.columns)


def test_read_uploaded_table_reads_parquet(
    tmp_path: Path, synthetic_features: pd.DataFrame
) -> None:
    """El formato Parquet se admite igual que el CSV."""
    path = tmp_path / "pacientes.parquet"
    synthetic_features.to_parquet(path, index=False)
    frame = read_uploaded_table(path.read_bytes(), path.name)
    assert len(frame) == N_SYNTHETIC_ROWS


def test_read_uploaded_table_raises_on_empty_content() -> None:
    """Un archivo de cero bytes se rechaza con un mensaje claro."""
    with pytest.raises(AppInferenceError, match="está vacío"):
        read_uploaded_table(b"", "pacientes.csv")


def test_read_uploaded_table_raises_on_unsupported_extension() -> None:
    """Sólo se admiten los formatos que el pipeline sabe leer."""
    with pytest.raises(AppInferenceError, match="Formato no soportado"):
        read_uploaded_table(csv_bytes(VALID_ROW), "pacientes.txt")


def test_read_uploaded_table_raises_on_corrupted_parquet() -> None:
    """Un Parquet ilegible produce un mensaje de usuario, no un traceback."""
    with pytest.raises(AppInferenceError, match="Parquet"):
        read_uploaded_table(b"\x00\x01binario", "pacientes.parquet")


def test_normalize_uploaded_frame_maps_titlecase_headers() -> None:
    """Los encabezados TitleCase se normalizan al contrato snake_case del pipeline."""
    raw = read_uploaded_table(csv_bytes(VALID_ROW), "pacientes.csv")
    features, _, ignored, _ = normalize_uploaded_frame(raw)
    assert list(features.columns) == list(REQUIRED_FEATURE_COLUMNS)
    assert ignored == ()


def test_normalize_uploaded_frame_separates_metadata_columns() -> None:
    """Los identificadores de trazabilidad se apartan de las características."""
    content = f"patient_id,{HEADER_TITLECASE}\n1001,{VALID_ROW}\n".encode()
    features, metadata, _, _ = normalize_uploaded_frame(read_uploaded_table(content, "p.csv"))
    assert "patient_id" in metadata.columns
    assert "patient_id" not in features.columns


def test_normalize_uploaded_frame_drops_target_columns() -> None:
    """La columna de diagnóstico real nunca se usa para predecir."""
    content = f"{HEADER_TITLECASE},Dataset\n{VALID_ROW},1\n".encode()
    features, _, ignored, _ = normalize_uploaded_frame(read_uploaded_table(content, "p.csv"))
    assert "dataset" not in features.columns
    assert "dataset" not in ignored


def test_normalize_uploaded_frame_reports_ignored_columns() -> None:
    """Las columnas que el modelo no usa se informan en lugar de descartarse en silencio."""
    content = f"{HEADER_TITLECASE},hospital\n{VALID_ROW},Central\n".encode()
    _, _, ignored, _ = normalize_uploaded_frame(read_uploaded_table(content, "p.csv"))
    assert ignored == ("hospital",)


def test_normalize_uploaded_frame_raises_on_duplicated_normalized_columns() -> None:
    """Dos columnas que normalizan al mismo nombre son ambiguas y deben rechazarse."""
    content = f"Age,age,{HEADER_TITLECASE.split(',', 1)[1]}\n65,65,{VALID_ROW.split(',', 1)[1]}\n"
    with pytest.raises(AppInferenceError, match="columnas duplicadas"):
        normalize_uploaded_frame(read_uploaded_table(content.encode(), "p.csv"))


def test_normalize_uploaded_frame_drops_fully_empty_rows() -> None:
    """Las filas vacías del final de un CSV no se cuentan como pacientes."""
    content = f"{HEADER_TITLECASE}\n{VALID_ROW}\n,,,,,,,,,\n{VALID_ROW}\n".encode()
    features, _, _, dropped = normalize_uploaded_frame(read_uploaded_table(content, "p.csv"))
    expected_rows = 2
    assert len(features) == expected_rows
    assert dropped == 1


def test_normalize_gender_maps_common_aliases() -> None:
    """Los alias en español y las abreviaturas se traducen a las categorías del modelo."""
    values = pd.Series(["masculino", "F", "hombre", "Female", "MUJER"])
    assert normalize_gender(values).tolist() == [
        "Male",
        "Female",
        "Male",
        "Female",
        "Female",
    ]


def test_normalize_gender_keeps_unknown_values() -> None:
    """Un valor irreconocible se conserva para que la validación pueda advertirlo."""
    assert normalize_gender(pd.Series(["Otro"])).tolist() == ["Otro"]


def test_run_batch_prediction_returns_pipeline_prediction_columns(
    real_bundle: ModelBundle,
) -> None:
    """La salida por lotes usa el mismo esquema que el Inference Pipeline del proyecto."""
    raw = read_uploaded_table(csv_bytes(VALID_ROW, VALID_ROW), "pacientes.csv")
    result = run_batch_prediction(real_bundle, raw)
    expected_rows = 2
    assert result.n_rows == expected_rows
    for column in ("prediction", "prediction_label", "probability_disease"):
        assert column in result.predictions.columns
    assert result.n_positive + result.n_negative == result.n_rows


def test_run_batch_prediction_preserves_patient_id(real_bundle: ModelBundle) -> None:
    """El identificador del paciente viaja hasta la salida para mantener la trazabilidad."""
    content = f"patient_id,{HEADER_TITLECASE}\n{SAMPLE_PATIENT_ID},{VALID_ROW}\n".encode()
    result = run_batch_prediction(real_bundle, read_uploaded_table(content, "p.csv"))
    assert result.predictions.loc[0, "patient_id"] == SAMPLE_PATIENT_ID


def test_run_batch_prediction_handles_single_row(real_bundle: ModelBundle) -> None:
    """Un archivo con un solo paciente se procesa igual que uno grande."""
    result = run_batch_prediction(real_bundle, read_uploaded_table(csv_bytes(VALID_ROW), "p.csv"))
    assert result.n_rows == 1
    assert result.positive_ratio in (0.0, 1.0)


def test_run_batch_prediction_returns_report_without_predicting_when_blocking(
    real_bundle: ModelBundle,
) -> None:
    """Con errores bloqueantes no se invoca al modelo y se devuelve el informe completo."""
    header = HEADER_TITLECASE.replace(",Albumin,", ",")
    content = f"{header}\n65,Female,0.7,0.1,187,16,18,6.8,0.9\n".encode()
    result = run_batch_prediction(real_bundle, read_uploaded_table(content, "p.csv"))
    assert result.report.is_blocking
    assert result.predictions.empty
    assert result.n_rows == 0


def test_run_batch_prediction_marks_rows_with_warnings(real_bundle: ModelBundle) -> None:
    """Las filas con advertencias quedan señaladas en el archivo descargable."""
    row = "65,Otro,0.7,0.1,187,16,18,6.8,3.3,0.9"
    result = run_batch_prediction(real_bundle, read_uploaded_table(csv_bytes(row), "p.csv"))
    assert not result.report.is_blocking
    assert result.predictions.loc[0, "advertencias"]


def test_run_batch_prediction_works_with_dummy_model(
    dummy_bundle: ModelBundle, synthetic_features: pd.DataFrame
) -> None:
    """El servicio no depende del artefacto real del proyecto."""
    content = synthetic_features.to_csv(index=False).encode()
    result = run_batch_prediction(dummy_bundle, read_uploaded_table(content, "p.csv"))
    assert result.n_rows == N_SYNTHETIC_ROWS


def test_order_prediction_columns_puts_predictions_first(real_bundle: ModelBundle) -> None:
    """La tabla de resultados muestra primero lo que el usuario vino a consultar."""
    result = run_batch_prediction(real_bundle, read_uploaded_table(csv_bytes(VALID_ROW), "p.csv"))
    ordered = order_prediction_columns(result.predictions)
    assert list(ordered.columns)[:2] == ["prediction", "prediction_label"]


def test_predictions_to_csv_bytes_preserves_accents() -> None:
    """El CSV descargable conserva los acentos al abrirse en Excel."""
    frame = pd.DataFrame({"diagnóstico": ["enfermedad hepática"]})
    content = predictions_to_csv_bytes(frame)
    assert content.startswith(b"\xef\xbb\xbf")
    assert "hepática" in content.decode("utf-8-sig")


def test_template_csv_bytes_contains_all_required_headers() -> None:
    """La plantilla descargable declara exactamente las columnas obligatorias."""
    header = template_csv_bytes().decode("utf-8-sig").splitlines()[0]
    assert header.split(",") == list(REQUIRED_FEATURE_COLUMNS)


def test_build_template_frame_rows_are_valid_input(real_bundle: ModelBundle) -> None:
    """Las filas de ejemplo de la plantilla son predecibles sin advertencias bloqueantes."""
    content = predictions_to_csv_bytes(build_template_frame())
    result = run_batch_prediction(real_bundle, read_uploaded_table(content, "plantilla.csv"))
    assert not result.report.is_blocking
    assert result.n_rows == len(build_template_frame())


def test_feature_input_specs_cover_every_numeric_variable() -> None:
    """El formulario expone todas las variables numéricas con sus límites clínicos."""
    specs = feature_input_specs()
    assert len(specs) == EXPECTED_FEATURE_COUNT - 1
    for spec in specs:
        assert spec.minimum <= spec.default <= spec.maximum
        assert spec.label
        assert "Rango clínico admitido" in spec.help_text


def test_summarize_metrics_handles_missing_metrics(dummy_bundle: ModelBundle) -> None:
    """Un artefacto con métricas parciales no rompe la ficha del modelo."""
    frame = summarize_metrics(dummy_bundle)
    assert list(frame.columns) == ["métrica", "valor"]
    assert len(frame) == 1


def test_confusion_matrix_frame_is_labelled_in_spanish(dummy_bundle: ModelBundle) -> None:
    """La matriz de confusión se muestra con etiquetas comprensibles."""
    frame = confusion_matrix_frame(dummy_bundle)
    assert frame.shape == (CONFUSION_MATRIX_SIZE, CONFUSION_MATRIX_SIZE)
    assert "Real: enfermedad" in frame.index


def test_confusion_matrix_frame_is_empty_without_metrics() -> None:
    """Sin matriz almacenada se devuelve una tabla vacía en lugar de fallar."""
    bundle = ModelBundle(
        pipeline=DummyClassifier(),
        model_type="x",
        metrics={},
        feature_names=[],
        created_at="",
        source_path="",
        supports_probabilities=False,
    )
    assert confusion_matrix_frame(bundle).empty


def test_label_counts_frame_and_histogram_handle_single_row(
    real_bundle: ModelBundle,
) -> None:
    """Las gráficas agregadas funcionan incluso con un único paciente."""
    result = run_batch_prediction(real_bundle, read_uploaded_table(csv_bytes(VALID_ROW), "p.csv"))
    counts = label_counts_frame(result.predictions)
    histogram = probability_histogram_frame(result.predictions)
    assert counts["pacientes"].sum() == 1
    assert len(histogram) == EXPECTED_HISTOGRAM_BINS
    assert histogram["pacientes"].sum() == 1


def test_label_counts_frame_is_empty_without_predictions() -> None:
    """Sin predicciones las tablas de gráficas quedan vacías, no fallan."""
    assert label_counts_frame(pd.DataFrame()).empty
    assert probability_histogram_frame(pd.DataFrame()).empty


def test_required_columns_frame_documents_every_variable() -> None:
    """La tabla de ayuda describe todas las columnas obligatorias."""
    frame = required_columns_frame()
    assert len(frame) == EXPECTED_FEATURE_COUNT
    assert list(frame.columns) == ["columna", "variable", "unidad", "valores admitidos"]
    assert set(frame["columna"]) == set(REQUIRED_FEATURE_COLUMNS)
