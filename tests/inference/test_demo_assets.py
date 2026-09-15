"""Pruebas unitarias para el generador de evidencia de la demo (src/inference/demo_assets.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.inference.app_service import load_app_model
from src.inference.demo_assets import (
    DEMO_DIR,
    DEMO_REPORT_FILENAME,
    DEMO_SUMMARY_FILENAME,
    EXAMPLE_INPUT_FILENAME,
    EXAMPLE_OUTPUT_FILENAME,
    FAULTY_INPUT_FILENAME,
    FAULTY_ROW_DESCRIPTIONS,
    RAW_DATA_PATH,
    TEMPLATE_FILENAME,
    TITLECASE_COLUMNS,
    _relative,
    build_demo_html_report,
    build_example_input_frame,
    build_faulty_example_frame,
    collect_robustness_evidence,
    generate_demo_assets,
    main,
)
from src.pipelines.inference_pipeline.inference_pipeline import DEFAULT_MODEL_PATH

SAMPLE_ROWS = 8
CONTROLLED_ERROR_LABEL = "Error controlado"

requires_raw_data = pytest.mark.skipif(
    not RAW_DATA_PATH.exists(), reason="Dataset crudo no disponible"
)
requires_model = pytest.mark.skipif(
    not DEFAULT_MODEL_PATH.exists(), reason="Modelo entrenado no disponible"
)


@requires_raw_data
def test_build_example_input_frame_is_deterministic() -> None:
    """La muestra de ejemplo es reproducible gracias a la semilla fija del proyecto."""
    first = build_example_input_frame(SAMPLE_ROWS)
    second = build_example_input_frame(SAMPLE_ROWS)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == SAMPLE_ROWS


@requires_raw_data
def test_build_example_input_frame_excludes_target_column() -> None:
    """El archivo de ejemplo no filtra el diagnóstico real hacia la demo."""
    frame = build_example_input_frame(SAMPLE_ROWS)
    assert "Dataset" not in frame.columns
    assert "Diagnosis" not in frame.columns
    assert list(frame.columns) == ["patient_id", *TITLECASE_COLUMNS]


@requires_raw_data
def test_build_example_input_frame_has_no_missing_values() -> None:
    """El ejemplo de referencia usa registros completos; los defectos van en otro archivo."""
    frame = build_example_input_frame(SAMPLE_ROWS)
    assert int(frame.isna().sum().sum()) == 0


def test_build_faulty_example_frame_contains_expected_defects() -> None:
    """El archivo de defectos cubre un caso por cada situación documentada."""
    frame = build_faulty_example_frame()
    assert len(frame) == len(FAULTY_ROW_DESCRIPTIONS)
    assert frame.loc[0, "Gender"] == "Masculino"
    assert frame.loc[1, "Gender"] == "Otro"
    assert frame.loc[3, "Albumin"] is None or pd.isna(frame.loc[3, "Albumin"])
    assert frame.loc[4, "Total_Bilirubin"] == 0.0
    assert frame.loc[5, "Albumin_and_Globulin_Ratio"] == "0,85"


@requires_model
def test_collect_robustness_evidence_controls_every_failure() -> None:
    """Ningún caso límite escapa como excepción: todos producen un mensaje para el usuario."""
    evidence = collect_robustness_evidence(load_app_model(DEFAULT_MODEL_PATH))
    assert evidence
    assert all(case["mensaje"] for case in evidence)
    assert any(case["resultado"] == CONTROLLED_ERROR_LABEL for case in evidence)
    assert any(case["resultado"].startswith("Predicciones generadas") for case in evidence)


def test_build_demo_html_report_embeds_images_as_data_uri(tmp_path: Path) -> None:
    """El reporte es autocontenido: las imágenes viajan incrustadas, no referenciadas."""
    image = tmp_path / "grafica.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n contenido de prueba")
    report_path = tmp_path / DEMO_REPORT_FILENAME
    build_demo_html_report(
        summary={"status": "success"},
        evidence=[{"caso": "x", "resultado": "y", "mensaje": "z"}],
        image_paths=[image],
        report_path=report_path,
    )
    content = report_path.read_text(encoding="utf-8")
    assert "data:image/png;base64" in content
    assert 'src="images/' not in content


def test_build_demo_html_report_escapes_markup(tmp_path: Path) -> None:
    """El contenido dinámico se escapa para no inyectar marcado en el reporte."""
    report_path = tmp_path / DEMO_REPORT_FILENAME
    build_demo_html_report(
        summary={"model_type": "<script>alert(1)</script>"},
        evidence=[],
        image_paths=[],
        report_path=report_path,
    )
    content = report_path.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in content
    assert "&lt;script&gt;" in content


@requires_model
@requires_raw_data
def test_generate_demo_assets_creates_expected_files(tmp_path: Path) -> None:
    """La generación produce plantilla, ejemplos, salida, reporte y resumen."""
    summary = generate_demo_assets(output_dir=tmp_path, n_rows=SAMPLE_ROWS)
    for filename in (
        TEMPLATE_FILENAME,
        EXAMPLE_INPUT_FILENAME,
        FAULTY_INPUT_FILENAME,
        EXAMPLE_OUTPUT_FILENAME,
        DEMO_REPORT_FILENAME,
        DEMO_SUMMARY_FILENAME,
    ):
        assert (tmp_path / filename).exists(), filename
    assert summary["status"] == "success"
    assert summary["n_example_rows"] == SAMPLE_ROWS


def test_relative_converts_paths_inside_the_project() -> None:
    """Las rutas del proyecto se persisten en forma relativa y con separadores POSIX."""
    assert _relative(DEFAULT_MODEL_PATH) == "models/liver_patient_model/model.joblib"


def test_relative_keeps_paths_outside_the_project(tmp_path: Path) -> None:
    """Una ruta externa al proyecto no puede relativizarse y se conserva tal cual."""
    external = tmp_path / "salida.csv"
    assert Path(_relative(external)).is_absolute()


@requires_model
@requires_raw_data
def test_demo_summary_uses_project_relative_paths() -> None:
    """La evidencia versionada no contiene rutas absolutas dependientes de la máquina.

    Se comprueba sobre el directorio real de la demo, que es el único caso en el que la
    convención de portabilidad del proyecto tiene sentido: un destino fuera del repositorio
    no admite una ruta relativa.
    """
    summary_path = DEMO_DIR / DEMO_SUMMARY_FILENAME
    if not summary_path.exists():
        pytest.skip("Evidencia de la demo todavía no generada")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in ("model_path", "template_path", "example_input_path", "report_path"):
        assert not Path(payload[key]).is_absolute(), key


@requires_model
@requires_raw_data
def test_generated_output_keeps_prediction_columns(tmp_path: Path) -> None:
    """La salida de ejemplo trae las columnas de predicción y conserva el identificador."""
    generate_demo_assets(output_dir=tmp_path, n_rows=SAMPLE_ROWS)
    predictions = pd.read_csv(tmp_path / EXAMPLE_OUTPUT_FILENAME)
    assert len(predictions) == SAMPLE_ROWS
    for column in ("patient_id", "prediction", "prediction_label", "probability_disease"):
        assert column in predictions.columns


@requires_model
@requires_raw_data
def test_main_returns_zero_on_success(tmp_path: Path) -> None:
    """La interfaz de línea de comandos termina con código de salida 0."""
    exit_code = main(["--output-dir", str(tmp_path), "--n-rows", str(SAMPLE_ROWS)])
    assert exit_code == 0


def test_main_returns_one_when_generation_fails(tmp_path: Path) -> None:
    """Un fallo durante la generación se reporta con código de salida 1, sin traceback."""
    blocker = tmp_path / "archivo"
    blocker.write_text("no es un directorio", encoding="utf-8")
    assert main(["--output-dir", str(blocker)]) == 1
