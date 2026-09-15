"""Pruebas end-to-end de la interfaz Streamlit (app.py) con `streamlit.testing.v1.AppTest`.

Estas pruebas ejecutan la aplicación real en modo headless, rellenan widgets, pulsan botones
y suben archivos, de modo que verifican el requisito central del despliegue: la interfaz no
debe fallar ante ninguna entrada. En cada caso se comprueba que no se propaga ninguna
excepción y que el usuario recibe un mensaje en español.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.inference.app_service import APP_MODEL_PATH_ENV
from src.pipelines.inference_pipeline.inference_pipeline import DEFAULT_MODEL_PATH

APP_PATH = Path(__file__).resolve().parents[2] / "app.py"
APP_TIMEOUT_SECONDS = 120
EXPECTED_TAB_COUNT = 3
EXPECTED_NUMERIC_INPUTS = 9
UPLOAD_MIME = "text/csv"

HEADER = (
    "Age,Gender,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,"
    "Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,"
    "Albumin_and_Globulin_Ratio"
)
VALID_ROW = "65,Female,0.7,0.1,187,16,18,6.8,3.3,0.9"

pytestmark = pytest.mark.skipif(
    not DEFAULT_MODEL_PATH.exists(), reason="Modelo entrenado no disponible"
)


def run_app() -> AppTest:
    """Arranca la aplicación real en modo headless."""
    return AppTest.from_file(str(APP_PATH), default_timeout=APP_TIMEOUT_SECONDS).run()


def upload_csv(app: AppTest, content: bytes, filename: str = "pacientes.csv") -> AppTest:
    """Sube un archivo a la pestaña de procesamiento por lotes y vuelve a ejecutar la app."""
    app.file_uploader[0].upload(filename, content, UPLOAD_MIME)
    return app.run()


def csv_content(*rows: str, header: str = HEADER, separator: str = ",") -> bytes:
    """Construye el contenido de un CSV sintético."""
    lines = [header.replace(",", separator), *(row.replace(",", separator) for row in rows)]
    return ("\n".join(lines) + "\n").encode()


def messages(app: AppTest) -> str:
    """Concatena todos los mensajes de error y advertencia mostrados en la interfaz."""
    return " ".join([element.value for element in app.error] + [e.value for e in app.warning])


def test_app_starts_without_exceptions() -> None:
    """La aplicación arranca limpia, sin ninguna excepción sin controlar."""
    app = run_app()
    assert not app.exception
    assert app.title[0].value


def test_app_renders_three_tabs() -> None:
    """Se ofrecen las tres pestañas: individual, por lotes e información del modelo."""
    app = run_app()
    assert len(app.tabs) == EXPECTED_TAB_COUNT


def test_app_shows_no_error_alerts_on_first_run() -> None:
    """Al abrir la aplicación con el modelo disponible no se muestra ningún error."""
    app = run_app()
    assert [element.value for element in app.error] == []


def test_single_form_exposes_every_clinical_variable() -> None:
    """El formulario individual expone las nueve variables numéricas y el género."""
    app = run_app()
    assert len(app.number_input) == EXPECTED_NUMERIC_INPUTS
    assert any(element.label == "Género" for element in app.selectbox)


def test_single_prediction_produces_result_with_probability() -> None:
    """Enviar el formulario produce un veredicto y su probabilidad, sin excepciones."""
    app = run_app()
    app.button[0].click().run()
    assert not app.exception
    labels = [element.label for element in app.metric]
    assert "Probabilidad de enfermedad" in labels
    assert app.success or app.error


def test_single_prediction_accepts_boundary_values() -> None:
    """Los valores en el límite del rango clínico se procesan sin fallar."""
    app = run_app()
    for element in app.number_input:
        element.set_value(element.max)
    app.button[0].click().run()
    assert not app.exception


def test_single_prediction_warns_on_zero_total_bilirubin() -> None:
    """Una bilirrubina total de cero advierte sobre el ratio clínico indefinido."""
    app = run_app()
    app.number_input(key="paciente_total_bilirubin").set_value(0.0)
    app.button[0].click().run()
    assert not app.exception
    assert "indefinido" in messages(app)


def test_preset_selector_fills_the_form() -> None:
    """Elegir un caso de ejemplo rellena el formulario y permite predecir."""
    app = run_app()
    app.selectbox(key="caso_ejemplo").select("Perfil con alteración hepática").run()
    assert not app.exception
    expected_age = 62.0
    assert app.number_input(key="paciente_age").value == expected_age


def test_batch_tab_offers_template_download() -> None:
    """La pestaña por lotes ofrece descargar la plantilla antes de subir nada."""
    app = run_app()
    labels = [element.label for element in app.download_button]
    assert "Descargar plantilla vacía" in labels


def test_batch_upload_valid_csv_produces_download_button() -> None:
    """Un archivo válido genera predicciones descargables."""
    app = upload_csv(run_app(), csv_content(VALID_ROW, VALID_ROW))
    assert not app.exception
    assert "Descargar predicciones (CSV)" in [e.label for e in app.download_button]


def test_batch_upload_semicolon_csv_is_parsed() -> None:
    """Un CSV exportado con punto y coma se interpreta correctamente."""
    app = upload_csv(run_app(), csv_content(VALID_ROW, separator=";"))
    assert not app.exception
    assert "Descargar predicciones (CSV)" in [e.label for e in app.download_button]


def test_batch_upload_snake_case_headers_is_parsed() -> None:
    """Los encabezados en snake_case se aceptan igual que los TitleCase."""
    app = upload_csv(run_app(), csv_content(VALID_ROW, header=HEADER.lower()))
    assert not app.exception
    assert "Descargar predicciones (CSV)" in [e.label for e in app.download_button]


def test_batch_upload_missing_column_shows_error_without_exception() -> None:
    """Falta una columna obligatoria: se informa el problema y no se predice."""
    header = HEADER.replace(",Albumin,", ",")
    content = csv_content("65,Female,0.7,0.1,187,16,18,6.8,0.9", header=header)
    app = upload_csv(run_app(), content)
    assert not app.exception
    assert "Faltan columnas obligatorias" in messages(app)
    assert "Descargar predicciones (CSV)" not in [e.label for e in app.download_button]


def test_batch_upload_non_numeric_value_shows_error_without_exception() -> None:
    """Un texto en una columna clínica se reporta con su fila, sin traceback."""
    app = upload_csv(run_app(), csv_content("65,Female,abc,0.1,187,16,18,6.8,3.3,0.9"))
    assert not app.exception
    assert "no son numéricos" in messages(app)


def test_batch_upload_unknown_gender_warns_and_predicts() -> None:
    """Un género desconocido advierte de forma destacada pero no impide predecir."""
    app = upload_csv(run_app(), csv_content("65,Otro,0.7,0.1,187,16,18,6.8,3.3,0.9"))
    assert not app.exception
    assert "Género no reconocido" in messages(app)
    assert "Descargar predicciones (CSV)" in [e.label for e in app.download_button]


def test_batch_upload_out_of_range_value_warns_and_predicts() -> None:
    """Un valor fuera del rango clínico advierte pero conserva la predicción."""
    app = upload_csv(run_app(), csv_content("200,Female,0.7,0.1,187,16,18,6.8,3.3,0.9"))
    assert not app.exception
    assert "fuera del rango clínico" in messages(app)
    assert "Descargar predicciones (CSV)" in [e.label for e in app.download_button]


def test_batch_upload_header_only_file_shows_error_without_exception() -> None:
    """Un archivo con encabezados pero sin datos se rechaza con un mensaje claro."""
    app = upload_csv(run_app(), csv_content())
    assert not app.exception
    assert "no contiene filas de datos" in messages(app)


def test_batch_upload_empty_file_shows_error_without_exception() -> None:
    """Un archivo de cero bytes se rechaza sin romper la interfaz."""
    app = upload_csv(run_app(), b"")
    assert not app.exception
    assert "está vacío" in messages(app)


def test_batch_upload_unreadable_content_shows_error_without_exception() -> None:
    """Un archivo que no es una tabla recibe una explicación sobre el formato esperado."""
    app = upload_csv(run_app(), b"esto no es un csv en absoluto")
    assert not app.exception
    assert "No se reconoció ninguna variable clínica" in messages(app)


def test_batch_upload_corrupted_parquet_shows_error_without_exception() -> None:
    """Un Parquet dañado produce un mensaje de usuario y no un traceback."""
    app = run_app()
    app.file_uploader[0].upload("pacientes.parquet", b"\x00\x01binario", "application/octet-stream")
    app.run()
    assert not app.exception
    assert "Parquet" in messages(app)


def test_batch_upload_single_row_file_works() -> None:
    """Un archivo con un solo paciente se procesa igual que uno grande."""
    app = upload_csv(run_app(), csv_content(VALID_ROW))
    assert not app.exception
    assert "Descargar predicciones (CSV)" in [e.label for e in app.download_button]


def test_batch_upload_duplicated_normalized_columns_shows_error() -> None:
    """Dos columnas ambiguas tras normalizar se rechazan con un mensaje explicativo."""
    header = f"Age,age,{HEADER.split(',', 1)[1]}"
    content = csv_content(f"65,65,{VALID_ROW.split(',', 1)[1]}", header=header)
    app = upload_csv(run_app(), content)
    assert not app.exception
    assert "columnas duplicadas" in messages(app)


def test_model_tab_shows_metrics_and_artifact_path() -> None:
    """La pestaña informativa muestra el algoritmo y la ruta del artefacto cargado."""
    app = run_app()
    assert not app.exception
    assert any(element.label == "Algoritmo" for element in app.metric)


def test_app_shows_friendly_error_when_model_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sin el artefacto del modelo la interfaz explica el problema en lugar de caerse."""
    monkeypatch.setenv(APP_MODEL_PATH_ENV, str(tmp_path / "inexistente.joblib"))
    app = run_app()
    assert not app.exception
    assert "No se encontró el modelo entrenado" in messages(app)
    assert len(app.tabs) == EXPECTED_TAB_COUNT
