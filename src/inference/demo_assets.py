"""Generador reproducible de los archivos de ejemplo y la evidencia de la demo Streamlit.

Produce, bajo `models/liver_patient_model/demo/`, todo lo que el despliegue debe entregar:

* la plantilla de carga por lotes y un archivo de entrada de ejemplo derivado del dataset
  original;
* un archivo de entrada con defectos deliberados, que sirve a la vez de evidencia de la
  robustez de la interfaz y de material de prueba;
* la salida de predicciones correspondiente;
* un reporte HTML autocontenido y un resumen JSON con rutas relativas al proyecto, siguiendo
  la convención de evidencia que ya usan el Training Pipeline y el Inference Pipeline.

Uso:
    uv run python src/inference/demo_assets.py
    uv run python -m src.inference.demo_assets
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import app_service as service  # noqa: E402
from src.pipelines.inference_pipeline.inference_pipeline import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    build_portable_summary,
    run_inference_pipeline,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("demo_assets")

DEMO_DIR = PROJECT_ROOT / "models" / "liver_patient_model" / "demo"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "01_raw" / "Pacientes_porblemas_higado_india.csv"

TEMPLATE_FILENAME = "plantilla_lote.csv"
EXAMPLE_INPUT_FILENAME = "ejemplo_entrada_lote.csv"
FAULTY_INPUT_FILENAME = "ejemplo_entrada_con_errores.csv"
EXAMPLE_OUTPUT_FILENAME = "ejemplo_salida_predicciones.csv"
DEMO_REPORT_FILENAME = "demo_report.html"
DEMO_SUMMARY_FILENAME = "demo_summary.json"
SCREENSHOTS_DIRNAME = "capturas"

DEFAULT_EXAMPLE_ROWS = 24
DEMO_RANDOM_SEED = 42
FIRST_PATIENT_ID = 1001
TARGET_COLUMN_RAW = "Dataset"
POSITIVE_RAW_LABEL = 1

TITLECASE_COLUMNS: tuple[str, ...] = (
    "Age",
    "Gender",
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Protiens",
    "Albumin",
    "Albumin_and_Globulin_Ratio",
)


class DemoAssetsError(Exception):
    """Excepción lanzada cuando la evidencia de la demo no puede generarse."""


def _relative(path: Path) -> str:
    """Devuelve la ruta relativa al proyecto, para no persistir rutas de la máquina local."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _read_raw_dataset() -> pd.DataFrame:
    """Lee el CSV crudo descartando las columnas de relleno vacías.

    Returns:
        DataFrame con las columnas clínicas originales y el target `Dataset`.

    Raises:
        DemoAssetsError: Si el archivo de datos crudos no está disponible.
    """
    if not RAW_DATA_PATH.exists():
        msg = f"No se encontró el dataset crudo en: {RAW_DATA_PATH}"
        raise DemoAssetsError(msg)
    header = pd.read_csv(RAW_DATA_PATH, nrows=0)
    valid = [
        column
        for column in header.columns
        if str(column).strip() and not str(column).startswith("Unnamed:")
    ]
    return pd.read_csv(RAW_DATA_PATH, usecols=valid)


def build_example_input_frame(n_rows: int = DEFAULT_EXAMPLE_ROWS) -> pd.DataFrame:
    """Construye el archivo de entrada de ejemplo a partir del dataset original.

    La muestra es estratificada y determinista: mitad de pacientes con diagnóstico positivo
    y mitad negativo, con la semilla fija del proyecto. Conserva deliberadamente los nombres
    de columna en formato TitleCase para demostrar que la normalización funciona, y añade
    `patient_id` para evidenciar la trazabilidad en la salida.

    Se descartan los registros incompletos del archivo crudo (hay 8 filas sin género y
    varias con variables clínicas ausentes): este archivo debe servir como ejemplo limpio de
    referencia, mientras que los casos problemáticos se concentran en
    `build_faulty_example_frame`.

    Args:
        n_rows: Número total de pacientes de la muestra.

    Returns:
        DataFrame listo para guardarse como CSV de entrada.
    """
    raw = (
        _read_raw_dataset()
        .dropna(subset=[*TITLECASE_COLUMNS, TARGET_COLUMN_RAW])
        .drop_duplicates(subset=list(TITLECASE_COLUMNS))
    )
    generator = np.random.default_rng(DEMO_RANDOM_SEED)
    half = n_rows // 2

    positives = raw[raw[TARGET_COLUMN_RAW] == POSITIVE_RAW_LABEL]
    negatives = raw[raw[TARGET_COLUMN_RAW] != POSITIVE_RAW_LABEL]
    chosen = pd.concat(
        [
            positives.iloc[generator.choice(len(positives), half, replace=False)],
            negatives.iloc[generator.choice(len(negatives), n_rows - half, replace=False)],
        ]
    ).sort_index()

    sample = chosen[list(TITLECASE_COLUMNS)].reset_index(drop=True)
    sample.insert(0, "patient_id", range(FIRST_PATIENT_ID, FIRST_PATIENT_ID + len(sample)))
    return sample


def build_faulty_example_frame() -> pd.DataFrame:
    """Construye un archivo con defectos deliberados para evidenciar el manejo de errores.

    Cada fila reproduce uno de los problemas que la aplicación debe detectar sin caerse.

    Returns:
        DataFrame con los defectos descritos en `FAULTY_ROW_DESCRIPTIONS`.
    """
    base = {
        "patient_id": 0,
        "Age": 45,
        "Gender": "Male",
        "Total_Bilirubin": 1.0,
        "Direct_Bilirubin": 0.3,
        "Alkaline_Phosphotase": 210,
        "Alamine_Aminotransferase": 36,
        "Aspartate_Aminotransferase": 42,
        "Total_Protiens": 6.5,
        "Albumin": 3.1,
        "Albumin_and_Globulin_Ratio": 0.9,
    }
    rows: list[dict[str, Any]] = []
    for index in range(len(FAULTY_ROW_DESCRIPTIONS)):
        row = dict(base)
        row["patient_id"] = FIRST_PATIENT_ID + index
        rows.append(row)

    rows[0]["Gender"] = "Masculino"
    rows[1]["Gender"] = "Otro"
    rows[2]["Age"] = 200
    rows[3]["Albumin"] = None
    rows[4]["Total_Bilirubin"] = 0.0
    rows[5]["Albumin_and_Globulin_Ratio"] = "0,85"

    return pd.DataFrame(rows, columns=["patient_id", *TITLECASE_COLUMNS])


#: Descripción de cada fila defectuosa, en el mismo orden que `build_faulty_example_frame`.
FAULTY_ROW_DESCRIPTIONS: tuple[str, ...] = (
    "Género escrito en español (Masculino): se normaliza automáticamente a Male.",
    "Género desconocido (Otro): se predice igual pero con advertencia destacada.",
    "Edad fuera del rango clínico (200): advertencia, no bloquea.",
    "Albúmina vacía: se imputa con la mediana del entrenamiento y se advierte.",
    "Bilirrubina total igual a 0: el ratio derivado queda indefinido y se advierte.",
    "Razón A/G con coma decimal (0,85): se reinterpreta como 0.85 y se advierte.",
)


def collect_robustness_evidence(bundle: service.ModelBundle) -> list[dict[str, str]]:
    """Ejecuta casos límite contra el servicio y registra el mensaje que vería el usuario.

    Esta es la evidencia de que la interfaz no falla: cada entrada problemática produce un
    mensaje en español en lugar de una excepción.

    Args:
        bundle: Modelo cargado.

    Returns:
        Lista de registros con el caso probado, el resultado y el mensaje mostrado.
    """
    header = ",".join(TITLECASE_COLUMNS)
    valid_row = "65,Female,0.7,0.1,187,16,18,6.8,3.3,0.9"
    cases: list[tuple[str, bytes, str]] = [
        ("Archivo válido", f"{header}\n{valid_row}\n".encode(), "pacientes.csv"),
        (
            "Separador de punto y coma",
            f"{header.replace(',', ';')}\n{valid_row.replace(',', ';')}\n".encode(),
            "pacientes.csv",
        ),
        (
            "Columna obligatoria ausente",
            f"{header.replace(',Albumin,', ',')}\n65,Female,0.7,0.1,187,16,18,6.8,0.9\n".encode(),
            "pacientes.csv",
        ),
        (
            "Valor no numérico",
            f"{header}\n65,Female,abc,0.1,187,16,18,6.8,3.3,0.9\n".encode(),
            "pacientes.csv",
        ),
        (
            "Género no reconocido",
            f"{header}\n65,Otro,0.7,0.1,187,16,18,6.8,3.3,0.9\n".encode(),
            "pacientes.csv",
        ),
        ("Archivo sin filas de datos", f"{header}\n".encode(), "pacientes.csv"),
        ("Archivo vacío", b"", "pacientes.csv"),
        ("Extensión no soportada", f"{header}\n{valid_row}\n".encode(), "pacientes.txt"),
        ("Contenido ilegible", b"esto no es un csv", "pacientes.csv"),
    ]

    evidence: list[dict[str, str]] = []
    for name, content, filename in cases:
        evidence.append(_evaluate_case(bundle, name, content, filename))
    return evidence


def _evaluate_case(
    bundle: service.ModelBundle, name: str, content: bytes, filename: str
) -> dict[str, str]:
    """Ejecuta un caso límite y describe el resultado sin dejar escapar la excepción."""
    try:
        raw = service.read_uploaded_table(content, filename)
        result = service.run_batch_prediction(bundle, raw)
    except service.AppInferenceError as error:
        return {"caso": name, "resultado": "Error controlado", "mensaje": str(error)}

    if result.report.is_blocking:
        messages = "; ".join(issue.message for issue in result.report.errors)
        return {"caso": name, "resultado": "Error controlado", "mensaje": messages}

    warnings_text = "; ".join(issue.message for issue in result.report.warnings)
    return {
        "caso": name,
        "resultado": f"Predicciones generadas ({result.n_rows} fila/s)",
        "mensaje": warnings_text or "Sin advertencias.",
    }


def _encode_image(path: Path) -> str:
    """Codifica una imagen PNG como data URI base64 para incrustarla en el reporte."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _render_table(frame: pd.DataFrame) -> str:
    """Convierte un DataFrame en una tabla HTML sin índice."""
    return str(frame.to_html(index=False, border=0, escape=True))


def _render_images(image_paths: list[Path]) -> str:
    """Construye el bloque HTML con todas las imágenes incrustadas en base64."""
    if not image_paths:
        return "<p style='color:#6b7280;'>Sin imágenes disponibles.</p>"
    blocks = [
        f"<figure><figcaption>{html.escape(path.stem)}</figcaption>"
        f'<img src="{_encode_image(path)}" alt="{html.escape(path.stem)}"/></figure>'
        for path in image_paths
    ]
    return "".join(blocks)


def build_demo_html_report(
    summary: dict[str, Any],
    evidence: list[dict[str, str]],
    image_paths: list[Path],
    report_path: Path,
) -> str:
    """Genera el reporte HTML autocontenido de la demo.

    Las imágenes se incrustan como data URI base64, de modo que el archivo puede abrirse o
    compartirse por separado sin visualizaciones rotas, igual que el resto de la evidencia
    del proyecto.

    Args:
        summary: Resumen de la generación de la demo.
        evidence: Casos de robustez y el mensaje mostrado en cada uno.
        image_paths: Imágenes PNG a incrustar.
        report_path: Ruta destino del archivo HTML.

    Returns:
        Ruta del archivo generado.
    """
    summary_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in summary.items()
        if not isinstance(value, (list, dict))
    )
    faulty_rows = "".join(
        f"<tr><td>{index + 1}</td><td>{html.escape(description)}</td></tr>"
        for index, description in enumerate(FAULTY_ROW_DESCRIPTIONS)
    )
    evidence_table = _render_table(pd.DataFrame(evidence))

    content = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Evidencia de la demo Streamlit</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         margin: 30px; color: #111827; background: #f9fafb; }}
  .card {{ max-width: 980px; margin: 0 auto; background: #fff; border: 1px solid #e5e7eb;
           border-radius: 8px; padding: 24px; }}
  h1 {{ font-size: 22px; margin-top: 0; }}
  h2 {{ font-size: 16px; color: #374151; margin-top: 28px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  th, td {{ border-bottom: 1px solid #e5e7eb; padding: 6px 10px; text-align: left;
            vertical-align: top; }}
  figure {{ margin: 16px 0; }}
  figcaption {{ color: #6b7280; font-size: 13px; margin-bottom: 6px; }}
  img {{ max-width: 100%; border: 1px solid #e5e7eb; border-radius: 6px; }}
</style>
</head>
<body>
<div class="card">
  <h1>Evidencia de funcionamiento de la demo Streamlit</h1>
  <p style="color:#4b5563;">Modelo de detección de enfermedad hepática: predicción
  individual y procesamiento por lotes.</p>

  <h2>Resumen de la generación</h2>
  <table>{summary_rows}</table>

  <h2>Manejo de entradas problemáticas</h2>
  <p style="color:#4b5563;">Cada caso se ejecutó contra el servicio real. En ninguno se
  produce una excepción sin controlar: el usuario siempre recibe un mensaje en español.</p>
  {evidence_table}

  <h2>Contenido del archivo de ejemplo con errores</h2>
  <table><tr><th>Fila</th><th>Defecto demostrado</th></tr>{faulty_rows}</table>

  <h2>Visualizaciones</h2>
  {_render_images(image_paths)}
</div>
</body>
</html>
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")
    logger.info(f"Reporte de la demo guardado en: {report_path}")
    return str(report_path)


def _collect_demo_images(output_dir: Path) -> list[Path]:
    """Reúne las imágenes disponibles: la distribución generada y las capturas manuales."""
    images: list[Path] = []
    distribution = output_dir / "images" / "prediction_distribution.png"
    if distribution.exists():
        images.append(distribution)
    screenshots = output_dir / "images" / SCREENSHOTS_DIRNAME
    if screenshots.is_dir():
        images.extend(sorted(screenshots.glob("*.png")))
    return images


def generate_demo_assets(
    output_dir: Path = DEMO_DIR, n_rows: int = DEFAULT_EXAMPLE_ROWS
) -> dict[str, Any]:
    """Genera todos los archivos de ejemplo y la evidencia de la demo.

    Args:
        output_dir: Directorio destino de los artefactos.
        n_rows: Número de pacientes del archivo de entrada de ejemplo.

    Returns:
        Diccionario con el resumen de la generación y las rutas relativas al proyecto.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generando archivos de la demo en: {output_dir}")

    template_path = output_dir / TEMPLATE_FILENAME
    template_path.write_bytes(service.template_csv_bytes())

    example_frame = build_example_input_frame(n_rows)
    example_path = output_dir / EXAMPLE_INPUT_FILENAME
    example_frame.to_csv(example_path, index=False)

    faulty_path = output_dir / FAULTY_INPUT_FILENAME
    build_faulty_example_frame().to_csv(faulty_path, index=False)

    inference = run_inference_pipeline(
        model_path=DEFAULT_MODEL_PATH,
        input_data_path=example_path,
        output_path=output_dir / EXAMPLE_OUTPUT_FILENAME,
        output_dir=output_dir,
        generate_plots=True,
    )

    bundle = service.load_app_model(DEFAULT_MODEL_PATH)
    evidence = collect_robustness_evidence(bundle)

    summary: dict[str, Any] = {
        "status": "success",
        "model_path": str(DEFAULT_MODEL_PATH),
        "model_type": bundle.model_type,
        "template_path": _relative(template_path),
        "example_input_path": _relative(example_path),
        "faulty_input_path": _relative(faulty_path),
        "predictions_path": inference["predictions_path"],
        "figures": inference["figures"],
        "n_example_rows": len(example_frame),
        "n_positive_predictions": inference["positive_predictions"],
        "n_negative_predictions": inference["negative_predictions"],
        "n_robustness_cases": len(evidence),
        "n_controlled_errors": sum(
            1 for case in evidence if case["resultado"] == "Error controlado"
        ),
    }
    portable = build_portable_summary(summary)

    report_path = build_demo_html_report(
        summary=portable,
        evidence=evidence,
        image_paths=_collect_demo_images(output_dir),
        report_path=output_dir / DEMO_REPORT_FILENAME,
    )
    portable["report_path"] = _relative(Path(report_path))

    summary_path = output_dir / DEMO_SUMMARY_FILENAME
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(portable, file, indent=4, ensure_ascii=False, default=str)
    logger.info(f"Resumen de la demo guardado en: {summary_path}")

    portable["summary_path"] = _relative(summary_path)
    return portable


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Genera los archivos de ejemplo y la evidencia de la demo Streamlit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEMO_DIR,
        help="Directorio destino de los artefactos de la demo.",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=DEFAULT_EXAMPLE_ROWS,
        help=f"Pacientes del archivo de ejemplo (por defecto: {DEFAULT_EXAMPLE_ROWS}).",
    )
    return parser.parse_args(args)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada principal para CLI."""
    args = parse_args(argv)
    try:
        summary = generate_demo_assets(output_dir=args.output_dir, n_rows=args.n_rows)
        logger.info(f"Evidencia de la demo generada: {summary['report_path']}")
    except Exception:
        logger.exception("Error al generar la evidencia de la demo.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
