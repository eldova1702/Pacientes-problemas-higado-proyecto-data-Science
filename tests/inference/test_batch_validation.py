"""Pruebas unitarias para la validación por lotes de la demo (src/inference/batch_validation.py)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.data.validation import get_patient_features_schema
from src.inference.batch_validation import (
    GENDER_COLUMN,
    NUMERIC_FEATURE_COLUMNS,
    SEVERITY_ERROR,
    SEVERITY_WARNING,
    ValidationIssue,
    ValidationReport,
    build_batch_features_schema,
    clinical_ranges,
    coerce_feature_types,
    describe_rows,
    label_for,
    valid_gender_values,
    validate_batch_features,
)
from src.pipelines.inference_pipeline.inference_pipeline import REQUIRED_FEATURE_COLUMNS

EXPECTED_FEATURE_COUNT = 10
SMALL_ROW_LIMIT = 2
MANY_ROWS = 5
EXPECTED_AGE_RANGE = (1.0, 120.0)

BASE_RECORD: dict[str, Any] = {
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


def make_frame(*overrides: dict[str, Any]) -> pd.DataFrame:
    """Construye un DataFrame de características aplicando cambios fila por fila."""
    rows = [dict(BASE_RECORD) for _ in range(max(len(overrides), 1))]
    for row, override in zip(rows, overrides, strict=False):
        row.update(override)
    return pd.DataFrame(rows, columns=list(REQUIRED_FEATURE_COLUMNS))


@pytest.fixture
def valid_frame() -> pd.DataFrame:
    """Lote de tres pacientes sin ninguna anomalía."""
    return make_frame({}, {}, {})


def test_build_batch_features_schema_contains_only_feature_columns() -> None:
    """El esquema de la app sólo exige las variables clínicas, no metadatos ni target."""
    schema = build_batch_features_schema()
    assert sorted(schema.columns) == sorted(REQUIRED_FEATURE_COLUMNS)
    assert len(schema.columns) == EXPECTED_FEATURE_COUNT


def test_build_batch_features_schema_allows_extra_columns(valid_frame: pd.DataFrame) -> None:
    """Las columnas adicionales no invalidan el archivo."""
    extended = valid_frame.assign(patient_id=[1, 2, 3], notas=["a", "b", "c"])
    report = validate_batch_features(extended)
    assert not report.is_blocking


def test_clinical_ranges_match_source_schema() -> None:
    """Los rangos se leen del esquema Pandera del proyecto, sin duplicar valores."""
    ranges = clinical_ranges()
    assert set(ranges) == set(NUMERIC_FEATURE_COLUMNS)
    assert ranges["age"] == EXPECTED_AGE_RANGE
    source = get_patient_features_schema()
    for column, (minimum, maximum) in ranges.items():
        statistics = source.columns[column].checks[0].statistics
        assert (minimum, maximum) == (statistics["min_value"], statistics["max_value"])


def test_valid_gender_values_match_source_schema() -> None:
    """Las categorías de género provienen del esquema Pandera del proyecto."""
    assert valid_gender_values() == ("Male", "Female")


def test_validate_batch_features_accepts_valid_frame(valid_frame: pd.DataFrame) -> None:
    """Un lote correcto no produce errores ni advertencias."""
    report = validate_batch_features(valid_frame)
    assert not report.is_blocking
    assert report.warnings == ()


def test_validate_batch_features_blocks_missing_column(valid_frame: pd.DataFrame) -> None:
    """Falta una columna obligatoria: el pipeline no podría predecir, así que bloquea."""
    report = validate_batch_features(valid_frame.drop(columns=["albumin"]))
    assert report.is_blocking
    assert "Faltan columnas obligatorias" in report.errors[0].message
    assert "albumin" in report.errors[0].message


def test_validate_batch_features_blocks_empty_frame() -> None:
    """Un archivo sin filas de datos bloquea, porque `predict` fallaría."""
    report = validate_batch_features(make_frame().iloc[0:0])
    assert report.is_blocking
    assert "no contiene filas de datos" in report.errors[0].message


def test_validate_batch_features_blocks_unrecognized_columns() -> None:
    """Un archivo sin ninguna variable clínica recibe un mensaje sobre el formato."""
    report = validate_batch_features(pd.DataFrame({"cualquier_cosa": []}))
    assert report.is_blocking
    assert "No se reconoció ninguna variable clínica" in report.errors[0].message


def test_validate_batch_features_blocks_row_limit_exceeded() -> None:
    """Superar el límite de filas bloquea para acotar el consumo de memoria y CPU."""
    frame = make_frame(*({} for _ in range(MANY_ROWS)))
    report = validate_batch_features(frame, max_rows=SMALL_ROW_LIMIT)
    assert report.is_blocking
    assert str(SMALL_ROW_LIMIT) in report.errors[0].message


def test_validate_batch_features_warns_on_null_values() -> None:
    """Los nulos no bloquean: el pipeline los imputa y sólo se advierte."""
    report = validate_batch_features(make_frame({"albumin": None}, {}))
    assert not report.is_blocking
    messages = [issue.message for issue in report.warnings]
    assert any("vacío" in message for message in messages)


def test_validate_batch_features_warns_on_fully_empty_column() -> None:
    """Una columna completamente vacía recibe una advertencia específica y más severa."""
    report = validate_batch_features(make_frame({"albumin": None}, {"albumin": None}))
    messages = [issue.message for issue in report.warnings]
    assert any("completamente vacía" in message for message in messages)


def test_validate_batch_features_warns_on_out_of_range_values() -> None:
    """Los valores fuera del rango clínico advierten pero no bloquean: son outliers reales."""
    report = validate_batch_features(make_frame({"age": 200.0}))
    assert not report.is_blocking
    issue = next(issue for issue in report.warnings if issue.column == "age")
    assert issue.rows == (1,)
    assert "fuera del rango clínico" in issue.message


def test_validate_batch_features_warns_on_unknown_gender() -> None:
    """Un género desconocido se advierte porque el codificador lo ignora en silencio."""
    report = validate_batch_features(make_frame({GENDER_COLUMN: "Otro"}))
    assert not report.is_blocking
    issue = next(issue for issue in report.warnings if issue.column == GENDER_COLUMN)
    assert "Género no reconocido" in issue.message


def test_validate_batch_features_warns_on_zero_denominator_ratio() -> None:
    """Una bilirrubina total de cero deja indefinido un ratio clínico y debe advertirse."""
    report = validate_batch_features(make_frame({"total_bilirubin": 0.0}))
    issue = next(issue for issue in report.warnings if issue.column == "total_bilirubin")
    assert "indefinido" in issue.message


def test_validate_batch_features_omits_warnings_when_blocking() -> None:
    """Con errores bloqueantes no se emiten advertencias derivadas de la coerción fallida."""
    frame = make_frame({"total_bilirubin": "abc"})
    coerced, issues = coerce_feature_types(frame)
    report = validate_batch_features(coerced, extra_issues=issues)
    assert report.is_blocking
    assert report.warnings == ()


def test_validation_report_is_blocking_only_with_errors() -> None:
    """Un informe con sólo advertencias no bloquea el procesamiento."""
    warning = ValidationIssue(severity=SEVERITY_WARNING, column="age", message="aviso")
    assert not ValidationReport(warnings=(warning,)).is_blocking
    error = ValidationIssue(severity=SEVERITY_ERROR, column="age", message="error")
    assert ValidationReport(errors=(error,)).is_blocking


def test_validation_report_to_frame_lists_severity_column_and_rows() -> None:
    """La tabla del informe expone severidad, columna, mensaje y filas afectadas."""
    issue = ValidationIssue(severity=SEVERITY_WARNING, column="age", message="aviso", rows=(2, 5))
    frame = ValidationReport(warnings=(issue,)).to_frame()
    assert list(frame.columns) == ["severidad", "columna", "mensaje", "filas"]
    assert frame.loc[0, "columna"] == label_for("age")
    assert frame.loc[0, "filas"] == "filas 2, 5"


def test_validation_report_warned_rows_collects_every_affected_row() -> None:
    """Las filas advertidas agregan los hallazgos de todas las columnas."""
    report = validate_batch_features(make_frame({"age": 200.0}, {GENDER_COLUMN: "Otro"}))
    assert report.warned_rows == frozenset({1, 2})


def test_coerce_feature_types_blocks_non_numeric_value() -> None:
    """Un texto en una columna clínica bloquea, porque el imputador lanzaría ValueError."""
    _, issues = coerce_feature_types(make_frame({"albumin": "abc"}))
    blocking = [issue for issue in issues if issue.severity == SEVERITY_ERROR]
    assert blocking
    assert "no son numéricos" in blocking[0].message
    assert blocking[0].rows == (1,)


def test_coerce_feature_types_can_impute_non_numeric_on_demand() -> None:
    """Con la opción activada, el texto se degrada a advertencia y queda como nulo."""
    coerced, issues = coerce_feature_types(make_frame({"albumin": "abc"}), impute_non_numeric=True)
    assert all(issue.severity == SEVERITY_WARNING for issue in issues)
    assert pd.isna(coerced.loc[0, "albumin"])


def test_coerce_feature_types_converts_decimal_comma() -> None:
    """La coma decimal se reinterpreta como punto y se informa la conversión."""
    coerced, issues = coerce_feature_types(make_frame({"albumin": "3,4"}))
    expected = 3.4
    assert coerced.loc[0, "albumin"] == pytest.approx(expected)
    assert any("coma como separador decimal" in issue.message for issue in issues)


def test_coerce_feature_types_keeps_valid_numeric_values(valid_frame: pd.DataFrame) -> None:
    """Un lote ya numérico se conserva intacto y en tipo float64."""
    coerced, issues = coerce_feature_types(valid_frame)
    assert issues == []
    for column in NUMERIC_FEATURE_COLUMNS:
        assert coerced[column].dtype == "float64"
    assert coerced["albumin"].tolist() == valid_frame["albumin"].tolist()


def test_describe_rows_truncates_long_lists() -> None:
    """La enumeración de filas se trunca para no desbordar la interfaz."""
    assert describe_rows([3]) == "fila 3"
    assert describe_rows([9, 3]) == "filas 3, 9"
    assert describe_rows([1, 2, 3], limit=2) == "filas 1, 2 y 1 más"
    assert describe_rows([]) == ""
