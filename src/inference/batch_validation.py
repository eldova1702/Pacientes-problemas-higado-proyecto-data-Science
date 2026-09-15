"""Validación de archivos de pacientes para el procesamiento por lotes de la app Streamlit.

Este módulo traduce las reglas clínicas ya definidas en `src/data/validation.py` a un
contrato apto para inferencia, donde el objetivo no es rechazar datos imperfectos sino
distinguir dos situaciones muy distintas:

1. **Errores bloqueantes**: el pipeline entrenado no puede producir una predicción
   (columna obligatoria ausente, valor no numérico, archivo sin filas). Cada uno de ellos
   hace que `Pipeline.predict` lance `ValueError`.
2. **Advertencias**: la predicción se genera, pero su confiabilidad baja (nulos que se
   imputan, valores fuera del rango clínico, género no reconocido, ratios indefinidos).

La distinción importa porque el preprocesamiento persistido imputa y escala sin fallar, y
`AGENTS.md` pide preservar los outliers clínicos: bloquear por rango rechazaría pacientes
reales con valores extremos legítimos.

Los rangos y las categorías válidas no se redefinen aquí: se extraen del esquema Pandera de
`src/data/validation.py`, que sigue siendo la única fuente de verdad.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pandera.pandas as pa

from src.data.validation import (
    DataValidationError,
    get_patient_features_schema,
    validate_patient_features,
)
from src.pipelines.inference_pipeline.inference_pipeline import REQUIRED_FEATURE_COLUMNS

MAX_BATCH_ROWS: int = 5000
MAX_ISSUE_ROWS_SHOWN: int = 20
MAX_SAMPLE_VALUES_SHOWN: int = 3

SEVERITY_ERROR: str = "error"
SEVERITY_WARNING: str = "advertencia"

GENDER_COLUMN: str = "gender"
NUMERIC_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    column for column in REQUIRED_FEATURE_COLUMNS if column != GENDER_COLUMN
)

# Columnas que actúan como denominador en los ratios clínicos derivados. Un cero aquí deja
# el ratio indefinido y el imputador lo sustituye por la mediana de entrenamiento, lo que
# puede desplazar la probabilidad estimada de forma notable.
ZERO_DENOMINATOR_COLUMNS: tuple[str, ...] = ("total_bilirubin", "alamine_aminotransferase")

# Etiquetas legibles para los mensajes dirigidos al usuario final.
COLUMN_LABELS: dict[str, str] = {
    "age": "Edad",
    "gender": "Género",
    "total_bilirubin": "Bilirrubina total",
    "direct_bilirubin": "Bilirrubina directa",
    "alkaline_phosphotase": "Fosfatasa alcalina",
    "alamine_aminotransferase": "ALT (alanina aminotransferasa)",
    "aspartate_aminotransferase": "AST (aspartato aminotransferasa)",
    "total_protiens": "Proteínas totales",
    "albumin": "Albúmina",
    "albumin_and_globulin_ratio": "Razón albúmina/globulina",
}


@dataclass(frozen=True)
class ValidationIssue:
    """Hallazgo individual de la validación de un archivo por lotes.

    Attributes:
        severity: `SEVERITY_ERROR` si impide predecir, `SEVERITY_WARNING` si sólo avisa.
        column: Nombre snake_case de la columna afectada, o cadena vacía si es de tabla.
        message: Mensaje en español listo para mostrarse en la interfaz.
        rows: Números de fila de datos (1-based, sin contar la cabecera) afectados.
    """

    severity: str
    column: str
    message: str
    rows: tuple[int, ...] = ()


@dataclass(frozen=True)
class ValidationReport:
    """Resultado completo de validar un archivo por lotes."""

    errors: tuple[ValidationIssue, ...] = ()
    warnings: tuple[ValidationIssue, ...] = ()

    @property
    def is_blocking(self) -> bool:
        """Indica si existe al menos un error que impide generar predicciones."""
        return len(self.errors) > 0

    @property
    def warned_rows(self) -> frozenset[int]:
        """Conjunto de filas de datos que acumulan al menos una advertencia."""
        return frozenset(row for issue in self.warnings for row in issue.rows)

    def to_frame(self) -> pd.DataFrame:
        """Construye una tabla con todos los hallazgos, apta para mostrar en la interfaz.

        Returns:
            DataFrame con las columnas `severidad`, `columna`, `mensaje` y `filas`.
        """
        records = [
            {
                "severidad": issue.severity,
                "columna": label_for(issue.column) if issue.column else "(archivo)",
                "mensaje": issue.message,
                "filas": describe_rows(issue.rows) if issue.rows else "-",
            }
            for issue in (*self.errors, *self.warnings)
        ]
        return pd.DataFrame(records, columns=["severidad", "columna", "mensaje", "filas"])


def describe_rows(rows: tuple[int, ...] | list[int], limit: int = MAX_ISSUE_ROWS_SHOWN) -> str:
    """Formatea una lista de filas para un mensaje al usuario, truncando si es muy larga.

    Args:
        rows: Números de fila 1-based.
        limit: Cantidad máxima de filas que se enumeran antes de resumir.

    Returns:
        Texto como `fila 3`, `filas 3, 9` o `filas 1, 2 y 40 más`.
    """
    ordered = sorted({int(row) for row in rows})
    if not ordered:
        return ""
    noun = "fila" if len(ordered) == 1 else "filas"
    shown = ", ".join(str(row) for row in ordered[:limit])
    if len(ordered) > limit:
        return f"{noun} {shown} y {len(ordered) - limit} más"
    return f"{noun} {shown}"


def label_for(column: str) -> str:
    """Devuelve la etiqueta legible de una columna, o su propio nombre si no está mapeada."""
    return COLUMN_LABELS.get(column, column)


def clinical_ranges() -> dict[str, tuple[float, float]]:
    """Extrae los rangos clínicos admitidos desde el esquema Pandera del proyecto.

    No duplica los valores: los lee de `get_patient_features_schema()`, de modo que si las
    reglas clínicas cambian, el formulario y la validación por lotes cambian con ellas.

    Returns:
        Diccionario `columna -> (mínimo, máximo)` para las variables numéricas.
    """
    schema = get_patient_features_schema()
    ranges: dict[str, tuple[float, float]] = {}
    for column in NUMERIC_FEATURE_COLUMNS:
        for check in schema.columns[column].checks:
            statistics: dict[str, Any] = check.statistics or {}
            if "min_value" in statistics and "max_value" in statistics:
                ranges[column] = (
                    float(statistics["min_value"]),
                    float(statistics["max_value"]),
                )
    return ranges


def valid_gender_values() -> tuple[str, ...]:
    """Extrae las categorías válidas de género desde el esquema Pandera del proyecto."""
    schema = get_patient_features_schema()
    for check in schema.columns[GENDER_COLUMN].checks:
        statistics: dict[str, Any] = check.statistics or {}
        allowed = statistics.get("allowed_values")
        if allowed:
            return tuple(str(value) for value in allowed)
    return ("Male", "Female")


def build_batch_features_schema() -> pa.DataFrameSchema:
    """Construye el esquema bloqueante de sólo-características para la app.

    A diferencia de `PATIENT_FEATURES_SCHEMA`, este esquema no exige `patient_id`,
    `event_time`, `diagnosis` ni las variables derivadas, permite columnas adicionales y no
    aplica los chequeos de rango (que aquí se tratan como advertencias, no como errores).
    Sólo verifica lo que realmente impide predecir: que las columnas clínicas existan.

    Returns:
        DataFrameSchema con las columnas de `REQUIRED_FEATURE_COLUMNS`.
    """
    source = get_patient_features_schema()
    columns = {
        name: pa.Column(
            source.columns[name].dtype,
            checks=[],
            nullable=True,
            coerce=True,
            required=True,
            description=source.columns[name].description,
        )
        for name in REQUIRED_FEATURE_COLUMNS
    }
    return pa.DataFrameSchema(
        columns=columns,
        checks=[
            pa.Check(
                lambda df: len(df) > 0,
                name="dataframe_not_empty",
                error="El archivo no contiene filas de datos",
            )
        ],
        strict=False,
        ordered=False,
    )


BATCH_FEATURES_SCHEMA: pa.DataFrameSchema = build_batch_features_schema()


def _row_numbers(index: pd.Index) -> tuple[int, ...]:
    """Convierte un índice posicional 0-based en números de fila 1-based para el usuario."""
    return tuple(int(value) + 1 for value in index)


def _repair_decimal_comma(original: pd.Series, numeric: pd.Series) -> pd.Series | None:
    """Reintenta la conversión numérica tratando la coma como separador decimal.

    Sólo se intenta sobre columnas no numéricas. La comprobación se hace por tipo lógico y
    no comparando con `object`, porque pandas 3.0 asigna el dtype `str` a las columnas
    formadas únicamente por texto.

    Args:
        original: Serie tal como se leyó del archivo.
        numeric: Resultado de la conversión directa a numérico.

    Returns:
        La serie reparada si recupera más valores que la conversión directa, o None.
    """
    if pd.api.types.is_numeric_dtype(original):
        return None
    repaired = pd.to_numeric(
        original.astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    )
    if repaired.notna().sum() > numeric.notna().sum():
        return repaired
    return None


def coerce_feature_types(
    features: pd.DataFrame, *, impute_non_numeric: bool = False
) -> tuple[pd.DataFrame, list[ValidationIssue]]:
    """Convierte las columnas clínicas a numérico, tolerando la coma decimal.

    Los valores que no pueden convertirse se marcan como error bloqueante, porque el
    `SimpleImputer(strategy="median")` del pipeline entrenado lanza `ValueError` ante datos
    no numéricos. Con `impute_non_numeric=True` se degradan a advertencia y quedan como
    nulos para que el imputador los rellene.

    Args:
        features: DataFrame con las columnas ya normalizadas a snake_case.
        impute_non_numeric: Si es True, los valores no numéricos no bloquean.

    Returns:
        Tupla con el DataFrame convertido y la lista de hallazgos detectados.
    """
    converted = features.copy()
    issues: list[ValidationIssue] = []
    severity = SEVERITY_WARNING if impute_non_numeric else SEVERITY_ERROR

    for column in NUMERIC_FEATURE_COLUMNS:
        if column not in converted.columns:
            continue
        original = converted[column]
        numeric = pd.to_numeric(original, errors="coerce")
        if numeric.isna().any():
            repaired = _repair_decimal_comma(original, numeric)
            if repaired is not None:
                issues.append(
                    ValidationIssue(
                        severity=SEVERITY_WARNING,
                        column=column,
                        message=(
                            f"Se interpretó la coma como separador decimal en {label_for(column)}."
                        ),
                    )
                )
                numeric = repaired
        failed = original.notna() & numeric.isna()
        if failed.any():
            issues.append(_non_numeric_issue(column, original, converted.index[failed], severity))
        converted[column] = numeric.astype("float64")

    return converted, issues


def _non_numeric_issue(
    column: str, original: pd.Series, failed_index: pd.Index, severity: str
) -> ValidationIssue:
    """Construye el hallazgo que describe valores no convertibles a número."""
    rows = _row_numbers(failed_index)
    sample = original.loc[failed_index].astype(str).head(MAX_SAMPLE_VALUES_SHOWN).tolist()
    return ValidationIssue(
        severity=severity,
        column=column,
        message=(
            f"{label_for(column)} contiene valores que no son numéricos "
            f"(por ejemplo: {', '.join(sample)}) en {describe_rows(rows)}."
        ),
        rows=rows,
    )


def _missing_columns(error: DataValidationError) -> list[str]:
    """Extrae los nombres de columna ausentes desde las fallas reportadas por Pandera."""
    failures = error.failure_cases
    if failures is None or failures.empty or "check" not in failures.columns:
        return []
    missing = failures.loc[failures["check"] == "column_in_dataframe", "failure_case"]
    return sorted(str(name) for name in missing.dropna().unique())


def _check_structure(features: pd.DataFrame, max_rows: int) -> list[ValidationIssue]:
    """Verifica tamaño y presencia de las columnas obligatorias (todo bloqueante)."""
    if features.empty:
        recognized = [column for column in REQUIRED_FEATURE_COLUMNS if column in features.columns]
        message = (
            "El archivo no contiene filas de datos."
            if recognized
            else (
                "No se reconoció ninguna variable clínica en el archivo. Verifique que la "
                "primera línea contenga los nombres de las columnas y descargue la "
                "plantilla para ver el formato esperado."
            )
        )
        return [ValidationIssue(severity=SEVERITY_ERROR, column="", message=message)]

    issues: list[ValidationIssue] = []
    if len(features) > max_rows:
        issues.append(
            ValidationIssue(
                severity=SEVERITY_ERROR,
                column="",
                message=(
                    f"El archivo contiene {len(features)} filas y el máximo admitido es "
                    f"{max_rows}. Divida el archivo en partes más pequeñas."
                ),
            )
        )

    try:
        validate_patient_features(features, schema=BATCH_FEATURES_SCHEMA)
    except DataValidationError as error:
        missing = _missing_columns(error)
        if missing:
            listed = ", ".join(f"{label_for(name)} ({name})" for name in missing)
            issues.append(
                ValidationIssue(
                    severity=SEVERITY_ERROR,
                    column="",
                    message=(
                        f"Faltan columnas obligatorias: {listed}. Descargue la plantilla "
                        f"para ver el formato esperado."
                    ),
                )
            )
    return issues


def _check_nulls(features: pd.DataFrame) -> list[ValidationIssue]:
    """Advierte sobre valores vacíos, que el pipeline imputa con la mediana o la moda."""
    issues: list[ValidationIssue] = []
    for column in REQUIRED_FEATURE_COLUMNS:
        if column not in features.columns:
            continue
        null_mask = features[column].isna()
        count = int(null_mask.sum())
        if count == 0:
            continue
        if count == len(features):
            message = (
                f"La columna {label_for(column)} está completamente vacía. Todas las filas "
                f"usarán el valor imputado del entrenamiento y la predicción será poco "
                f"confiable."
            )
        else:
            message = (
                f"{label_for(column)} tiene {count} valor(es) vacío(s); se imputarán con el "
                f"valor de referencia aprendido durante el entrenamiento."
            )
        issues.append(
            ValidationIssue(
                severity=SEVERITY_WARNING,
                column=column,
                message=message,
                rows=_row_numbers(features.index[null_mask]),
            )
        )
    return issues


def _check_ranges(features: pd.DataFrame) -> list[ValidationIssue]:
    """Advierte sobre valores fuera del rango clínico, sin bloquear la predicción."""
    issues: list[ValidationIssue] = []
    for column, (minimum, maximum) in clinical_ranges().items():
        if column not in features.columns:
            continue
        values = features[column]
        out_of_range = values.notna() & ((values < minimum) | (values > maximum))
        if not out_of_range.any():
            continue
        rows = _row_numbers(features.index[out_of_range])
        issues.append(
            ValidationIssue(
                severity=SEVERITY_WARNING,
                column=column,
                message=(
                    f"{label_for(column)} tiene valores fuera del rango clínico "
                    f"[{minimum:g}, {maximum:g}] en {describe_rows(rows)}."
                ),
                rows=rows,
            )
        )
    return issues


def _check_gender(features: pd.DataFrame) -> list[ValidationIssue]:
    """Advierte sobre géneros no reconocidos, que el modelo codifica en silencio.

    `OneHotEncoder(handle_unknown="ignore")` convierte una categoría desconocida en un vector
    de ceros sin lanzar error, de modo que la predicción se degrada sin ninguna señal. Este
    chequeo es la señal que falta.
    """
    if GENDER_COLUMN not in features.columns:
        return []
    allowed = valid_gender_values()
    values = features[GENDER_COLUMN]
    unknown = values.notna() & ~values.isin(allowed)
    if not unknown.any():
        return []
    rows = _row_numbers(features.index[unknown])
    sample = sorted({str(value) for value in values[unknown].head(5)})
    return [
        ValidationIssue(
            severity=SEVERITY_WARNING,
            column=GENDER_COLUMN,
            message=(
                f"Género no reconocido en {describe_rows(rows)} (por ejemplo: "
                f"{', '.join(sample)}). El modelo lo trata como categoría desconocida y la "
                f"predicción de esas filas es menos confiable. Valores admitidos: "
                f"{', '.join(allowed)}."
            ),
            rows=rows,
        )
    ]


def _check_zero_denominators(features: pd.DataFrame) -> list[ValidationIssue]:
    """Advierte sobre ceros que dejan indefinidos los ratios clínicos derivados."""
    issues: list[ValidationIssue] = []
    for column in ZERO_DENOMINATOR_COLUMNS:
        if column not in features.columns:
            continue
        zeros = features[column] == 0
        if not zeros.any():
            continue
        rows = _row_numbers(features.index[zeros])
        issues.append(
            ValidationIssue(
                severity=SEVERITY_WARNING,
                column=column,
                message=(
                    f"{label_for(column)} vale 0 en {describe_rows(rows)}. El ratio clínico "
                    f"que la usa como denominador queda indefinido y se imputa con la "
                    f"mediana del entrenamiento, lo que puede desviar la probabilidad "
                    f"estimada de forma significativa."
                ),
                rows=rows,
            )
        )
    return issues


def validate_batch_features(
    features: pd.DataFrame,
    *,
    max_rows: int = MAX_BATCH_ROWS,
    extra_issues: list[ValidationIssue] | None = None,
) -> ValidationReport:
    """Valida un lote de características y separa errores bloqueantes de advertencias.

    Args:
        features: DataFrame con columnas snake_case y tipos ya coaccionados.
        max_rows: Número máximo de filas admitidas.
        extra_issues: Hallazgos previos (por ejemplo, de `coerce_feature_types`) a integrar.

    Returns:
        ValidationReport con los errores y las advertencias encontradas.
    """
    collected: list[ValidationIssue] = list(extra_issues or [])
    collected.extend(_check_structure(features, max_rows))

    blocking = [issue for issue in collected if issue.severity == SEVERITY_ERROR]
    advisory = [issue for issue in collected if issue.severity == SEVERITY_WARNING]

    # Con errores bloqueantes no se ejecutan los chequeos de advertencia: los valores que no
    # pudieron convertirse ya quedaron como nulos, de modo que informar sobre ellos como
    # «columna vacía» contradiría el error real que el usuario debe corregir primero.
    if blocking:
        return ValidationReport(errors=tuple(blocking), warnings=())

    if not features.empty:
        advisory.extend(_check_nulls(features))
        advisory.extend(_check_ranges(features))
        advisory.extend(_check_gender(features))
        advisory.extend(_check_zero_denominators(features))

    return ValidationReport(errors=tuple(blocking), warnings=tuple(advisory))
