"""Módulo de validación de calidad y consistencia de datos para el Feature Pipeline.

Este módulo define las reglas de validación clínicas para el dataset de pacientes con
problemas hepáticos utilizando Pandera. Garantiza que los datos procesados cumplan con:
- Tipos de datos correctos en cada columna.
- Rangos fisiológicos y clínicos plausibles.
- Porcentajes máximos permitidos de valores nulos.
- Categorías válidas y formatos de marcas de tiempo UTC.
- Claves primarias únicas y consistencia entre columnas.
"""

from __future__ import annotations

import logging

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

logger = logging.getLogger("data_validation")


class DataValidationError(Exception):
    """Excepción lanzada cuando los datos no superan las reglas de validación del esquema."""

    def __init__(self, message: str, failure_cases: pd.DataFrame | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.failure_cases = failure_cases


MIN_SAMPLE_SIZE_FOR_NULL_CHECK: int = 20


def max_null_ratio(max_ratio: float = 0.05) -> pa.Check:
    """Crea un Check de Pandera para verificar el porcentaje máximo de valores nulos.

    Args:
        max_ratio: Fracción máxima de nulos permitida (por defecto: 0.05 o 5%).

    Returns:
        Check de Pandera configurado con ignore_na=False.
    """
    return pa.Check(
        lambda s: len(s) < MIN_SAMPLE_SIZE_FOR_NULL_CHECK or (s.isna().mean() <= max_ratio),
        name=f"max_null_ratio_{int(max_ratio * 100)}%",
        error=f"La proporción de valores nulos excede el máximo permitido ({max_ratio:.1%})",
        ignore_na=False,
    )


def get_patient_features_schema(max_null_fraction: float = 0.05) -> pa.DataFrameSchema:
    """Genera el esquema de validación de Pandera para las características de pacientes.

    Args:
        max_null_fraction: Proporción máxima permitida de nulos en columnas clínicas.

    Returns:
        DataFrameSchema con todas las especificaciones y reglas clínicas.
    """
    null_check = max_null_ratio(max_null_fraction)

    return pa.DataFrameSchema(
        columns={
            "patient_id": pa.Column(
                int,
                checks=[pa.Check.gt(0)],
                unique=True,
                nullable=False,
                coerce=True,
                description="Identificador único del paciente (entero positivo)",
            ),
            "event_time": pa.Column(
                checks=[
                    pa.Check(
                        pd.api.types.is_datetime64_any_dtype,
                        name="is_datetime",
                        error="event_time debe ser un tipo datetime válido",
                    ),
                    pa.Check(
                        lambda s: str(s.dt.tz) == "UTC",
                        name="is_utc_timezone",
                        error="event_time debe poseer zona horaria UTC",
                    ),
                ],
                nullable=False,
                description="Marca temporal del evento con zona horaria UTC obligatoria",
            ),
            "age": pa.Column(
                float,
                checks=[
                    pa.Check.in_range(1.0, 120.0),
                    null_check,
                ],
                nullable=True,
                coerce=True,
                description="Edad del paciente en años (rango clínico: 1 a 120)",
            ),
            "gender": pa.Column(
                checks=[
                    pa.Check.isin(["Male", "Female"]),
                    null_check,
                ],
                nullable=True,
                description="Género biológico del paciente ('Male' o 'Female')",
            ),
            "total_bilirubin": pa.Column(
                float,
                checks=[
                    pa.Check.in_range(0.0, 100.0),
                    null_check,
                ],
                nullable=True,
                coerce=True,
                description="Nivel de bilirrubina total en sangre (0.0 a 100.0 mg/dL)",
            ),
            "direct_bilirubin": pa.Column(
                float,
                checks=[
                    pa.Check.in_range(0.0, 50.0),
                    null_check,
                ],
                nullable=True,
                coerce=True,
                description="Nivel de bilirrubina directa en sangre (0.0 a 50.0 mg/dL)",
            ),
            "alkaline_phosphotase": pa.Column(
                float,
                checks=[
                    pa.Check.in_range(10.0, 3000.0),
                    null_check,
                ],
                nullable=True,
                coerce=True,
                description="Fosfatasa alcalina en suero (10.0 a 3000.0 UI/L)",
            ),
            "alamine_aminotransferase": pa.Column(
                float,
                checks=[
                    pa.Check.in_range(1.0, 3000.0),
                    null_check,
                ],
                nullable=True,
                coerce=True,
                description="Alanina aminotransferasa ALT (1.0 a 3000.0 UI/L)",
            ),
            "aspartate_aminotransferase": pa.Column(
                float,
                checks=[
                    pa.Check.in_range(1.0, 6000.0),
                    null_check,
                ],
                nullable=True,
                coerce=True,
                description="Aspartato aminotransferasa AST (1.0 a 6000.0 UI/L)",
            ),
            "total_protiens": pa.Column(
                float,
                checks=[
                    pa.Check.in_range(1.0, 15.0),
                    null_check,
                ],
                nullable=True,
                coerce=True,
                description="Proteínas totales en suero (1.0 a 15.0 g/dL)",
            ),
            "albumin": pa.Column(
                float,
                checks=[
                    pa.Check.in_range(0.5, 10.0),
                    null_check,
                ],
                nullable=True,
                coerce=True,
                description="Nivel de albúmina en suero (0.5 a 10.0 g/dL)",
            ),
            "albumin_and_globulin_ratio": pa.Column(
                float,
                checks=[
                    pa.Check.in_range(0.0, 10.0),
                    null_check,
                ],
                nullable=True,
                coerce=True,
                description="Razón albúmina/globulina A/G (0.0 a 10.0)",
            ),
            "diagnosis": pa.Column(
                checks=[
                    pa.Check.isin([1, 2, 1.0, 2.0]),
                    null_check,
                ],
                nullable=True,
                description="Diagnóstico de enfermedad hepática (1: enfermo, 2: sano)",
            ),
            "direct_to_total_bilirubin": pa.Column(
                float,
                checks=[
                    pa.Check.ge(0.0),
                    null_check,
                ],
                nullable=True,
                coerce=True,
                description="Razón directa/total de bilirrubina (valor no negativo)",
            ),
            "ast_to_alt": pa.Column(
                float,
                checks=[
                    pa.Check.ge(0.0),
                    null_check,
                ],
                nullable=True,
                coerce=True,
                description="Cociente De Ritis AST/ALT (valor no negativo)",
            ),
        },
        checks=[
            pa.Check(
                lambda df: len(df) > 0, name="dataframe_not_empty", error="El DataFrame está vacío"
            ),
        ],
        strict=True,
        ordered=False,
    )


PATIENT_FEATURES_SCHEMA = get_patient_features_schema()


def validate_patient_features(
    df: pd.DataFrame,
    schema: pa.DataFrameSchema | None = None,
    *,
    lazy: bool = True,
) -> pd.DataFrame:
    """Valida que el DataFrame cumpla rigurosamente con las reglas de calidad del esquema.

    Args:
        df: DataFrame procesado con las 15 características clínicas.
        schema: Esquema de validación a utilizar (por defecto usa PATIENT_FEATURES_SCHEMA).
        lazy: Si es True, recolecta todas las fallas antes de lanzar la excepción.

    Returns:
        El DataFrame validado si cumple con todas las reglas.

    Raises:
        DataValidationError: Si una o más validaciones del esquema no se cumplen.
    """
    validation_schema = schema if schema is not None else PATIENT_FEATURES_SCHEMA

    try:
        validated_df: pd.DataFrame = validation_schema.validate(df, lazy=lazy)
    except SchemaErrors as exc:
        failure_df = exc.failure_cases
        logger.exception(
            f"Fallo en la validación de calidad de datos con Pandera. "
            f"Se detectaron {len(failure_df)} infracciones:\n{failure_df}"
        )
        missing_cols = failure_df.loc[
            failure_df["check"] == "column_in_dataframe", "failure_case"
        ].tolist()
        err_cols = failure_df["column"].dropna().unique().tolist()
        all_affected = sorted(set(missing_cols + err_cols))
        failed_checks = failure_df["check"].dropna().unique().tolist()

        msg = (
            f"Los datos no cumplen con las reglas de calidad del Feature Store. "
            f"Total de fallas detectadas: {len(failure_df)}. "
            f"Columnas afectadas: {all_affected}. "
            f"Reglas/chequeos fallidos: {failed_checks}"
        )
        raise DataValidationError(msg, failure_cases=failure_df) from exc
    except SchemaError as exc:
        logger.exception("Error de validación en esquema.")
        msg = f"Error de validación en esquema: {exc}"
        raise DataValidationError(msg) from exc
    else:
        return validated_df
