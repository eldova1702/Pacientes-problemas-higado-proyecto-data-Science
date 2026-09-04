"""Módulo para la preparación y registro de características en el Feature Store (Hopsworks).

Este módulo proporciona funciones para estandarizar los datos del dataset ILPD
(Indian Liver Patient Dataset), gestionar la conexión segura con Hopsworks,
crear/obtener grupos de características (Feature Groups) y realizar el backfill
histórico de datos.
"""

from __future__ import annotations

import contextlib
import datetime
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Mapeo estándar de columnas originales/intermedias a nombres en minúsculas para Hopsworks
FEATURE_NAME_MAPPING: dict[str, str] = {
    "Age": "age",
    "Gender": "gender",
    "Total_Bilirubin": "total_bilirubin",
    "Direct_Bilirubin": "direct_bilirubin",
    "Alkaline_Phosphotase": "alkaline_phosphotase",
    "Alamine_Aminotransferase": "alamine_aminotransferase",
    "Aspartate_Aminotransferase": "aspartate_aminotransferase",
    "Total_Protiens": "total_protiens",
    "Albumin": "albumin",
    "Albumin_and_Globulin_Ratio": "albumin_and_globulin_ratio",
    "Diagnosis": "diagnosis",
    "Dataset": "diagnosis",
}

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "patient_id": "Identificador único asignado a cada paciente",
    "age": "Edad del paciente en años",
    "gender": "Género del paciente (Male / Female)",
    "total_bilirubin": "Nivel de bilirrubina total en sangre",
    "direct_bilirubin": "Nivel de bilirrubina directa conjugada en sangre",
    "alkaline_phosphotase": "Nivel de enzima fosfatasa alcalina (ALP)",
    "alamine_aminotransferase": "Nivel de enzima alanina aminotransferasa (ALT)",
    "aspartate_aminotransferase": "Nivel de enzima aspartato aminotransferasa (AST)",
    "total_protiens": "Nivel de proteínas totales en suero",
    "albumin": "Nivel de albúmina en suero",
    "albumin_and_globulin_ratio": "Razón albúmina/globulina (A/G ratio)",
    "diagnosis": "Diagnóstico de enfermedad hepática (1: enfermo, 2: sano)",
    "direct_to_total_bilirubin": "Razón de bilirrubina directa sobre bilirrubina total",
    "ast_to_alt": "Razón AST/ALT (Cociente De Ritis) para evaluación hepática",
    "event_time": "Marca temporal del evento o fecha de ingestión para el Feature Store",
}


DEFAULT_HISTORICAL_TIMESTAMP: pd.Timestamp = pd.Timestamp("2026-08-31 00:00:00+00:00")


def load_environment_variables(env_path: Path | str | None = None) -> None:
    """Carga las variables de entorno desde un archivo .env si existe."""
    if env_path is not None:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()


def prepare_patient_features_for_feature_store(
    df: pd.DataFrame,
    start_id: int = 1,
    base_timestamp: datetime.datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Prepara y estandariza el DataFrame de pacientes para el Feature Store.

    Aplica las transformaciones necesarias para Hopsworks:
    - Renombra columnas a minúsculas en formato snake_case.
    - Genera una clave primaria única (`patient_id`).
    - Añade una columna de marca temporal de evento (`event_time`).
    - Convierte tipos de datos a formatos compatibles con esquemas Avro.

    Args:
        df: DataFrame original o intermedio con las variables clínicas.
        start_id: Índice inicial para la clave primaria de pacientes.
        base_timestamp: Marca de tiempo base para los eventos históricos (por defecto
            usa DEFAULT_HISTORICAL_TIMESTAMP para garantizar reproducibilidad).

    Returns:
        DataFrame formateado y listo para inserción en el Feature Store.
    """
    df_prepared = df.copy()

    # Renombrar columnas según el estándar
    rename_dict = {
        col: FEATURE_NAME_MAPPING[col] for col in df_prepared.columns if col in FEATURE_NAME_MAPPING
    }
    df_prepared = df_prepared.rename(columns=rename_dict)

    # Convertir todas las columnas restantes a minúsculas por compatibilidad
    df_prepared.columns = [str(col).lower() for col in df_prepared.columns]

    # Generar clave primaria única si no existe
    if "patient_id" not in df_prepared.columns:
        df_prepared["patient_id"] = range(start_id, start_id + len(df_prepared))

    # Generar marca temporal de evento determinista si no existe
    if "event_time" not in df_prepared.columns:
        ts = base_timestamp if base_timestamp is not None else DEFAULT_HISTORICAL_TIMESTAMP
        df_prepared["event_time"] = pd.to_datetime(ts, utc=True)

    # Convertir tipos de datos
    if "gender" in df_prepared.columns:
        df_prepared["gender"] = df_prepared["gender"].apply(
            lambda x: (
                str(x)
                if pd.notna(x) and str(x).strip().lower() not in ("nan", "none", "")
                else None
            )
        )

    if "patient_id" in df_prepared.columns:
        df_prepared["patient_id"] = df_prepared["patient_id"].astype("int64")

    # Columnas numéricas a float64
    numeric_cols = [
        "age",
        "total_bilirubin",
        "direct_bilirubin",
        "alkaline_phosphotase",
        "alamine_aminotransferase",
        "aspartate_aminotransferase",
        "total_protiens",
        "albumin",
        "albumin_and_globulin_ratio",
        "diagnosis",
    ]
    for col in numeric_cols:
        if col in df_prepared.columns:
            df_prepared[col] = pd.to_numeric(df_prepared[col], errors="coerce")

    # Ordenar columnas dejando patient_id y event_time primero
    priority_cols = ["patient_id", "event_time"]
    remaining_cols = [col for col in df_prepared.columns if col not in priority_cols]
    df_prepared = df_prepared[priority_cols + remaining_cols]

    # Retornar preservando dtypes nativos (int64, datetime64, float64) y nulos normalizados
    return df_prepared


def get_hopsworks_project(
    api_key: str | None = None,
    project_name: str | None = None,
    host: str | None = None,
) -> Any:
    """Conecta e inicia sesión en Hopsworks de forma segura.

    Obtiene la API key desde los argumentos, variables de entorno o archivo .env.

    Args:
        api_key: Llave API de Hopsworks. Si es None, se consulta HOPSWORKS_API_KEY.
        project_name: Nombre del proyecto en Hopsworks. Si es None, se consulta
            HOPSWORKS_PROJECT_NAME.
        host: Host personalizado de Hopsworks (opcional).

    Returns:
        Objeto de proyecto de Hopsworks conectado.

    Raises:
        ImportError: Si el paquete hopsworks no está instalado.
        ValueError: Si no se encuentra una API key válida.
    """
    try:
        import hopsworks  # noqa: PLC0415
    except ImportError as exc:
        msg = (
            "El paquete 'hopsworks' no está instalado en el entorno. "
            "Instálelo con 'uv add hopsworks --group dev' (o ejecute 'make install_feature_store_libs')."
        )
        logger.exception(msg)
        raise ImportError(msg) from exc

    load_environment_variables()

    resolved_api_key = (api_key or os.getenv("HOPSWORKS_API_KEY") or "").strip()
    resolved_project = (project_name or os.getenv("HOPSWORKS_PROJECT_NAME") or "").strip()

    if resolved_api_key.startswith(('"', "'")) and resolved_api_key.endswith(('"', "'")):
        resolved_api_key = resolved_api_key[1:-1].strip()

    if resolved_project.startswith(('"', "'")) and resolved_project.endswith(('"', "'")):
        resolved_project = resolved_project[1:-1].strip()

    if not resolved_api_key:
        msg = (
            "No se encontró la variable de entorno 'HOPSWORKS_API_KEY'. "
            "Configure su API key en el archivo .env o proporciónela como argumento."
        )
        logger.error(msg)
        raise ValueError(msg)

    login_kwargs: dict[str, Any] = {"api_key_value": resolved_api_key}
    if resolved_project:
        login_kwargs["project"] = resolved_project
    if host:
        login_kwargs["host"] = host

    # En Windows, Hopsworks almacena certificados en /tmp por defecto; asegurar que exista
    if sys.platform.startswith("win"):
        with contextlib.suppress(Exception):
            os.makedirs("C:\\tmp", exist_ok=True)

    logger.info("Iniciando sesión en Hopsworks...")
    project = hopsworks.login(**login_kwargs)
    logger.info(f"Conexión exitosa al proyecto Hopsworks: {project.name}")
    return project


def get_or_create_patient_feature_group(  # noqa: PLR0913
    fs: Any,
    name: str = "pacientes_higado_fg",
    version: int = 2,
    *,
    description: str = "Grupo de características históricas de pacientes para detección de enfermedad hepática",
    primary_key: list[str] | None = None,
    event_time: str = "event_time",
    online_enabled: bool = False,
    time_travel_format: str = "HUDI",
) -> Any:
    """Crea u obtiene un Feature Group en el Feature Store de Hopsworks.

    Args:
        fs: Objeto del Feature Store (`project.get_feature_store()`).
        name: Nombre del Feature Group.
        version: Versión del Feature Group (por defecto: 2).
        description: Descripción del Feature Group.
        primary_key: Lista de columnas que componen la clave primaria.
        event_time: Columna que representa la fecha del evento.
        online_enabled: Indica si se habilita el almacenamiento online para inferencia en tiempo real.
        time_travel_format: Formato de almacenamiento temporal ('HUDI' o 'DELTA').

    Returns:
        Objeto Feature Group de Hopsworks.
    """
    pk = primary_key if primary_key is not None else ["patient_id"]

    logger.info(f"Obteniendo o creando Feature Group '{name}' v{version}...")
    return fs.get_or_create_feature_group(
        name=name,
        version=version,
        description=description,
        primary_key=pk,
        event_time=event_time,
        online_enabled=online_enabled,
        time_travel_format=time_travel_format,
    )


def backfill_patient_features_to_store(  # noqa: PLR0913
    df: pd.DataFrame,
    feature_group_name: str = "pacientes_higado_fg",
    version: int = 2,
    *,
    api_key: str | None = None,
    project_name: str | None = None,
    time_travel_format: str = "HUDI",
    wait_for_job: bool = True,
) -> dict[str, Any]:
    """Ejecuta el flujo completo de preparación y carga histórica (backfill) en Hopsworks.

    Args:
        df: DataFrame de entrada con los datos de pacientes.
        feature_group_name: Nombre del Feature Group.
        version: Versión del Feature Group (por defecto: 2).
        api_key: Llave API de Hopsworks (opcional).
        project_name: Nombre del proyecto (opcional).
        time_travel_format: Formato temporal ('HUDI' o 'DELTA').
        wait_for_job: Si es True, espera a que el trabajo de inserción finalice.

    Returns:
        Diccionario con el resumen y estado de la operación.
    """
    logger.info("Preparando datos de pacientes para el Feature Store...")
    df_prepared = prepare_patient_features_for_feature_store(df)
    logger.info(f"Datos preparados: {df_prepared.shape[0]} filas, {df_prepared.shape[1]} columnas")

    project = get_hopsworks_project(api_key=api_key, project_name=project_name)
    fs = project.get_feature_store()

    feature_group = get_or_create_patient_feature_group(
        fs=fs,
        name=feature_group_name,
        version=version,
        time_travel_format=time_travel_format,
    )

    logger.info(
        f"Insertando {len(df_prepared)} registros en el Feature Group '{feature_group_name}'..."
    )
    feature_group.insert(df_prepared, write_options={"wait_for_job": wait_for_job})
    logger.info("Inserción completada exitosamente.")

    # Adjuntar descripciones a las características reportando fallos si ocurren
    metadata_warnings: list[str] = []
    for feat_name, desc in FEATURE_DESCRIPTIONS.items():
        if feat_name in df_prepared.columns:
            try:
                feature_group.update_feature_description(feat_name, desc)
            except Exception as err:
                logger.warning(
                    f"No se pudo actualizar la descripción de la característica '{feat_name}': {err}"
                )
                metadata_warnings.append(feat_name)

    if metadata_warnings:
        logger.warning(
            f"Algunas descripciones de características no pudieron registrarse: {metadata_warnings}"
        )
    else:
        logger.info("Descripciones de características actualizadas en Hopsworks.")

    return {
        "status": "success",
        "feature_group_name": feature_group_name,
        "version": version,
        "records_inserted": len(df_prepared),
        "columns": list(df_prepared.columns),
        "metadata_warnings": metadata_warnings,
    }
