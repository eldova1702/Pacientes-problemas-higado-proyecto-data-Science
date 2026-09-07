"""Pipeline de entrenamiento de modelos (Training Pipeline) para detección de enfermedad hepática.

Este módulo implementa el pipeline de entrenamiento de la arquitectura FTI
(Feature, Training, Inference). Sus responsabilidades son:
1. Leer los datos procesados (features) desde el Feature Store (Hopsworks) o almacenamiento local.
2. Dividir los datos en conjuntos de entrenamiento y prueba (train/test) de forma estratificada.
3. Entrenar el pipeline completo incluyendo preprocesamiento y modelo de Machine Learning.
4. Evaluar el modelo con métricas clínicas (accuracy, balanced accuracy, recall, ROC-AUC, etc.).
5. Guardar los artefactos del modelo entrenado, métricas y gráficas en disco y en el Model Registry.
6. Ejecutarse de forma autónoma mediante interfaz de línea de comandos (CLI).

Uso:
    uv run python src/pipelines/training_pipeline/train_pipeline.py --dry-run
    uv run python -m src.pipelines.training_pipeline --dry-run
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Permitir ejecución directa del script
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_store import (  # noqa: E402
    get_hopsworks_project,
)
from src.model.preprocessing import (  # noqa: E402
    build_feature_pipeline,
    prepare_supervised_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_pipeline")

DEFAULT_FEATURE_GROUP_NAME = "pacientes_higado_fg"
DEFAULT_FEATURE_GROUP_VERSION = 3
DEFAULT_FEATURE_VIEW_NAME = "pacientes_higado_fv"
DEFAULT_FEATURE_VIEW_VERSION = 1
DEFAULT_MODEL_TYPE = "logistic_regression"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "liver_patient_model"
DEFAULT_LOCAL_DATA_PATH = (
    PROJECT_ROOT / "data" / "02_intermediate" / "pacientes_higado_exploracion.parquet"
)


def get_or_create_feature_view(  # noqa: PLR0913, PLR0917
    fs: Any,
    feature_group: Any,
    name: str = DEFAULT_FEATURE_VIEW_NAME,
    version: int = DEFAULT_FEATURE_VIEW_VERSION,
    labels: list[str] | None = None,
    description: str = "Feature view para entrenamiento de pacientes con problemas hepáticos",
) -> Any:
    """Obtiene o crea un Feature View en Hopsworks con el query del Feature Group.

    Args:
        fs: Objeto del Feature Store (`project.get_feature_store()`).
        feature_group: Feature Group del cual seleccionar las características.
        name: Nombre del Feature View.
        version: Versión del Feature View.
        labels: Lista de columnas que corresponden al target (por defecto: ['diagnosis']).
        description: Descripción del Feature View.

    Returns:
        Objeto Feature View de Hopsworks.
    """
    target_labels = labels if labels is not None else ["diagnosis"]
    logger.info(f"Obteniendo o creando Feature View '{name}' v{version}...")

    try:
        fv = fs.get_feature_view(name=name, version=version)
    except Exception:
        logger.info(f"Creando nueva Feature View '{name}' v{version}...")
        query = feature_group.select_all()
        return fs.create_feature_view(
            name=name,
            version=version,
            query=query,
            labels=target_labels,
            description=description,
        )
    else:
        logger.info(f"Feature View existente recuperada: '{name}' v{version}")
        return fv


def fetch_training_data(  # noqa: PLR0913, PLR0917
    use_feature_store: bool = True,
    feature_group_name: str = DEFAULT_FEATURE_GROUP_NAME,
    feature_group_version: int = DEFAULT_FEATURE_GROUP_VERSION,
    feature_view_name: str = DEFAULT_FEATURE_VIEW_NAME,
    feature_view_version: int = DEFAULT_FEATURE_VIEW_VERSION,
    local_data_path: Path | None = None,
    api_key: str | None = None,
    project_name: str | None = None,
) -> tuple[pd.DataFrame, Any, Any]:
    """Recupera los datos de entrenamiento desde el Feature Store o fallback local.

    Args:
        use_feature_store: Si es True, intenta leer desde Hopsworks.
        feature_group_name: Nombre del Feature Group en Hopsworks.
        feature_group_version: Versión del Feature Group.
        feature_view_name: Nombre del Feature View.
        feature_view_version: Versión del Feature View.
        local_data_path: Ruta a archivo local (Parquet o CSV). Si use_feature_store es False,
            se usa esta ruta o DEFAULT_LOCAL_DATA_PATH.
        api_key: API Key de Hopsworks (opcional).
        project_name: Nombre del proyecto de Hopsworks (opcional).

    Returns:
        Tupla con (df_features, feature_view_obj_or_None, hopsworks_project_or_None).
    """
    if use_feature_store:
        try:
            logger.info("Conectando con Hopsworks para recuperar características...")
            project = get_hopsworks_project(api_key=api_key, project_name=project_name)
            fs = project.get_feature_store()
            fg = fs.get_feature_group(name=feature_group_name, version=feature_group_version)
            logger.info(f"Feature Group '{feature_group_name}' v{feature_group_version} obtenido.")

            fv = get_or_create_feature_view(
                fs=fs,
                feature_group=fg,
                name=feature_view_name,
                version=feature_view_version,
                labels=["diagnosis"],
            )

            logger.info("Leyendo registros desde el Feature Store...")
            df = fg.read()
            logger.info(
                f"Lectura exitosa del Feature Store: {df.shape[0]} filas, {df.shape[1]} columnas."
            )
        except Exception as exc:
            logger.warning(
                f"No fue posible recuperar datos del Feature Store ({exc}). "
                "Verificando fallback a datos locales..."
            )
        else:
            return df, fv, project

    # Fallback o modo local explícito
    source_path = local_data_path or DEFAULT_LOCAL_DATA_PATH
    if not source_path.exists():
        msg = f"No se pudo cargar desde Feature Store y la ruta local no existe: {source_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info(f"Cargando datos de entrenamiento desde archivo local: {source_path}")
    if source_path.suffix.lower() == ".parquet":
        df_local = pd.read_parquet(source_path)
    else:
        df_local = pd.read_csv(source_path)

    logger.info(f"Datos locales cargados: {df_local.shape[0]} filas, {df_local.shape[1]} columnas.")
    return df_local, None, None


def split_training_data(
    df: pd.DataFrame,
    target_col: str = "diagnosis",
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepara y divide los datos en conjuntos train/test de forma estratificada.

    Args:
        df: DataFrame que contiene características y target.
        target_col: Nombre de la columna objetivo.
        test_size: Fracción del conjunto de prueba (por defecto: 0.20 o 20%).
        random_state: Semilla aleatoria para reproducibilidad.

    Returns:
        Tupla (X_train, X_test, y_train, y_test).
    """
    logger.info("Preparando datos y dividiendo en particiones de entrenamiento y prueba...")
    X, y = prepare_supervised_data(df, target=target_col)

    # Verificación de distribución de clases
    counts = y.value_counts().to_dict()
    logger.info(f"Distribución del target en el dataset completo: {counts}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(
        f"División completada -> Train: {len(X_train)} muestras ({y_train.sum()} positivos), "
        f"Test: {len(X_test)} muestras ({y_test.sum()} positivos)"
    )
    return X_train, X_test, y_train, y_test


def get_classifier(
    model_type: str = DEFAULT_MODEL_TYPE, random_state: int = DEFAULT_RANDOM_STATE
) -> BaseEstimator:
    """Instancia el estimador configurado para clasificación binaria.

    Args:
        model_type: Tipo de clasificador ('logistic_regression' o 'random_forest').
        random_state: Semilla aleatoria.

    Returns:
        Estimador scikit-learn instanciado.

    Raises:
        ValueError: Si el model_type no está soportado.
    """
    normalized_type = model_type.strip().lower()
    if normalized_type in ("logistic_regression", "lr"):
        logger.info("Instanciando clasificador LogisticRegression(class_weight='balanced')...")
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        )
    elif normalized_type in ("random_forest", "rf"):
        logger.info(
            "Instanciando clasificador RandomForestClassifier(n_estimators=300, class_weight='balanced')..."
        )
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        msg = (
            f"Tipo de modelo '{model_type}' no soportado. "
            f"Opciones disponibles: ['logistic_regression', 'random_forest']."
        )
        logger.error(msg)
        raise ValueError(msg)


def build_training_pipeline(
    model_type: str = DEFAULT_MODEL_TYPE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Construye el pipeline de scikit-learn acoplando preprocesamiento y modelo.

    Args:
        model_type: Tipo de modelo a instanciar.
        random_state: Semilla aleatoria.

    Returns:
        Pipeline de scikit-learn no ajustado.
    """
    feature_pipeline = build_feature_pipeline()
    classifier = get_classifier(model_type=model_type, random_state=random_state)

    return Pipeline(
        steps=[
            ("preprocessing", feature_pipeline),
            ("classifier", classifier),
        ]
    )


def train_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Ajusta el pipeline completo sobre el conjunto de entrenamiento.

    Args:
        pipeline: Pipeline que contiene preprocesamiento y clasificador.
        X_train: Características de entrenamiento.
        y_train: Etiquetas de entrenamiento (0/1).

    Returns:
        Pipeline ajustado.
    """
    logger.info("Entrenando pipeline de preprocesamiento y clasificador...")
    start_time = datetime.datetime.now(datetime.timezone.utc)
    pipeline.fit(X_train, y_train)
    elapsed = (datetime.datetime.now(datetime.timezone.utc) - start_time).total_seconds()
    logger.info(f"Entrenamiento completado exitosamente en {elapsed:.2f} segundos.")
    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Evalúa el modelo sobre el conjunto de prueba y calcula métricas clínicas exhaustivas.

    Args:
        pipeline: Pipeline entrenado.
        X_test: Características de prueba.
        y_test: Etiquetas reales de prueba.

    Returns:
        Diccionario con las métricas de rendimiento evaluadas.
    """
    logger.info("Evaluando el modelo en el conjunto de prueba...")
    y_pred = pipeline.predict(X_test)

    # Probabilidades para ROC-AUC
    y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred

    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    avg_prec = float(average_precision_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "average_precision": round(avg_prec, 4),
        "confusion_matrix": cm,
        "test_samples": len(y_test),
        "test_positive_ratio": round(float(y_test.mean()), 4),
    }

    logger.info("Métricas de evaluación obtenidas:")
    logger.info(f"  - Accuracy:          {metrics['accuracy']:.4f}")
    logger.info(f"  - Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    logger.info(f"  - Precision (clase 1): {metrics['precision']:.4f}")
    logger.info(f"  - Recall / Sensibilidad: {metrics['recall']:.4f}")
    logger.info(f"  - F1-Score:          {metrics['f1_score']:.4f}")
    logger.info(f"  - ROC-AUC:           {metrics['roc_auc']:.4f}")
    logger.info(f"  - Matriz de Confusión: {cm}")

    return metrics


def plot_and_save_figures(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> list[str]:
    """Genera y guarda gráficos diagnósticos (matriz de confusión y curva ROC).

    Args:
        pipeline: Pipeline entrenado.
        X_test: Características de prueba.
        y_test: Etiquetas de prueba.
        output_dir: Directorio donde almacenar las imágenes.

    Returns:
        Lista con las rutas de las imágenes generadas.
    """
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    generated_images: list[str] = []

    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # 1. Gráfica Matriz de Confusión
    cm_path = images_dir / "confusion_matrix.png"
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(cm, cmap="Blues", alpha=0.8)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    x=j,
                    y=i,
                    s=str(cm[i, j]),
                    va="center",
                    ha="center",
                    size="large",
                    weight="bold",
                )
        fig.colorbar(cax)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Sano (0)", "Enfermo (1)"])
        ax.set_yticklabels(["Sano (0)", "Enfermo (1)"])
        ax.set_xlabel("Predicción", labelpad=10, fontsize=11)
        ax.set_ylabel("Valor Real", labelpad=10, fontsize=11)
        ax.set_title("Matriz de Confusión - Diagnóstico Hepático", pad=15, fontsize=12)
        plt.tight_layout()
        plt.savefig(cm_path, dpi=150)
        plt.close(fig)
        generated_images.append(str(cm_path))
        logger.info(f"Gráfica de matriz de confusión guardada en: {cm_path}")
    except Exception as exc:
        logger.warning(f"No fue posible guardar matriz de confusión: {exc}")

    # 2. Gráfica Curva ROC
    if hasattr(pipeline, "predict_proba"):
        roc_path = images_dir / "roc_curve.png"
        try:
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            score = roc_auc_score(y_test, y_prob)

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {score:.3f})")
            ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Azar (AUC = 0.50)")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.05)
            ax.set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)", fontsize=10)
            ax.set_ylabel("Tasa de Verdaderos Positivos (Sensibilidad)", fontsize=10)
            ax.set_title("Curva ROC - Modelo Detección Hepática", fontsize=12)
            ax.legend(loc="lower right")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(roc_path, dpi=150)
            plt.close(fig)
            generated_images.append(str(roc_path))
            logger.info(f"Gráfica de curva ROC guardada en: {roc_path}")
        except Exception as exc:
            logger.warning(f"No fue posible guardar curva ROC: {exc}")

    return generated_images


def save_model_artifacts(  # noqa: PLR0913, PLR0917
    pipeline: Pipeline,
    metrics: dict[str, Any],
    output_dir: Path,
    model_type: str,
    feature_names: list[str],
    sync_with_main_model: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Almacena el pipeline entrenado y sus métricas en disco.

    Args:
        pipeline: Pipeline scikit-learn entrenado.
        metrics: Métricas de evaluación.
        output_dir: Directorio de destino para los artefactos.
        model_type: Identificador del modelo entrenado.
        feature_names: Nombres de las variables de entrada.
        sync_with_main_model: Si es True, sincroniza y guarda también una copia en
            models/modelo_final.joblib para compatibilidad directa con app.py.
        **kwargs: Argumentos adicionales de compatibilidad (ej. save_main_model_symlink).

    Returns:
        Diccionario con las rutas absolutas de los archivos guardados.
    """
    if "save_main_model_symlink" in kwargs:
        sync_with_main_model = bool(kwargs["save_main_model_symlink"])

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    metrics_path = output_dir / "metrics.json"

    # Estructura del artefacto compatible con app.py (que espera artifact["pipeline"])
    artifact_payload = {
        "pipeline": pipeline,
        "model_type": model_type,
        "metrics": metrics,
        "feature_names": feature_names,
        "target": "diagnosis",
        "target_mapping": {1: 1, 2: 0},
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    logger.info(f"Guardando artefacto del modelo en: {model_path}")
    joblib.dump(artifact_payload, model_path)

    logger.info(f"Guardando métricas de evaluación en: {metrics_path}")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    saved_paths = {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
    }

    # Sincronizar en models/modelo_final.joblib si corresponde
    if sync_with_main_model:
        main_model_path = PROJECT_ROOT / "models" / "modelo_final.joblib"
        try:
            main_model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(artifact_payload, main_model_path)
            saved_paths["main_model_path"] = str(main_model_path)
            logger.info(f"Modelo principal sincronizado en: {main_model_path}")
        except Exception as exc:
            logger.warning(f"No se pudo sincronizar modelo principal: {exc}")

    return saved_paths


def register_model_in_hopsworks(  # noqa: PLR0913, PLR0917
    project: Any,
    feature_view: Any,
    model_dir: Path,
    metrics: dict[str, Any],
    model_name: str = "pacientes_higado_modelo",
    description: str = "Modelo clasificador para diagnóstico de enfermedad hepática",
) -> Any:
    """Registra los artefactos del modelo en el Hopsworks Model Registry.

    Args:
        project: Objeto del proyecto Hopsworks.
        feature_view: Objeto Feature View asociado al modelo.
        model_dir: Directorio local con los artefactos del modelo.
        metrics: Métricas de evaluación para registrar en el catálogo.
        model_name: Nombre del modelo en el registro.
        description: Descripción del modelo.

    Returns:
        Objeto del modelo registrado en Hopsworks.
    """
    logger.info(f"Registrando modelo '{model_name}' en Hopsworks Model Registry...")
    mr = project.get_model_registry()

    # Formatear métricas escalares como strings para el registro
    hw_metrics = {
        k: str(v)
        for k, v in metrics.items()
        if isinstance(v, (int, float, str)) and k != "confusion_matrix"
    }

    create_kwargs: dict[str, Any] = {
        "name": model_name,
        "metrics": hw_metrics,
        "description": description,
    }
    if feature_view is not None:
        create_kwargs["feature_view"] = feature_view

    hw_model = mr.python.create_model(**create_kwargs)
    hw_model.save(str(model_dir))
    logger.info(
        f"Modelo registrado exitosamente en Hopsworks Model Registry (versión: {hw_model.version})."
    )
    return hw_model


def run_training_pipeline(  # noqa: PLR0913, PLR0917
    model_type: str = DEFAULT_MODEL_TYPE,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    use_feature_store: bool = True,
    feature_group_name: str = DEFAULT_FEATURE_GROUP_NAME,
    feature_group_version: int = DEFAULT_FEATURE_GROUP_VERSION,
    feature_view_name: str = DEFAULT_FEATURE_VIEW_NAME,
    feature_view_version: int = DEFAULT_FEATURE_VIEW_VERSION,
    local_data_path: Path | None = None,
    output_dir: Path | None = None,
    register_hopsworks: bool = True,
    dry_run: bool = False,
    api_key: str | None = None,
    project_name: str | None = None,
) -> dict[str, Any]:
    """Ejecuta el ciclo de vida completo del Training Pipeline.

    Args:
        model_type: Algoritmo de clasificación ('logistic_regression', 'random_forest').
        test_size: Fracción de test split.
        random_state: Semilla aleatoria.
        use_feature_store: Si es True y dry_run es False, lee de Hopsworks.
        feature_group_name: Nombre del Feature Group.
        feature_group_version: Versión del Feature Group.
        feature_view_name: Nombre del Feature View.
        feature_view_version: Versión del Feature View.
        local_data_path: Ruta a datos locales si no se usa Feature Store.
        output_dir: Directorio para almacenar artefactos de salida.
        register_hopsworks: Si es True, registra en Hopsworks Model Registry.
        dry_run: Si es True, ejecuta en modo local sin llamadas de red a Hopsworks.
        api_key: Llave API de Hopsworks.
        project_name: Nombre de proyecto de Hopsworks.

    Returns:
        Diccionario con el resultado de ejecución, métricas y rutas de artefactos.
    """
    logger.info("=== Iniciando Training Pipeline ===")
    target_output_dir = output_dir or DEFAULT_MODEL_DIR

    # Si dry_run está activo, forzar modo local
    effective_use_fs = False if dry_run else use_feature_store

    # 1. Extracción de datos
    df, feature_view, project = fetch_training_data(
        use_feature_store=effective_use_fs,
        feature_group_name=feature_group_name,
        feature_group_version=feature_group_version,
        feature_view_name=feature_view_name,
        feature_view_version=feature_view_version,
        local_data_path=local_data_path,
        api_key=api_key,
        project_name=project_name,
    )

    # 2. División train/test estratificada
    X_train, X_test, y_train, y_test = split_training_data(
        df=df,
        target_col="diagnosis",
        test_size=test_size,
        random_state=random_state,
    )

    # 3. Construcción del pipeline y entrenamiento
    pipeline = build_training_pipeline(model_type=model_type, random_state=random_state)
    trained_pipeline = train_model(pipeline=pipeline, X_train=X_train, y_train=y_train)

    # 4. Evaluación de rendimiento
    metrics = evaluate_model(pipeline=trained_pipeline, X_test=X_test, y_test=y_test)
    metrics["model_type"] = model_type
    metrics["train_samples"] = len(X_train)

    # 5. Generación de imágenes y almacenamiento de artefactos
    saved_paths = save_model_artifacts(
        pipeline=trained_pipeline,
        metrics=metrics,
        output_dir=target_output_dir,
        model_type=model_type,
        feature_names=list(X_train.columns),
        sync_with_main_model=True,
    )
    figures = plot_and_save_figures(
        pipeline=trained_pipeline,
        X_test=X_test,
        y_test=y_test,
        output_dir=target_output_dir,
    )
    saved_paths["figures"] = figures

    # 6. Registro opcional en Hopsworks Model Registry
    hw_model_version = None
    if register_hopsworks and not dry_run and project is not None:
        try:
            hw_model = register_model_in_hopsworks(
                project=project,
                feature_view=feature_view,
                model_dir=target_output_dir,
                metrics=metrics,
            )
            hw_model_version = hw_model.version
        except Exception as exc:
            logger.warning(f"No se pudo registrar el modelo en Hopsworks: {exc}")

    result = {
        "status": "dry_run_success" if dry_run else "success",
        "model_type": model_type,
        "metrics": metrics,
        "saved_paths": saved_paths,
        "hopsworks_model_version": hw_model_version,
    }
    logger.info(f"=== Training Pipeline Finalizado ({result['status']}) ===")
    return result


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Training Pipeline para detección de enfermedad hepática.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=DEFAULT_MODEL_TYPE,
        choices=["logistic_regression", "random_forest"],
        help="Tipo de clasificador a entrenar (logistic_regression o random_forest).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Proporción del conjunto de prueba (por defecto: 0.20).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Semilla aleatoria para reproducibilidad (por defecto: 42).",
    )
    parser.add_argument(
        "--feature-group-name",
        type=str,
        default=DEFAULT_FEATURE_GROUP_NAME,
        help="Nombre del Feature Group en Hopsworks.",
    )
    parser.add_argument(
        "--feature-group-version",
        type=int,
        default=DEFAULT_FEATURE_GROUP_VERSION,
        help="Versión del Feature Group en Hopsworks.",
    )
    parser.add_argument(
        "--feature-view-name",
        type=str,
        default=DEFAULT_FEATURE_VIEW_NAME,
        help="Nombre del Feature View en Hopsworks.",
    )
    parser.add_argument(
        "--feature-view-version",
        type=int,
        default=DEFAULT_FEATURE_VIEW_VERSION,
        help="Versión del Feature View en Hopsworks.",
    )
    parser.add_argument(
        "--local-data",
        type=Path,
        default=None,
        help="Ruta a datos locales para omitir lectura del Feature Store remoto.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directorio de salida para almacenar los artefactos del modelo.",
    )
    parser.add_argument(
        "--no-hopsworks-registry",
        action="store_true",
        help="Desactiva el registro del modelo en Hopsworks Model Registry.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Modo simulación: entrena localmente sin conectar a Hopsworks.",
    )
    return parser.parse_args(args)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada principal para CLI."""
    args = parse_args(argv)

    try:
        result = run_training_pipeline(
            model_type=args.model_type,
            test_size=args.test_size,
            random_state=args.random_state,
            use_feature_store=not args.dry_run and args.local_data is None,
            feature_group_name=args.feature_group_name,
            feature_group_version=args.feature_group_version,
            feature_view_name=args.feature_view_name,
            feature_view_version=args.feature_view_version,
            local_data_path=args.local_data,
            output_dir=args.output_dir,
            register_hopsworks=not args.no_hopsworks_registry and not args.dry_run,
            dry_run=args.dry_run,
        )
        logger.info(f"Resultado del Training Pipeline: {result['status']}")
    except Exception:
        logger.exception("Error durante la ejecución del Training Pipeline.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
