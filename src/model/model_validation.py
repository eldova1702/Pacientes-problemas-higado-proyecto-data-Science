"""Módulo de validación de rendimiento y generalización del modelo.

Implementa la validación cruzada sobre el conjunto de entrenamiento, el cálculo de
métricas de clasificación y la comparación entre los resultados de train, validación
cruzada y test para diagnosticar underfitting y overfitting.

Referencias:
    https://joserzapata.github.io/courses/ciencia-datos-en-produccion/model-validation/
"""

from __future__ import annotations

import base64
import html
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger("model_validation")

DEFAULT_CV_FOLDS = 5
DEFAULT_RANDOM_STATE = 42
PRIMARY_METRIC = "balanced_accuracy"

DEFAULT_SCORING: dict[str, str] = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1_score": "f1",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}

OVERFIT_GAP_THRESHOLD = 0.10
SEVERE_OVERFIT_GAP_THRESHOLD = 0.20
UNDERFIT_SCORE_THRESHOLD = 0.60
SEVERE_UNDERFIT_SCORE_THRESHOLD = 0.50
CV_INSTABILITY_THRESHOLD = 0.15

RECOMMENDATIONS: dict[str, list[str]] = {
    "underfitting": [
        "Incrementar la complejidad del modelo (más profundidad, más estimadores o features).",
        "Reducir la regularización o usar un modelo de mayor capacidad.",
        "Incorporar nuevas variables clínicas con poder predictivo.",
        "Revisar la calidad y consistencia de las etiquetas (labels).",
    ],
    "overfitting": [
        "Aumentar la regularización (C menor, max_depth menor, min_samples_leaf mayor).",
        "Reducir la complejidad del modelo o seleccionar un subconjunto de características.",
        "Recolectar más datos o aplicar técnicas de aumento de datos.",
        "Ajustar hiperparámetros con validación cruzada y early stopping/poda.",
    ],
    "possible_overfitting": [
        "Revisar la diferencia entre validación cruzada y test para descartar test poco representativo.",
        "Aumentar el número de folds o usar validación repetida.",
        "Analizar posibles cambios de distribución entre train y test.",
    ],
    "good_fit": [
        "Mantener la configuración actual y monitorear el rendimiento en producción.",
        "Registrar métricas y reentrenar periódicamente con nuevos datos.",
    ],
}


class ModelValidationError(Exception):
    """Excepción lanzada cuando la validación del modelo detecta fallos críticos."""


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convierte un valor a float de forma tolerante a None y NaN."""
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if np.isnan(result) else result


def cross_validate_model(  # noqa: PLR0913, PLR0917
    pipeline: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = DEFAULT_CV_FOLDS,
    scoring: dict[str, str] | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Ejecuta validación cruzada estratificada sobre el conjunto de entrenamiento.

    El pipeline completo (preprocesamiento + clasificador) se reajusta dentro de cada
    fold, por lo que no existe fuga de información desde el conjunto de validación.

    Args:
        pipeline: Pipeline o estimador a evaluar (sin necesidad de estar entrenado).
        X_train: Características de entrenamiento.
        y_train: Etiquetas de entrenamiento (0/1).
        cv_folds: Número de particiones de la validación cruzada.
        scoring: Diccionario nombre -> scorer de scikit-learn.
        random_state: Semilla aleatoria para la partición estratificada.

    Returns:
        Diccionario con métricas medias, desviaciones y valores por fold.
    """
    scorer_map = scoring if scoring is not None else DEFAULT_SCORING
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    logger.info(
        f"Ejecutando validación cruzada StratifiedKFold ({cv_folds} folds) sobre {len(X_train)} muestras..."
    )
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=splitter,
        scoring=scorer_map,
        return_train_score=True,
        error_score="raise",
    )

    metrics: dict[str, dict[str, Any]] = {}
    for name in scorer_map:
        test_scores = np.asarray(cv_results[f"test_{name}"], dtype=float)
        train_scores = np.asarray(cv_results[f"train_{name}"], dtype=float)
        metrics[name] = {
            "mean": round(float(np.mean(test_scores)), 4),
            "std": round(float(np.std(test_scores)), 4),
            "min": round(float(np.min(test_scores)), 4),
            "max": round(float(np.max(test_scores)), 4),
            "train_mean": round(float(np.mean(train_scores)), 4),
            "train_std": round(float(np.std(train_scores)), 4),
            "fold_values": [round(float(v), 4) for v in test_scores],
        }

    logger.info(
        "Validación cruzada completada. "
        + ", ".join(
            f"{name}: {data['mean']:.4f} ± {data['std']:.4f}" for name, data in metrics.items()
        )
    )

    return {
        "cv_folds": cv_folds,
        "splitter": "StratifiedKFold",
        "random_state": random_state,
        "metrics": metrics,
    }


def compute_split_metrics(
    pipeline: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, Any]:
    """Calcula métricas de clasificación para un conjunto de datos etiquetado.

    Args:
        pipeline: Pipeline entrenado.
        X: Características a evaluar.
        y: Etiquetas reales.

    Returns:
        Diccionario con las métricas calculadas sobre el conjunto.
    """
    y_pred = pipeline.predict(X)
    metrics: dict[str, Any] = {
        "n_samples": len(y),
        "positive_ratio": round(float(np.mean(y)), 4),
        "accuracy": round(float(accuracy_score(y, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, y_pred)), 4),
        "precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y, y_pred, zero_division=0)), 4),
    }

    if hasattr(pipeline, "predict_proba"):
        try:
            y_prob = pipeline.predict_proba(X)[:, 1]
            metrics["roc_auc"] = round(float(roc_auc_score(y, y_prob)), 4)
            metrics["average_precision"] = round(float(average_precision_score(y, y_prob)), 4)
        except Exception as exc:
            logger.warning(f"No fue posible calcular métricas probabilísticas: {exc}")
            metrics["roc_auc"] = None
            metrics["average_precision"] = None
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None

    return metrics


def assess_fit(
    train_metrics: dict[str, Any],
    cv_results: dict[str, Any],
    test_metrics: dict[str, Any],
    primary_metric: str = PRIMARY_METRIC,
) -> dict[str, Any]:
    """Diagnostica underfitting u overfitting comparando train, validación cruzada y test.

    Args:
        train_metrics: Métricas calculadas sobre train.
        cv_results: Resultado de `cross_validate_model`.
        test_metrics: Métricas calculadas sobre test.
        primary_metric: Métrica principal usada para el diagnóstico.

    Returns:
        Diccionario con el diagnóstico, las brechas y las acciones sugeridas.
    """
    cv_metric = cv_results.get("metrics", {}).get(primary_metric, {})
    train_score = _safe_float(train_metrics.get(primary_metric))
    cv_score = _safe_float(cv_metric.get("mean"))
    cv_std = _safe_float(cv_metric.get("std"))
    test_score = _safe_float(test_metrics.get(primary_metric))

    gap_train_cv = train_score - cv_score
    gap_cv_test = cv_score - test_score
    gap_train_test = train_score - test_score

    if cv_score < UNDERFIT_SCORE_THRESHOLD and train_score < UNDERFIT_SCORE_THRESHOLD:
        diagnosis = "underfitting"
    elif gap_train_cv > OVERFIT_GAP_THRESHOLD:
        diagnosis = "overfitting"
    elif gap_cv_test > OVERFIT_GAP_THRESHOLD:
        diagnosis = "possible_overfitting"
    else:
        diagnosis = "good_fit"

    severe = (diagnosis == "underfitting" and cv_score < SEVERE_UNDERFIT_SCORE_THRESHOLD) or (
        diagnosis == "overfitting" and gap_train_cv > SEVERE_OVERFIT_GAP_THRESHOLD
    )

    if diagnosis == "good_fit":
        status = "passed"
    elif severe:
        status = "failed"
    else:
        status = "warning"

    observations: list[str] = [
        f"Métrica principal '{primary_metric}': train={train_score:.4f}, "
        f"CV={cv_score:.4f} ± {cv_std:.4f}, test={test_score:.4f}.",
        f"Brecha train-CV={gap_train_cv:+.4f}, CV-test={gap_cv_test:+.4f}, "
        f"train-test={gap_train_test:+.4f}.",
    ]
    if cv_std > CV_INSTABILITY_THRESHOLD:
        observations.append(
            f"Alta variabilidad entre folds (std={cv_std:.4f} > {CV_INSTABILITY_THRESHOLD:.2f})."
        )

    if diagnosis == "good_fit":
        message = (
            "El modelo generaliza correctamente: no se detectan brechas significativas "
            "entre train, validación cruzada y test."
        )
    elif diagnosis == "underfitting":
        message = (
            "Posible underfitting: el rendimiento es bajo tanto en train como en validación "
            "cruzada, el modelo no captura la señal de los datos."
        )
    elif diagnosis == "overfitting":
        message = (
            "Posible overfitting: el rendimiento en train supera notablemente al de validación "
            "cruzada, el modelo memoriza los datos de entrenamiento."
        )
    else:
        message = (
            "Posible sobreajuste leve: la validación cruzada supera al test, revise la "
            "representatividad del conjunto de prueba."
        )

    return {
        "primary_metric": primary_metric,
        "train_score": round(train_score, 4),
        "cv_score": round(cv_score, 4),
        "cv_std": round(cv_std, 4),
        "test_score": round(test_score, 4),
        "gap_train_cv": round(gap_train_cv, 4),
        "gap_cv_test": round(gap_cv_test, 4),
        "gap_train_test": round(gap_train_test, 4),
        "diagnosis": diagnosis,
        "status": status,
        "observations": observations,
        "recommendations": RECOMMENDATIONS.get(diagnosis, []),
        "message": message,
    }


def plot_train_cv_test_metrics(
    train_metrics: dict[str, Any],
    cv_results: dict[str, Any],
    test_metrics: dict[str, Any],
    output_path: Path,
    metrics: list[str] | None = None,
) -> str | None:
    """Genera una gráfica de barras comparando métricas de train, CV y test.

    Args:
        train_metrics: Métricas de train.
        cv_results: Resultado de la validación cruzada.
        test_metrics: Métricas de test.
        output_path: Ruta destino de la imagen PNG.
        metrics: Métricas a graficar (por defecto: todas las disponibles).

    Returns:
        Ruta de la imagen generada, o None si falla la generación.
    """
    if metrics is None:
        cv_metrics = cv_results.get("metrics", {})
        metrics = [
            name
            for name in DEFAULT_SCORING
            if train_metrics.get(name) is not None
            and test_metrics.get(name) is not None
            and name in cv_metrics
        ]

    if not metrics:
        logger.warning("No hay métricas comunes para graficar la comparación train/CV/test.")
        return None

    train_values = [train_metrics.get(name, np.nan) for name in metrics]
    cv_values = [
        cv_results.get("metrics", {}).get(name, {}).get("mean", np.nan) for name in metrics
    ]
    test_values = [test_metrics.get(name, np.nan) for name in metrics]

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        x_pos = np.arange(len(metrics))
        width = 0.26

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x_pos - width, train_values, width, label="Train", color="#3b82f6")
        ax.bar(x_pos, cv_values, width, label="CV (media)", color="#10b981")
        ax.bar(x_pos + width, test_values, width, label="Test", color="#f59e0b")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics, rotation=30, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Puntuación")
        ax.set_title("Comparación de métricas: Train vs CV vs Test")
        ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info(f"Gráfica de métricas train/CV/test guardada en: {output_path}")
    except Exception as exc:
        logger.warning(f"No fue posible generar la gráfica de métricas: {exc}")
        return None
    return str(output_path)


def plot_learning_curve(  # noqa: PLR0913, PLR0917
    pipeline: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_path: Path,
    cv_folds: int = DEFAULT_CV_FOLDS,
    scoring: str = PRIMARY_METRIC,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> str | None:
    """Genera la curva de aprendizaje para diagnosticar underfitting u overfitting.

    Args:
        pipeline: Pipeline o estimador a evaluar.
        X_train: Características de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        output_path: Ruta destino de la imagen PNG.
        cv_folds: Número de folds de la validación cruzada.
        scoring: Scorer único a evaluar en la curva.
        random_state: Semilla aleatoria.

    Returns:
        Ruta de la imagen generada, o None si falla la generación.
    """
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    try:
        train_sizes, train_scores, valid_scores = learning_curve(
            pipeline,
            X_train,
            y_train,
            train_sizes=np.linspace(0.2, 1.0, 5),
            cv=splitter,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
    except Exception as exc:
        logger.warning(f"No fue posible calcular la curva de aprendizaje: {exc}")
        return None

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(train_sizes, train_mean, "o-", color="#3b82f6", label="Puntuación de train")
        ax.fill_between(
            train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#3b82f6"
        )
        ax.plot(train_sizes, valid_mean, "s-", color="#10b981", label="Puntuación de validación")
        ax.fill_between(
            train_sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.15, color="#10b981"
        )
        ax.set_xlabel("Muestras de entrenamiento")
        ax.set_ylabel(f"Puntuación ({scoring})")
        ax.set_title("Curva de Aprendizaje")
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info(f"Curva de aprendizaje guardada en: {output_path}")
    except Exception as exc:
        logger.warning(f"No fue posible guardar la curva de aprendizaje: {exc}")
        return None
    return str(output_path)


def generate_html_report(validation_results: dict[str, Any], report_path: Path) -> None:
    """Genera un reporte HTML con los resultados de la validación del modelo.

    Args:
        validation_results: Diccionario con los resultados de la validación.
        report_path: Ruta destino del archivo .html.
    """
    status_colors = {
        "passed": "#10b981",
        "warning": "#f59e0b",
        "failed": "#ef4444",
    }
    overall_status = validation_results.get("status", "passed")
    badge_color = status_colors.get(overall_status, "#3b82f6")

    cv_metrics = validation_results.get("cross_validation", {}).get("metrics", {})
    train_metrics = validation_results.get("train_metrics", {})
    test_metrics = validation_results.get("test_metrics", {})

    rows_html = ""
    for name, cv_data in cv_metrics.items():
        train_val = train_metrics.get(name)
        test_val = test_metrics.get(name)
        rows_html += (
            "<tr>"
            f"<td style='padding:6px 10px;'>{html.escape(name)}</td>"
            f"<td style='padding:6px 10px;'>{train_val if train_val is not None else '-'}</td>"
            f"<td style='padding:6px 10px;'>{cv_data.get('mean', '-')} ± {cv_data.get('std', '-')}</td>"
            f"<td style='padding:6px 10px;'>{test_val if test_val is not None else '-'}</td>"
            "</tr>"
        )

    fit = validation_results.get("fit_analysis", {})
    fit_color = status_colors.get(fit.get("status", "passed"), "#6b7280")
    recommendations_html = "".join(
        f"<li>{html.escape(str(rec))}</li>" for rec in fit.get("recommendations", [])
    )
    observations_html = "".join(
        f"<li>{html.escape(str(obs))}</li>" for obs in fit.get("observations", [])
    )

    figures = validation_results.get("generated_figures", [])
    figures_html = ""
    for figure in figures:
        figure_path = Path(str(figure))
        figure_name = figure_path.name

        resolved_path = (
            figure_path if figure_path.is_absolute() else report_path.parent / figure_path
        )
        if not resolved_path.exists():
            candidate = report_path.parent / "images" / figure_name
            if candidate.exists():
                resolved_path = candidate

        img_src = f"images/{html.escape(figure_name)}"
        if resolved_path.exists():
            try:
                encoded = base64.b64encode(resolved_path.read_bytes()).decode("ascii")
                img_src = f"data:image/png;base64,{encoded}"
            except Exception as exc:
                logger.warning(
                    f"No fue posible codificar la imagen '{figure_name}' en base64: {exc}"
                )

        figures_html += (
            "<div style='margin: 12px 0;'>"
            f"<img src='{img_src}' alt='{html.escape(figure_name)}' "
            "style='max-width: 100%; border: 1px solid #e5e7eb; border-radius: 6px;'/>"
            "</div>"
        )

    safe_overall_status = html.escape(str(overall_status))
    fit_message = html.escape(str(fit.get("message", "")))

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8">
    <title>Model Validation Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 30px; color: #111827; }}
        .card {{ max-width: 900px; margin: 0 auto; background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #e5e7eb; padding-bottom: 16px; margin-bottom: 20px; }}
        h1 {{ margin: 0; font-size: 20px; }}
        h2 {{ font-size: 16px; color: #374151; margin-top: 24px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border-bottom: 1px solid #e5e7eb; text-align: left; }}
        th {{ background: #f3f4f6; padding: 8px 10px; font-size: 13px; }}
    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <div>
                <h1>Model Validation Report</h1>
                <small style="color: #6b7280;">Validación cruzada y análisis de generalización</small>
            </div>
            <span style="background: {badge_color}; color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold; text-transform: uppercase; font-size: 14px;">
                {safe_overall_status}
            </span>
        </div>
        <h2>Comparación de métricas (Train / CV / Test)</h2>
        <table>
            <thead>
                <tr>
                    <th>Métrica</th>
                    <th>Train</th>
                    <th>CV (media ± std)</th>
                    <th>Test</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        <h2>Diagnóstico de ajuste</h2>
        <div style="padding: 12px; border-left: 4px solid {fit_color}; background: #f9fafb; border-radius: 4px;">
            <strong style="text-transform: uppercase;">{html.escape(str(fit.get("diagnosis", "")))}</strong>
            <p style="margin: 6px 0 0; color: #4b5563;">{fit_message}</p>
        </div>
        <h2>Observaciones</h2>
        <ul>{observations_html}</ul>
        <h2>Acciones de mejora sugeridas</h2>
        <ul>{recommendations_html}</ul>
        <h2>Visualizaciones</h2>
        {figures_html}
    </div>
</body>
</html>
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def validate_model_performance(  # noqa: C901, PLR0912, PLR0913, PLR0915, PLR0917
    pipeline: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_folds: int = DEFAULT_CV_FOLDS,
    scoring: dict[str, str] | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    primary_metric: str = PRIMARY_METRIC,
    raise_on_error: bool = False,
    output_dir: Path | None = None,
    generate_plots: bool = True,
) -> dict[str, Any]:
    """Ejecuta la suite de validación de rendimiento y generalización del modelo.

    Compara las métricas de train, validación cruzada (StratifiedKFold) y test,
    diagnostica underfitting/overfitting y genera reportes y visualizaciones.

    Args:
        pipeline: Pipeline entrenado (o sin entrenar para la validación cruzada).
        X_train: Características de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        X_test: Características de prueba.
        y_test: Etiquetas de prueba.
        cv_folds: Número de folds de la validación cruzada.
        scoring: Diccionario nombre -> scorer de scikit-learn.
        random_state: Semilla aleatoria.
        primary_metric: Métrica principal para el diagnóstico.
        raise_on_error: Si es True, lanza ModelValidationError ante fallos críticos.
        output_dir: Directorio para exportar reportes y gráficas.
        generate_plots: Si es True, genera las visualizaciones.

    Returns:
        Diccionario estructurado con métricas, diagnóstico y rutas de artefactos.

    Raises:
        ModelValidationError: Si raise_on_error es True y existe al menos un fallo crítico.
    """
    logger.info("=== Iniciando Validación de Rendimiento del Modelo ===")

    cv_results = cross_validate_model(
        pipeline=pipeline,
        X_train=X_train,
        y_train=y_train,
        cv_folds=cv_folds,
        scoring=scoring,
        random_state=random_state,
    )
    train_metrics = compute_split_metrics(pipeline, X_train, y_train)
    test_metrics = compute_split_metrics(pipeline, X_test, y_test)
    fit_analysis = assess_fit(train_metrics, cv_results, test_metrics, primary_metric)

    checks: dict[str, dict[str, Any]] = {
        "fit_quality": {
            "status": fit_analysis["status"],
            "diagnosis": fit_analysis["diagnosis"],
            "message": fit_analysis["message"],
        }
    }

    warnings: list[str] = []
    errors: list[str] = []

    primary_cv = cv_results.get("metrics", {}).get(primary_metric, {})
    cv_std = _safe_float(primary_cv.get("std"))
    if cv_std > CV_INSTABILITY_THRESHOLD:
        message = (
            f"Alta variabilidad en validación cruzada para '{primary_metric}' "
            f"(std={cv_std:.4f} > {CV_INSTABILITY_THRESHOLD:.2f})."
        )
        checks["cv_stability"] = {"status": "warning", "message": message}
        warnings.append(message)
    else:
        checks["cv_stability"] = {
            "status": "passed",
            "message": f"Validación cruzada estable (std={cv_std:.4f}).",
        }

    if fit_analysis["status"] == "failed":
        errors.append(fit_analysis["message"])
    elif fit_analysis["status"] == "warning":
        warnings.append(fit_analysis["message"])

    if errors:
        overall_status = "failed"
        passed = False
    elif warnings:
        overall_status = "warning"
        passed = True
    else:
        overall_status = "passed"
        passed = True

    for err in errors:
        logger.error(f"[ERROR] Model Validation: {err}")
    for warn in warnings:
        logger.warning(f"[WARNING] Model Validation: {warn}")

    logger.info(
        f"Diagnóstico de ajuste: {fit_analysis['diagnosis']} "
        f"(train={fit_analysis['train_score']:.4f}, cv={fit_analysis['cv_score']:.4f}, "
        f"test={fit_analysis['test_score']:.4f})."
    )

    result: dict[str, Any] = {
        "status": overall_status,
        "passed": passed,
        "summary": {
            "cv_folds": cv_folds,
            "primary_metric": primary_metric,
            "train_score": fit_analysis["train_score"],
            "cv_score": fit_analysis["cv_score"],
            "test_score": fit_analysis["test_score"],
            "diagnosis": fit_analysis["diagnosis"],
            "checks_passed": sum(1 for c in checks.values() if c.get("status") == "passed"),
            "checks_warning": sum(1 for c in checks.values() if c.get("status") == "warning"),
            "checks_failed": sum(1 for c in checks.values() if c.get("status") == "failed"),
        },
        "cross_validation": cv_results,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "fit_analysis": fit_analysis,
        "checks": checks,
        "warnings": warnings,
        "errors": errors,
        "report_paths": {},
        "generated_figures": [],
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        images_dir = output_dir / "images"

        if generate_plots:
            metrics_figure = plot_train_cv_test_metrics(
                train_metrics=train_metrics,
                cv_results=cv_results,
                test_metrics=test_metrics,
                output_path=images_dir / "train_cv_test_metrics.png",
            )
            if metrics_figure:
                result["generated_figures"].append(metrics_figure)

            learning_figure = plot_learning_curve(
                pipeline=pipeline,
                X_train=X_train,
                y_train=y_train,
                output_path=images_dir / "learning_curve.png",
                cv_folds=cv_folds,
                scoring=primary_metric,
                random_state=random_state,
            )
            if learning_figure:
                result["generated_figures"].append(learning_figure)

        json_path = output_dir / "model_validation_report.json"
        html_path = output_dir / "model_validation_report.html"
        result["report_paths"] = {"json": str(json_path), "html": str(html_path)}

        try:
            generate_html_report(result, html_path)
            logger.info(f"Reporte de validación del modelo guardado en HTML: {html_path}")
        except Exception as exc:
            logger.warning(f"No fue posible guardar reporte HTML de validación: {exc}")
            result["report_paths"].pop("html", None)

        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False, default=str)
            logger.info(f"Reporte de validación del modelo guardado en JSON: {json_path}")
        except Exception as exc:
            logger.warning(f"No fue posible guardar reporte JSON de validación: {exc}")
            result["report_paths"].pop("json", None)

    if raise_on_error and errors:
        msg = f"Fallo en la validación del modelo ({len(errors)} errores detectados): " + "; ".join(
            errors
        )
        raise ModelValidationError(msg)

    return result
