"""Módulo de verificación y validación de separación train/test.

Este módulo implementa las verificaciones recomendadas para evitar fuga de información
(data leakage), detectar desvíos de distribución (feature drift y label drift), y asegurar
que la partición entre conjuntos de entrenamiento y prueba sea estadísticamente representativa
antes de proceder con el entrenamiento de modelos de Machine Learning.

Referencias:
    https://joserzapata.github.io/courses/ciencia-datos-en-produccion/data-validation/train_test-checks/
"""

from __future__ import annotations

import html
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("split_validation")

CRITICAL_SAMPLE_LEAKAGE_RATIO = 0.05
MIN_REQUIRED_CLASSES = 2
MIN_SAMPLE_SIZE_FOR_KS = 5
WARNING_DRIFT_RATIO = 0.30


class TrainTestSplitValidationError(Exception):
    """Excepción lanzada cuando la partición train/test presenta fallos críticos de integridad."""


def check_index_leakage(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> dict[str, Any]:
    """Verifica que no exista solapamiento de índices entre train y test.

    Args:
        X_train: Conjunto de características de entrenamiento.
        X_test: Conjunto de características de prueba.

    Returns:
        Diccionario con el resultado de la verificación de fuga de índices.
    """
    train_idx = set(X_train.index)
    test_idx = set(X_test.index)
    overlap = train_idx.intersection(test_idx)
    overlap_count = len(overlap)
    leakage_ratio = overlap_count / max(len(test_idx), 1)

    has_leakage = overlap_count > 0
    status = "failed" if has_leakage else "passed"

    return {
        "status": status,
        "overlap_count": overlap_count,
        "leakage_ratio": round(leakage_ratio, 4),
        "overlapping_indices": list(overlap)[:10],
        "message": (
            f"Fuga de índices detectada: {overlap_count} índices compartidos entre train y test."
            if has_leakage
            else "Sin fuga de índices: Los índices de train y test son disjuntos."
        ),
    }


def check_sample_leakage(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    subset_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Detecta fuga de muestras (sample leakage) o duplicados exactos entre train y test.

    Compara las filas de ambas particiones para evitar que observaciones idénticas
    se encuentren presentes tanto en entrenamiento como en evaluación.

    Retorna:
        Diccionario con el resultado de la validación.
    """
    if X_train.empty or X_test.empty:
        return {
            "status": "failed",
            "duplicate_count": 0,
            "leakage_ratio": 0.0,
            "message": "Conjuntos vacíos para validación de mezcla de muestras.",
        }

    # Intersección por filas exactas
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    if not common_cols:
        return {
            "status": "passed",
            "duplicate_count": 0,
            "leakage_ratio": 0.0,
            "message": "No hay columnas comunes entre train y test para evaluar mezcla.",
        }

    merged = pd.merge(
        X_test[common_cols].drop_duplicates(),
        X_train[common_cols].drop_duplicates(),
        how="inner",
    )
    dup_count = len(merged)
    leakage_ratio = dup_count / max(len(X_test), 1)

    has_leakage = dup_count > 0
    # Si más del 5% de las muestras de test se repiten en train, se considera fallo crítico;
    # si es menor pero > 0, se considera advertencia.
    if dup_count == 0:
        status = "passed"
    elif leakage_ratio > CRITICAL_SAMPLE_LEAKAGE_RATIO:
        status = "failed"
    else:
        status = "warning"

    return {
        "status": status,
        "duplicate_count": dup_count,
        "leakage_ratio": round(leakage_ratio, 4),
        "message": (
            f"Mezcla de muestras detectada: {dup_count} filas de test ({leakage_ratio:.1%}) "
            "aparecen idénticas en el conjunto de entrenamiento."
            if has_leakage
            else "Sin mezcla de muestras: No hay filas de test duplicadas en entrenamiento."
        ),
    }


def check_dataset_sizes(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    expected_test_ratio: float = 0.20,
    tolerance: float = 0.10,
) -> dict[str, Any]:
    """Verifica la proporción relativa de tamaño entre los conjuntos train y test.

    Args:
        X_train: DataFrame de entrenamiento.
        X_test: DataFrame de prueba.
        expected_test_ratio: Proporción esperada de test respecto al total (ej. 0.20 para 20%).
        tolerance: Margen de tolerancia aceptable para la proporción.

    Returns:
        Diccionario con los tamaños y evaluación de proporción.
    """
    total = len(X_train) + len(X_test)
    if len(X_train) == 0 or len(X_test) == 0:
        return {
            "status": "failed",
            "train_size": len(X_train),
            "test_size": len(X_test),
            "total_samples": total,
            "actual_test_ratio": round(len(X_test) / max(total, 1), 4),
            "expected_test_ratio": round(expected_test_ratio, 4),
            "message": (
                "Uno o ambos conjuntos (train o test) están vacíos. "
                f"Train={len(X_train)}, Test={len(X_test)}."
            ),
        }

    actual_ratio = len(X_test) / total
    ratio_diff = abs(actual_ratio - expected_test_ratio)
    is_valid = ratio_diff <= tolerance

    return {
        "status": "passed" if is_valid else "warning",
        "train_size": len(X_train),
        "test_size": len(X_test),
        "total_samples": total,
        "actual_test_ratio": round(actual_ratio, 4),
        "expected_test_ratio": round(expected_test_ratio, 4),
        "message": (
            f"Proporción de tamaños adecuada: Train={len(X_train)}, "
            f"Test={len(X_test)} ({actual_ratio:.1%})."
            if is_valid
            else f"Desviación en proporción de test: Obtenido {actual_ratio:.1%}, "
            f"esperado {expected_test_ratio:.1%} ± {tolerance:.1%}."
        ),
    }


def check_new_categories(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Verifica que el conjunto de prueba no contenga categorías no observadas en entrenamiento.

    Args:
        X_train: DataFrame de entrenamiento.
        X_test: DataFrame de prueba.
        categorical_cols: Lista de columnas categóricas a verificar.

    Returns:
        Diccionario con el resultado de categorías no observadas.
    """
    if categorical_cols is None:
        cat_cols = [
            c
            for c in X_train.columns
            if pd.api.types.is_string_dtype(X_train[c])
            or pd.api.types.is_object_dtype(X_train[c])
            or isinstance(X_train[c].dtype, pd.CategoricalDtype)
        ]
    else:
        cat_cols = [c for c in categorical_cols if c in X_train.columns and c in X_test.columns]

    new_categories_found: dict[str, list[str]] = {}
    for col in cat_cols:
        train_cats = set(X_train[col].dropna().astype(str).unique())
        test_cats = set(X_test[col].dropna().astype(str).unique())
        diff = test_cats - train_cats
        if diff:
            new_categories_found[col] = sorted(list(diff))

    has_new = len(new_categories_found) > 0
    status = "failed" if has_new else "passed"

    return {
        "status": status,
        "categorical_columns_checked": cat_cols,
        "new_categories": new_categories_found,
        "message": (
            f"Nuevas categorías detectadas en test: {new_categories_found}."
            if has_new
            else "Sin nuevas categorías: Todos los niveles categóricos de test están presentes en train."
        ),
    }


def check_label_distribution(
    y_train: pd.Series,
    y_test: pd.Series,
    max_proportion_diff: float = 0.15,
) -> dict[str, Any]:
    """Evalúa que ambas clases estén presentes en train y test y que no exista label drift excesivo.

    Args:
        y_train: Serie de etiquetas de entrenamiento.
        y_test: Serie de etiquetas de prueba.
        max_proportion_diff: Diferencia máxima aceptable entre proporciones de la clase positiva (1).

    Returns:
        Diccionario con el resultado del análisis de distribución del target.
    """
    train_labels = set(y_train.dropna().unique())
    test_labels = set(y_test.dropna().unique())

    # Se requieren al menos 2 clases para problemas de clasificación binaria
    if len(train_labels) < MIN_REQUIRED_CLASSES or len(test_labels) < MIN_REQUIRED_CLASSES:
        return {
            "status": "failed",
            "train_distribution": y_train.value_counts(normalize=True).to_dict(),
            "test_distribution": y_test.value_counts(normalize=True).to_dict(),
            "message": (
                f"Falta al menos una clase clínica para clasificación binaria. "
                f"Train tiene clases {sorted(train_labels)}, Test tiene clases {sorted(test_labels)}."
            ),
        }

    # Ambas clases deben coincidir entre ambos conjuntos
    missing_in_test = train_labels - test_labels
    if missing_in_test:
        return {
            "status": "failed",
            "train_distribution": y_train.value_counts(normalize=True).to_dict(),
            "test_distribution": y_test.value_counts(normalize=True).to_dict(),
            "message": f"Clases ausentes en el conjunto de prueba: {sorted(missing_in_test)}.",
        }

    missing_in_train = test_labels - train_labels
    if missing_in_train:
        return {
            "status": "failed",
            "train_distribution": y_train.value_counts(normalize=True).to_dict(),
            "test_distribution": y_test.value_counts(normalize=True).to_dict(),
            "message": f"Clases presentes en test pero ausentes en train: {sorted(missing_in_train)}.",
        }

    train_pos_ratio = float((y_train == 1).mean())
    test_pos_ratio = float((y_test == 1).mean())
    prop_diff = abs(train_pos_ratio - test_pos_ratio)

    is_balanced = prop_diff <= max_proportion_diff
    status = "passed" if is_balanced else "warning"

    return {
        "status": status,
        "train_positive_ratio": round(train_pos_ratio, 4),
        "test_positive_ratio": round(test_pos_ratio, 4),
        "proportion_difference": round(prop_diff, 4),
        "train_counts": y_train.value_counts().to_dict(),
        "test_counts": y_test.value_counts().to_dict(),
        "message": (
            f"Distribución de clases consistente: Train pos={train_pos_ratio:.1%}, "
            f"Test pos={test_pos_ratio:.1%} (diff={prop_diff:.1%})."
            if is_balanced
            else f"Advertencia de Label Drift: Diferencia de proporciones ({prop_diff:.1%}) "
            f"supera el umbral permitido ({max_proportion_diff:.1%})."
        ),
    }


def check_feature_drift(  # noqa: C901
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Evalúa el desvío estadístico de distribución de características mediante Kolmogorov-Smirnov.

    Args:
        X_train: DataFrame de entrenamiento.
        X_test: DataFrame de prueba.
        numeric_cols: Lista de columnas numéricas a evaluar (por defecto: numéricas en común).
        alpha: Nivel de significancia estadística para rechazar la hipótesis de igual distribución.

    Returns:
        Diccionario con las pruebas de desvío por cada variable evaluada.
    """
    if numeric_cols is None:
        cols = [
            c for c in X_train.select_dtypes(include=[np.number]).columns if c in X_test.columns
        ]
    else:
        cols = [c for c in numeric_cols if c in X_train.columns and c in X_test.columns]

    drift_results: dict[str, dict[str, Any]] = {}
    drifted_columns: list[str] = []
    skipped_columns: list[str] = []
    failed_eval_columns: list[str] = []

    for col in cols:
        s_train = pd.to_numeric(X_train[col], errors="coerce").dropna()
        s_test = pd.to_numeric(X_test[col], errors="coerce").dropna()

        if len(s_train) < MIN_SAMPLE_SIZE_FOR_KS or len(s_test) < MIN_SAMPLE_SIZE_FOR_KS:
            drift_results[col] = {
                "statistic": None,
                "p_value": None,
                "drift_detected": False,
                "skipped": True,
                "reason": (
                    f"Muestras insuficientes (train: {len(s_train)}, test: {len(s_test)}, "
                    f"mínimo: {MIN_SAMPLE_SIZE_FOR_KS})."
                ),
            }
            skipped_columns.append(col)
            continue

        try:
            ks_res = stats.ks_2samp(s_train, s_test)
            p_val = float(ks_res.pvalue)
            stat = float(ks_res.statistic)
            has_drift = p_val < alpha
            drift_results[col] = {
                "statistic": round(stat, 4),
                "p_value": round(p_val, 4),
                "drift_detected": has_drift,
            }
            if has_drift:
                drifted_columns.append(col)
        except Exception as exc:
            drift_results[col] = {
                "statistic": None,
                "p_value": None,
                "drift_detected": False,
                "evaluation_error": str(exc),
            }
            failed_eval_columns.append(col)

    evaluated_count = len(cols) - len(skipped_columns) - len(failed_eval_columns)
    drift_ratio = len(drifted_columns) / max(evaluated_count, 1)

    has_issues = bool(drifted_columns or skipped_columns or failed_eval_columns)
    if not has_issues:
        status = "passed"
    elif failed_eval_columns or skipped_columns or drift_ratio > WARNING_DRIFT_RATIO:
        status = "warning"
    else:
        status = "passed"

    message_parts: list[str] = []
    if drifted_columns:
        message_parts.append(
            f"Desvío detectado en {len(drifted_columns)} variables: {drifted_columns}"
        )
    if skipped_columns:
        message_parts.append(
            f"{len(skipped_columns)} variables omitidas por datos insuficientes: {skipped_columns}"
        )
    if failed_eval_columns:
        message_parts.append(
            f"{len(failed_eval_columns)} variables con error en cálculo: {failed_eval_columns}"
        )

    msg = (
        "; ".join(message_parts)
        if message_parts
        else (
            f"Sin feature drift relevante: {len(cols)}/{len(cols)} "
            f"variables presentan distribuciones homogéneas (alpha={alpha})."
        )
    )

    return {
        "status": status,
        "drifted_columns_count": len(drifted_columns),
        "drifted_columns": drifted_columns,
        "skipped_columns": skipped_columns,
        "failed_eval_columns": failed_eval_columns,
        "total_numeric_columns": len(cols),
        "features": drift_results,
        "message": msg,
    }


def generate_html_report(validation_results: dict[str, Any], report_path: Path) -> None:
    """Genera un reporte visual HTML limpio con los resultados de las pruebas train/test.

    Args:
        validation_results: Diccionario con los resultados de todas las verificaciones.
        report_path: Ruta destino del archivo .html.
    """
    status_colors = {
        "passed": "#10b981",
        "warning": "#f59e0b",
        "failed": "#ef4444",
    }
    overall_status = validation_results.get("status", "passed")
    badge_color = status_colors.get(overall_status, "#3b82f6")

    checks_html = ""
    for check_name, check_data in validation_results.get("checks", {}).items():
        c_status = check_data.get("status", "passed")
        c_color = status_colors.get(c_status, "#6b7280")
        msg = check_data.get("message", "")

        safe_name = html.escape(str(check_name).replace("_", " ").title())
        safe_status = html.escape(str(c_status))
        safe_msg = html.escape(str(msg))

        checks_html += f"""
        <div style="margin-bottom: 12px; padding: 12px; border-left: 4px solid {c_color}; background: #f9fafb; border-radius: 4px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <strong style="font-size: 15px; color: #1f2937;">{safe_name}</strong>
                <span style="background: {c_color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; text-transform: uppercase;">{safe_status}</span>
            </div>
            <p style="margin: 6px 0 0; color: #4b5563; font-size: 14px;">{safe_msg}</p>
        </div>
        """

    safe_overall_status = html.escape(str(overall_status))
    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8">
    <title>Train/Test Split Validation Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 30px; color: #111827; }}
        .card {{ max-width: 800px; margin: 0 auto; background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #e5e7eb; padding-bottom: 16px; margin-bottom: 20px; }}
        h1 {{ margin: 0; font-size: 20px; color: #111827; }}
    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <div>
                <h1>Train/Test Split Validation Report</h1>
                <small style="color: #6b7280;">Detección de Data Leakage y Distribuciones Clínicas</small>
            </div>
            <span style="background: {badge_color}; color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold; text-transform: uppercase; font-size: 14px;">
                {safe_overall_status}
            </span>
        </div>
        <div class="checks">
            {checks_html}
        </div>
    </div>
</body>
</html>
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def validate_train_test_split(  # noqa: C901, PLR0912, PLR0913, PLR0915, PLR0917
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    target_col: str = "diagnosis",
    categorical_cols: list[str] | None = None,
    numeric_cols: list[str] | None = None,
    expected_test_ratio: float = 0.20,
    raise_on_error: bool = False,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Función testeable e independiente que ejecuta la suite de validación de separación train/test.

    Evalúa:
    1. Fuga de índices (Index Leakage).
    2. Fuga de muestras duplicadas (Sample Mix).
    3. Proporción relativa de partición (Dataset Sizes).
    4. Categorías no observadas en test (New Categories).
    5. Distribución de clases y ausencia de clases (Label Drift).
    6. Desvío estadístico de características (Feature Drift).

    Args:
        X_train: Características de entrenamiento.
        X_test: Características de prueba.
        y_train: Etiquetas de entrenamiento.
        y_test: Etiquetas de prueba.
        target_col: Nombre de la variable objetivo.
        categorical_cols: Columnas categóricas a verificar.
        numeric_cols: Columnas numéricas a verificar.
        expected_test_ratio: Fracción esperada de test (por defecto: 0.20).
        raise_on_error: Si es True, lanza TrainTestSplitValidationError ante fallos críticos.
        output_dir: Directorio para exportar los reportes en JSON y HTML.

    Returns:
        Diccionario estructurado con los resultados de todas las verificaciones.

    Raises:
        TrainTestSplitValidationError: Si raise_on_error es True y al menos un check falla críticamente.
    """
    logger.info("=== Iniciando Verificación de Separación Train/Test ===")

    checks: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    errors: list[str] = []

    # 1. Index Leakage
    idx_res = check_index_leakage(X_train, X_test)
    checks["index_leakage"] = idx_res
    if idx_res["status"] == "failed":
        errors.append(idx_res["message"])

    # 2. Sample Leakage
    sample_res = check_sample_leakage(X_train, X_test)
    checks["sample_leakage"] = sample_res
    if sample_res["status"] == "failed":
        errors.append(sample_res["message"])
    elif sample_res["status"] == "warning":
        warnings.append(sample_res["message"])

    # 3. Dataset Sizes
    size_res = check_dataset_sizes(X_train, X_test, expected_test_ratio=expected_test_ratio)
    checks["dataset_sizes"] = size_res
    if size_res["status"] == "failed":
        errors.append(size_res["message"])
    elif size_res["status"] == "warning":
        warnings.append(size_res["message"])

    # 4. New Categories
    cat_res = check_new_categories(X_train, X_test, categorical_cols=categorical_cols)
    checks["new_categories"] = cat_res
    if cat_res["status"] == "failed":
        errors.append(cat_res["message"])

    # 5. Label Distribution
    label_res = check_label_distribution(y_train, y_test)
    checks["label_drift"] = label_res
    if label_res["status"] == "failed":
        errors.append(label_res["message"])
    elif label_res["status"] == "warning":
        warnings.append(label_res["message"])

    # 6. Feature Drift
    drift_res = check_feature_drift(X_train, X_test, numeric_cols=numeric_cols)
    checks["feature_drift"] = drift_res
    if drift_res["status"] == "failed":
        errors.append(drift_res["message"])
    elif drift_res["status"] == "warning":
        warnings.append(drift_res["message"])

    # Consolidar estado general
    if errors:
        overall_status = "failed"
        passed = False
    elif warnings:
        overall_status = "warning"
        passed = True
    else:
        overall_status = "passed"
        passed = True

    # Logs estructurados
    for err in errors:
        logger.error(f"[ERROR] Train/Test Split: {err}")
    for warn in warnings:
        logger.warning(f"[WARNING] Train/Test Split: {warn}")

    if passed:
        logger.info(
            f"Verificación Train/Test completada con éxito (Estado: {overall_status}). "
            f"{len(checks) - len(errors) - len(warnings)} checks pasados, "
            f"{len(warnings)} advertencias, {len(errors)} errores."
        )

    report_paths: dict[str, str] = {}
    result = {
        "status": overall_status,
        "passed": passed,
        "summary": {
            "total_checks": len(checks),
            "checks_passed": sum(1 for c in checks.values() if c.get("status") == "passed"),
            "checks_warning": sum(1 for c in checks.values() if c.get("status") == "warning"),
            "checks_failed": sum(1 for c in checks.values() if c.get("status") == "failed"),
        },
        "checks": checks,
        "warnings": warnings,
        "errors": errors,
        "report_paths": report_paths,
    }

    # Guardar reporte si se especificó output_dir
    if output_dir is not None:
        json_path = output_dir / "train_test_validation_report.json"
        html_path = output_dir / "train_test_validation_report.html"
        output_dir.mkdir(parents=True, exist_ok=True)
        report_paths["json"] = str(json_path)
        report_paths["html"] = str(html_path)

        try:
            generate_html_report(result, html_path)
            logger.info(f"Reporte de validación train/test guardado en HTML: {html_path}")
        except Exception as exc:
            logger.warning(f"No fue posible guardar reporte HTML de validación: {exc}")
            report_paths.pop("html", None)

        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            logger.info(f"Reporte de validación train/test guardado en JSON: {json_path}")
        except Exception as exc:
            logger.warning(f"No fue posible guardar reporte JSON de validación: {exc}")
            report_paths.pop("json", None)

    if raise_on_error and errors:
        msg = (
            f"Fallo en la validación de separación train/test ({len(errors)} errores detectados): "
            + "; ".join(errors)
        )
        raise TrainTestSplitValidationError(msg)

    return result
