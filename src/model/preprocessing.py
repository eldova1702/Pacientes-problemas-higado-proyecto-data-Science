"""Módulo compartido de preprocesamiento para pipelines de entrenamiento e inferencia."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, RobustScaler

TARGET_COLUMN = "diagnosis"

SKEWED_NUMERIC_FEATURES = [
    "total_bilirubin",
    "direct_bilirubin",
    "alkaline_phosphotase",
    "alamine_aminotransferase",
    "aspartate_aminotransferase",
    "direct_to_total_bilirubin",
    "ast_to_alt",
]

REGULAR_NUMERIC_FEATURES = [
    "age",
    "total_protiens",
    "albumin",
    "albumin_and_globulin_ratio",
]

CATEGORICAL_FEATURES = ["gender"]


class ColumnStandardizer(BaseEstimator, TransformerMixin):
    """Estandariza los nombres de columnas a minúsculas y snake_case para consistencia."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> ColumnStandardizer:
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = X.copy()
        transformed.columns = (
            pd.Index([str(c).strip() for c in transformed.columns])
            .str.replace(r"[^a-zA-Z0-9]+", "_", regex=True)
            .str.strip("_")
            .str.lower()
        )
        return transformed

    def get_feature_names_out(
        self, input_features: np.ndarray | list[str] | None = None
    ) -> np.ndarray:
        if input_features is None:
            input_features = self.feature_names_in_
        cleaned = (
            pd.Index([str(c).strip() for c in input_features])
            .str.replace(r"[^a-zA-Z0-9]+", "_", regex=True)
            .str.strip("_")
            .str.lower()
        )
        return np.asarray(cleaned, dtype=object)


class ClinicalFeatureBuilder(BaseEstimator, TransformerMixin):
    """Crea y valida ratios clínicos deterministas sin ajustar sobre el target."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> ClinicalFeatureBuilder:
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = X.copy()
        cols_map = {str(c).strip().lower(): c for c in transformed.columns}

        tb_col = cols_map.get("total_bilirubin")
        db_col = cols_map.get("direct_bilirubin")
        alt_col = cols_map.get("alamine_aminotransferase")
        ast_col = cols_map.get("aspartate_aminotransferase")

        # Ratio Direct_to_Total_Bilirubin
        if "direct_to_total_bilirubin" not in cols_map:
            if tb_col and db_col:
                total_b = pd.to_numeric(transformed[tb_col], errors="coerce").astype("float64")
                direct_b = pd.to_numeric(transformed[db_col], errors="coerce").astype("float64")
                transformed["direct_to_total_bilirubin"] = direct_b / total_b.replace(0, np.nan)
            else:
                transformed["direct_to_total_bilirubin"] = np.nan
        else:
            # Asegurar tipo numérico float64
            ratio_col = cols_map["direct_to_total_bilirubin"]
            transformed[ratio_col] = pd.to_numeric(transformed[ratio_col], errors="coerce").astype(
                "float64"
            )

        # Ratio AST_to_ALT (Cociente De Ritis)
        if "ast_to_alt" not in cols_map:
            if ast_col and alt_col:
                ast = pd.to_numeric(transformed[ast_col], errors="coerce").astype("float64")
                alt = pd.to_numeric(transformed[alt_col], errors="coerce").astype("float64")
                transformed["ast_to_alt"] = ast / alt.replace(0, np.nan)
            else:
                transformed["ast_to_alt"] = np.nan
        else:
            # Asegurar tipo numérico float64
            ast_alt_col = cols_map["ast_to_alt"]
            transformed[ast_alt_col] = pd.to_numeric(
                transformed[ast_alt_col], errors="coerce"
            ).astype("float64")

        return transformed

    def get_feature_names_out(
        self, input_features: np.ndarray | list[str] | None = None
    ) -> np.ndarray:
        if input_features is None:
            input_features = self.feature_names_in_
        features = [str(f).strip().lower() for f in input_features]
        if "direct_to_total_bilirubin" not in features:
            features.append("direct_to_total_bilirubin")
        if "ast_to_alt" not in features:
            features.append("ast_to_alt")
        return np.asarray(features, dtype=object)


def build_feature_pipeline() -> Pipeline:
    """Construye un pipeline de preprocesamiento robusto para entrenamiento e inferencia."""
    skewed_numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("power_transform", PowerTransformer(method="yeo-johnson", standardize=False)),
            ("scaler", RobustScaler()),
        ]
    )
    regular_numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("skewed_numeric", skewed_numeric_pipeline, SKEWED_NUMERIC_FEATURES),
            ("regular_numeric", regular_numeric_pipeline, REGULAR_NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("standardizer", ColumnStandardizer()),
            ("clinical_features", ClinicalFeatureBuilder()),
            ("preprocessor", preprocessor),
        ]
    )


def prepare_supervised_data(
    data: pd.DataFrame, target: str = TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series]:
    """Aplica limpieza determinista y retorna características y target binario."""
    prepared = data.copy()

    # Identificar columna target tolerando mayúsculas, minúsculas o alias 'Dataset'
    target_candidates = {target.strip().lower()}
    if target.strip().lower() in ("diagnosis", "dataset"):
        target_candidates.update({"diagnosis", "dataset"})

    target_match = None
    for col in prepared.columns:
        if str(col).strip().lower() in target_candidates:
            target_match = col
            break

    if target_match is None:
        msg = f"No se encontró la columna objetivo '{target}' en el DataFrame."
        raise KeyError(msg)

    # Eliminar filas con target nulo y duplicados
    prepared = prepared.dropna(subset=[target_match]).drop_duplicates()

    # Mapeo clínico: 1 -> 1 (enfermo), 2 -> 0 (sano)
    target_series = pd.to_numeric(prepared[target_match], errors="coerce")
    unique_vals = set(target_series.dropna().unique())
    if unique_vals.issubset({0, 1}):
        labels = target_series.astype("int64")
    elif unique_vals.issubset({1, 2}):
        labels = target_series.map({1: 1, 2: 0})
    else:
        msg = f"El target contiene valores inválidos: {unique_vals}. Se esperaba [1, 2] o [0, 1]."
        raise ValueError(msg)

    if labels.isna().any():
        msg = "El target contiene valores nulos o no mapeables."
        raise ValueError(msg)

    features = prepared.drop(columns=[target_match])

    # Omitir columnas de metadatos de Hopsworks si están presentes
    meta_cols = [
        c for c in features.columns if str(c).strip().lower() in ("patient_id", "event_time")
    ]
    if meta_cols:
        features = features.drop(columns=meta_cols)

    return features, labels.astype("int64")


def load_supervised_data(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Carga datos desde Parquet o CSV y los prepara para aprendizaje supervisado."""
    df = (
        pd.read_parquet(data_path)
        if data_path.suffix.lower() == ".parquet"
        else pd.read_csv(data_path)
    )
    return prepare_supervised_data(df)
