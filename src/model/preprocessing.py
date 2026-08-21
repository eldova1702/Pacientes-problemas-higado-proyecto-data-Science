"""Shared preprocessing pipeline for training and inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, RobustScaler

TARGET_COLUMN = "Diagnosis"

SKEWED_NUMERIC_FEATURES = [
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Direct_to_Total_Bilirubin",
    "AST_to_ALT",
]

REGULAR_NUMERIC_FEATURES = [
    "Age",
    "Total_Protiens",
    "Albumin",
    "Albumin_and_Globulin_Ratio",
]

CATEGORICAL_FEATURES = ["Gender"]


class ClinicalFeatureBuilder(BaseEstimator, TransformerMixin):
    """Create deterministic clinical ratios without fitting to the target."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> ClinicalFeatureBuilder:
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = X.copy()
        total_bilirubin = transformed["Total_Bilirubin"].astype("float64")
        alt = transformed["Alamine_Aminotransferase"].astype("float64")
        transformed["Direct_to_Total_Bilirubin"] = transformed[
            "Direct_Bilirubin"
        ] / total_bilirubin.replace(0, np.nan)
        transformed["AST_to_ALT"] = transformed["Aspartate_Aminotransferase"] / alt.replace(
            0, np.nan
        )
        return transformed

    def get_feature_names_out(
        self, input_features: np.ndarray | list[str] | None = None
    ) -> np.ndarray:
        if input_features is None:
            input_features = self.feature_names_in_
        return np.asarray(
            [*input_features, "Direct_to_Total_Bilirubin", "AST_to_ALT"],
            dtype=object,
        )


def build_feature_pipeline() -> Pipeline:
    """Build a fresh preprocessing pipeline for a training or inference flow."""
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
            ("clinical_features", ClinicalFeatureBuilder()),
            ("preprocessor", preprocessor),
        ]
    )


def prepare_supervised_data(
    data: pd.DataFrame, target: str = TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply deterministic cleaning and return features plus binary target."""
    prepared = data.copy()
    prepared["Age"] = prepared["Age"].astype("Int64")
    prepared["Gender"] = prepared["Gender"].astype("category")
    prepared[target] = prepared[target].astype("Int64").astype("category")
    prepared = prepared.dropna(subset=[target]).drop_duplicates()
    features = prepared.drop(columns=[target])
    labels = prepared[target].map({1: 1, 2: 0})
    if labels.isna().any():
        raise ValueError("El target contiene valores distintos de 1 y 2.")
    return features, labels.astype("int64")


def load_supervised_data(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the intermediate Parquet and prepare it for supervised learning."""
    return prepare_supervised_data(pd.read_parquet(data_path))
