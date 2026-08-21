"""Streamlit demo for the selected liver-disease model."""

from pathlib import Path
from typing import Any, cast

import pandas as pd
import streamlit as st
from joblib import load

# Import the module so joblib can resolve the custom transformer during loading.
from src.model import preprocessing as _preprocessing  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "modelo_final.joblib"

FEATURE_LABELS = {
    "Age": "Edad",
    "Gender": "Género",
    "Total_Bilirubin": "Bilirrubina total",
    "Direct_Bilirubin": "Bilirrubina directa",
    "Alkaline_Phosphotase": "Fosfatasa alcalina",
    "Alamine_Aminotransferase": "ALT (Alamina aminotransferasa)",
    "Aspartate_Aminotransferase": "AST (Aspartato aminotransferasa)",
    "Total_Protiens": "Proteínas totales",
    "Albumin": "Albúmina",
    "Albumin_and_Globulin_Ratio": "Razón albúmina/globulina",
}


@st.cache_resource
def load_model_artifact() -> dict:
    """Load the trained pipeline once per Streamlit process."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")
    return cast(dict[str, Any], load(MODEL_PATH))


def build_input_frame(values: dict[str, float | str]) -> pd.DataFrame:
    """Create the raw feature frame expected by the persisted pipeline."""
    return pd.DataFrame([values])


st.set_page_config(
    page_title="Predicción de enfermedad hepática",
    page_icon="🩺",
    layout="centered",
)

st.title("Predicción de enfermedad hepática")
st.write(
    "Ingrese los resultados clínicos para obtener una predicción del modelo. "
    "Esta herramienta es una demostración académica y no sustituye un diagnóstico médico."
)

try:
    artifact = load_model_artifact()
except (FileNotFoundError, ModuleNotFoundError, AttributeError) as error:
    st.error(f"No fue posible cargar el modelo: {error}")
    st.stop()

pipeline = artifact["pipeline"]

with st.form("prediction_form"):
    st.subheader("Datos del paciente")
    first_column, second_column = st.columns(2)

    with first_column:
        age = st.number_input(FEATURE_LABELS["Age"], min_value=0, max_value=120, value=45, step=1)
        gender = st.selectbox(FEATURE_LABELS["Gender"], options=["Male", "Female"])
        total_bilirubin = st.number_input(
            FEATURE_LABELS["Total_Bilirubin"], min_value=0.0, value=1.0, step=0.1
        )
        direct_bilirubin = st.number_input(
            FEATURE_LABELS["Direct_Bilirubin"], min_value=0.0, value=0.3, step=0.1
        )
        alkaline_phosphotase = st.number_input(
            FEATURE_LABELS["Alkaline_Phosphotase"], min_value=0.0, value=210.0, step=1.0
        )

    with second_column:
        alamine_aminotransferase = st.number_input(
            FEATURE_LABELS["Alamine_Aminotransferase"], min_value=0.0, value=36.0, step=1.0
        )
        aspartate_aminotransferase = st.number_input(
            FEATURE_LABELS["Aspartate_Aminotransferase"], min_value=0.0, value=42.0, step=1.0
        )
        total_protiens = st.number_input(
            FEATURE_LABELS["Total_Protiens"], min_value=0.0, value=6.5, step=0.1
        )
        albumin = st.number_input(FEATURE_LABELS["Albumin"], min_value=0.0, value=3.1, step=0.1)
        albumin_globulin_ratio = st.number_input(
            FEATURE_LABELS["Albumin_and_Globulin_Ratio"], min_value=0.0, value=0.9, step=0.1
        )

    submitted = st.form_submit_button("Realizar predicción", type="primary")

if submitted:
    input_values = {
        "Age": age,
        "Gender": gender,
        "Total_Bilirubin": total_bilirubin,
        "Direct_Bilirubin": direct_bilirubin,
        "Alkaline_Phosphotase": alkaline_phosphotase,
        "Alamine_Aminotransferase": alamine_aminotransferase,
        "Aspartate_Aminotransferase": aspartate_aminotransferase,
        "Total_Protiens": total_protiens,
        "Albumin": albumin,
        "Albumin_and_Globulin_Ratio": albumin_globulin_ratio,
    }
    input_frame = build_input_frame(input_values)
    prediction = int(pipeline.predict(input_frame)[0])
    probabilities = pipeline.predict_proba(input_frame)[0]
    model_classes = list(pipeline.classes_)
    disease_probability = float(probabilities[model_classes.index(1)])

    st.divider()
    if prediction == 1:
        st.error("Resultado del modelo: posible enfermedad hepática")
    else:
        st.success("Resultado del modelo: sin enfermedad hepática detectada")

    st.metric("Probabilidad estimada de enfermedad", f"{disease_probability:.1%}")
    st.caption(
        "El modelo fue entrenado con datos del Indian Liver Patient Dataset y su resultado "
        "requiere validación clínica independiente."
    )
