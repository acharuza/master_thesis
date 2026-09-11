import streamlit as st

PREDICTION_TARGETS = [
    "Composite structural heart disease",
    "Aortic stenosis",
    "Aortic regurgitation",
    "LVEF ≤ 45%",
    "LVWT ≥ 13 mm",
    "Mitral regurgitation",
    "PASP ≥ 45 mmHg",
    "Pericardial effusion",
    "Pulmonary regurgitation",
    "Right ventricular systolic dysfunction",
    "TR Vmax ≥ 3.2 m/s",
    "Tricuspid regurgitation",
]


def render_sidebar(
    pretrain_dataset,
    finetune_dataset,
):
    with st.sidebar:
        st.header("Model")
        st.info("CNN-LSTM")

        prediction_target = st.selectbox(
            "Prediction target",
            PREDICTION_TARGETS,
        )

        if prediction_target == "Composite structural heart disease":
            dataset = pretrain_dataset
            model_type = "pretrain"
        else:
            dataset = finetune_dataset
            model_type = "finetune"

        st.divider()
        st.header("ECG")

        selected_ecg = st.selectbox(
            "Select ECG recording",
            options=list(range(len(dataset))),
            format_func=lambda x: f"ECG {x}",
        )

        st.divider()
        run_prediction = st.button(
            "Run prediction",
            type="primary",
            use_container_width=True,
        )

    return {
        "prediction_target": prediction_target,
        "selected_ecg": selected_ecg,
        "run_prediction": run_prediction,
        "model_type": model_type,
    }
