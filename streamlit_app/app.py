import streamlit as st
from components.ecg_viewer import display_ecg
from components.explainability import display_explainability
from components.sidebar import render_sidebar
from data.data_loader import load_dataset
from models.model import (
    load_calibration_scaler,
    load_model,
    predict,
)
from master_thesis.deep_learning.explainability.pretraining.local import (
    explain_patient,
)
from master_thesis.deep_learning.explainability.common.grad_cam import (
    compute_cnn_grad_cam,
)
from master_thesis.deep_learning.utils import load_config
from utils.config import (
    DISEASE_NAME_MAPPING,
    FINETUNE_CONFIG_PATH,
    PRETRAIN_CONFIG_PATH,
    TABULAR_FEATURES,
    THRESHOLDS,
)


@st.cache_resource
def get_dataset(
    pretrain=False,
):
    return load_dataset(pretrain=pretrain)


@st.cache_resource
def get_model(
    diseases,
    config,
    model_type,
):
    return load_model(
        diseases=diseases,
        config=config,
        model_type=model_type,
    )


@st.cache_resource
def get_calibration_scaler():
    return load_calibration_scaler()


def initialise_session_state():
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "explanation" not in st.session_state:
        st.session_state.explanation = None
    if "prediction_context" not in st.session_state:
        st.session_state.prediction_context = None


def main():
    st.set_page_config(
        page_title="ECG SHD Detection",
        layout="wide",
    )
    st.title("Explainable ECG-based SHD Detection")
    initialise_session_state()

    pretrain_dataset, pretrain_metadata = get_dataset(pretrain=True)
    finetune_dataset, finetune_metadata = get_dataset(pretrain=False)

    settings = render_sidebar(
        pretrain_dataset=pretrain_dataset,
        finetune_dataset=finetune_dataset,
    )

    prediction_target = settings["prediction_target"]
    selected_ecg = settings["selected_ecg"]
    model_type = settings["model_type"]
    run_prediction = settings["run_prediction"]

    if model_type == "pretrain":
        dataset = pretrain_dataset
        metadata = pretrain_metadata
        diseases = ["shd_moderate_or_greater_flag"]
        config_path = PRETRAIN_CONFIG_PATH
        explanation_target = "shd_moderate_or_greater_flag"
    else:
        dataset = finetune_dataset
        metadata = finetune_metadata
        diseases = [
            disease
            for disease in DISEASE_NAME_MAPPING.values()
            if disease != "shd_moderate_or_greater_flag"
        ]
        config_path = FINETUNE_CONFIG_PATH
        explanation_target = DISEASE_NAME_MAPPING[prediction_target]

    config = load_config(config_path)
    sample = dataset[selected_ecg]
    metadata_row = metadata.iloc[selected_ecg]

    model = get_model(
        diseases=diseases,
        config=config,
        model_type=model_type,
    )

    current_context = (
        selected_ecg,
        prediction_target,
        model_type,
    )
    if (
        st.session_state.prediction_context is not None
        and st.session_state.prediction_context != current_context
    ):

        st.session_state.prediction_result = None
        st.session_state.explanation = None

    viewer_tab, prediction_explainability_tab = st.tabs(
        [
            "Viewer",
            "Prediction & Explainability",
        ]
    )

    with viewer_tab:
        display_ecg(
            sample=sample,
            metadata_row=metadata_row,
            ecg_index=selected_ecg,
        )

    with prediction_explainability_tab:
        if run_prediction:
            waveform = sample["waveform"].unsqueeze(0)
            tabular_features = sample["tabular"].unsqueeze(0)

            calibration_scaler = None
            if model_type == "finetune":
                calibration_scaler = get_calibration_scaler()

            result = predict(
                model=model,
                waveform=waveform,
                tabular_features=tabular_features,
                calibration_scaler=calibration_scaler,
                prediction_target=prediction_target,
                disease_names=diseases,
                model_type=model_type,
            )

            with st.spinner("Generating explanation..."):
                threshold = THRESHOLDS[explanation_target]
                explanation = explain_patient(
                    model=model,
                    dataset=dataset,
                    index=selected_ecg,
                    target=explanation_target,
                    threshold=threshold,
                )
                grad_cam_result = compute_cnn_grad_cam(
                    model=model,
                    waveform=waveform,
                    target=explanation_target,
                    tabular=tabular_features,
                )
                explanation["grad_cam"] = grad_cam_result

            st.session_state.prediction_result = result
            st.session_state.explanation = explanation
            st.session_state.prediction_context = current_context

        if st.session_state.prediction_result is None:
            st.info(
                "Select an ECG recording and click " "'Run prediction' in the sidebar."
            )
        else:
            result = st.session_state.prediction_result
            explanation = st.session_state.explanation

            st.header("Prediction")
            st.subheader(prediction_target)
            probability_column, threshold_column = st.columns(2)
            probability_column.metric(
                "Probability",
                f"{result['probability']:.3f}",
            )
            threshold_column.metric(
                "Decision threshold",
                f"{result['threshold']:.3f}",
            )

            if result["prediction"]:
                st.warning("The patient may have this condition.")
            else:
                st.info("The patient probably does not have " "this condition.")

            st.divider()
            st.header("Explainability")
            st.caption(
                "The following explanations show which "
                "input information the model relied on "
                "for this prediction."
            )

            display_explainability(
                explanation=explanation,
                waveform=sample["waveform"].numpy(),
                feature_names=TABULAR_FEATURES,
            )


if __name__ == "__main__":
    main()
