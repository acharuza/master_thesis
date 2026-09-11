import joblib
import numpy as np
import torch
from master_thesis.deep_learning.models.ecg_model import ECGModel
from utils.config import (
    CNN_LSTM_FINETUNE_CHECKPOINT,
    CNN_LSTM_PRETRAIN_CHECKPOINT,
    ISOTONIC_SCALER_PATH,
    DISEASE_NAME_MAPPING,
    THRESHOLDS,
)


def load_model(
    diseases,
    config,
    model_type,
):
    model = ECGModel(
        diseases=diseases,
        config=config,
    )

    if model_type == "pretrain":
        checkpoint_path = CNN_LSTM_PRETRAIN_CHECKPOINT
    elif model_type == "finetune":
        checkpoint_path = CNN_LSTM_FINETUNE_CHECKPOINT
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_calibration_scaler():
    return joblib.load(ISOTONIC_SCALER_PATH)


def transform_with_isotonic(
    logits,
    scalers,
):
    logits = np.asarray(logits)
    calibrated_probs = np.zeros_like(
        logits,
        dtype=float,
    )

    for d, iso in enumerate(scalers):
        if iso is not None:
            calibrated_probs[:, d] = iso.predict(logits[:, d])
        else:
            calibrated_probs[:, d] = 1.0 / (
                1.0
                + np.exp(
                    -np.clip(
                        logits[:, d],
                        -50,
                        50,
                    )
                )
            )
    return calibrated_probs


def predict_proba(
    model,
    waveform,
    tabular_features,
    calibration_scaler,
):
    with torch.inference_mode():
        outputs, _ = model(
            waveform,
            tabular_features,
        )

    logits = (
        torch.cat(
            [outputs[disease] for disease in outputs],
            dim=1,
        )
        .cpu()
        .numpy()
    )
    calibrated_probs = transform_with_isotonic(
        logits,
        calibration_scaler,
    )
    return calibrated_probs


def predict(
    model,
    waveform,
    tabular_features,
    calibration_scaler,
    prediction_target,
    disease_names,
    model_type,
):
    if model_type == "pretrain":
        disease_name = "shd_moderate_or_greater_flag"
    elif model_type == "finetune":
        disease_name = DISEASE_NAME_MAPPING[prediction_target]
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    with torch.inference_mode():
        outputs, _ = model(
            waveform,
            tabular_features,
        )

    if disease_name not in outputs:
        raise KeyError(
            f"Disease '{disease_name}' was not found "
            f"in model outputs. Available outputs: "
            f"{list(outputs.keys())}"
        )

    logits = outputs[disease_name].cpu().numpy()

    if model_type == "pretrain":
        probabilities = 1.0 / (
            1.0
            + np.exp(
                -np.clip(
                    logits,
                    -50,
                    50,
                )
            )
        )
        threshold = float(THRESHOLDS["shd_moderate_or_greater_flag"])
        probability = float(probabilities[0, 0])
    else:
        if calibration_scaler is None:
            raise RuntimeError(
                "Calibration scaler is required " "for the fine-tuned model."
            )

        disease_idx = disease_names.index(disease_name)
        calibrated_logits = np.zeros(
            (
                logits.shape[0],
                len(disease_names),
            ),
            dtype=float,
        )
        calibrated_logits[:, disease_idx] = logits[:, 0]
        probabilities = transform_with_isotonic(
            calibrated_logits,
            calibration_scaler,
        )
        threshold = float(THRESHOLDS[disease_name])
        probability = float(
            probabilities[
                0,
                disease_idx,
            ]
        )

    prediction = probability >= threshold
    return {
        "probability": probability,
        "threshold": threshold,
        "prediction": bool(prediction),
        "disease_name": disease_name,
    }
