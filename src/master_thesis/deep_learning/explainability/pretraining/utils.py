from __future__ import annotations
import torch
from torch.nn.functional import sigmoid

DEFAULT_THRESHOLD = 0.4


def find_prediction_cases(
    y_true,
    y_probs,
    threshold=DEFAULT_THRESHOLD,
):
    """Find TP, TN, FP, and FN samples for a selected target."""
    cases = {
        "true_positive": [],
        "true_negative": [],
        "false_positive": [],
        "false_negative": [],
    }

    for index in range(len(y_true)):
        true_label = y_true[index]
        prediction = int(y_probs[index] >= threshold)

        if true_label == 1 and prediction == 1:
            case_type = "true_positive"
        elif true_label == 0 and prediction == 0:
            case_type = "true_negative"
        elif true_label == 0 and prediction == 1:
            case_type = "false_positive"
        elif true_label == 1 and prediction == 0:
            case_type = "false_negative"
        else:
            raise ValueError(
                "Unexpected labels/prediction: "
                f"true_label={true_label}, "
                f"prediction={prediction}"
            )

        cases[case_type].append(
            {
                "index": index,
                "true_label": true_label,
                "probability": y_probs[index],
                "prediction": prediction,
            }
        )

    return cases


def summarize_prediction_cases(cases):
    """Print a compact summary of TP/TN/FP/FN cases."""
    print("Prediction case summary")
    print("=" * 40)
    for case_type, samples in cases.items():
        print(f"{case_type:18s}: " f"{len(samples)}")


def get_representative_cases(
    cases,
    n=3,
    threshold=DEFAULT_THRESHOLD,
):
    """Select representative cases for visualization."""
    if n <= 0:
        raise ValueError("n must be greater than 0.")

    representative = {}
    for case_type, samples in cases.items():
        if not samples:
            representative[case_type] = []
            continue
        ranked = sorted(
            samples,
            key=lambda x: abs(x["probability"] - threshold),
            reverse=True,
        )
        representative[case_type] = ranked[:n]
    return representative
