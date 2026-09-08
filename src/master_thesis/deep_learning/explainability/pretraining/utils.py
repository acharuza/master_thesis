from __future__ import annotations
import torch

DEFAULT_THRESHOLD = 0.4


def find_prediction_cases(
    model,
    dataset,
    target,
    device=None,
    threshold=DEFAULT_THRESHOLD,
):
    """Find TP, TN, FP, and FN samples for a selected target."""
    if device is None:
        device = next(model.parameters()).device

    model = model.to(device)
    model.eval()

    if target not in dataset.label_columns:
        raise ValueError(
            f"Target '{target}' not found in dataset labels. "
            f"Available labels: {dataset.label_columns}"
        )

    target_position = dataset.label_columns.index(target)
    cases = {
        "true_positive": [],
        "true_negative": [],
        "false_positive": [],
        "false_negative": [],
    }

    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            waveform = sample["waveform"].unsqueeze(0).to(device)
            tabular = sample.get("tabular")

            if tabular is not None:
                tabular = tabular.unsqueeze(0).to(device)

            outputs = model(
                waveform,
                tabular,
            )

            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if target not in outputs:
                raise KeyError(
                    f"Target '{target}' not found in model outputs. "
                    f"Available outputs: {list(outputs.keys())}"
                )

            logit = outputs[target].squeeze(-1)
            probability = torch.sigmoid(logit)
            probability_value = float(probability[0].cpu())
            prediction = int(probability_value >= threshold)
            true_label = float(dataset.labels[index][target_position])

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
                    "probability": probability_value,
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
