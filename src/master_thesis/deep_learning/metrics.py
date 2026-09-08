import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
)


def compute_metrics(
    y_true,
    y_prob,
    label_names=None,
    threshold=0.5,
):
    results = {}

    n_labels = y_true.shape[1]

    aucs = []
    aps = []

    for i in range(n_labels):
        try:
            auc = roc_auc_score(
                y_true[:, i],
                y_prob[:, i],
            )
            ap = average_precision_score(
                y_true[:, i],
                y_prob[:, i],
            )
        except ValueError:
            auc = np.nan
            ap = np.nan

        aucs.append(auc)
        aps.append(ap)

    results["disease_macro_auroc"] = np.nanmean(aucs)
    results["disease_macro_auprc"] = np.nanmean(aps)

    preds = (y_prob >= threshold).astype(int)

    results["disease_macro_f1"] = f1_score(
        y_true,
        preds,
        average="macro",
        zero_division=0,
    )

    if label_names:
        for name, auc in zip(
            label_names,
            aucs,
        ):
            results[f"AUROC_{name}"] = auc

    return results


def compute_binary_metrics(
    y_true,
    y_prob,
    threshold=0.5,
):
    try:
        auroc = roc_auc_score(
            y_true,
            y_prob,
        )
    except ValueError:
        auroc = np.nan

    try:
        auprc = average_precision_score(
            y_true,
            y_prob,
        )
    except ValueError:
        auprc = np.nan

    preds = (y_prob >= threshold).astype(int)

    f1 = f1_score(
        y_true,
        preds,
        zero_division=0,
    )

    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
    }
