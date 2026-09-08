from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.metrics import fbeta_score, make_scorer
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from .ml_classifier_factory import ClassifierFactory

F2_SCORER = make_scorer(fbeta_score, beta=2)
SCORING_REGISTRY = {
    "f1": "f1",
    "f2": F2_SCORER,
    "precision": "precision",
    "recall": "recall",
}

HYPERPARAMETERS = {
    "RandomForest": {
        "n_estimators": [100, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 10],
        "min_samples_leaf": [1, 2],
    },
    "ExtraTrees": {
        "n_estimators": [100, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 10],
        "min_samples_leaf": [1, 2],
    },
    "XGBoost": {
        "n_estimators": [200, 400],
        "learning_rate": [0.01, 0.05],
        "max_depth": [3, 6],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_lambda": [1, 10],
    },
    "LGBM": {
        "n_estimators": [200, 400],
        "learning_rate": [0.01, 0.05],
        "max_depth": [3, 6],
        "num_leaves": [31, 63],
        "feature_fraction": [0.8, 1.0],
        "bagging_fraction": [0.8, 1.0],
        "lambda_l2": [1, 10],
    },
    "CatBoost": {
        "iterations": [200, 400],
        "learning_rate": [0.01, 0.05],
        "depth": [4, 6, 8],
        "l2_leaf_reg": [3, 10],
    },
    "LogisticRegression": {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
    },
    # "SVC": {
    #     "C": [0.1, 1, 10],
    #     "kernel": ["linear"],
    # },
}


class ModelEvaluator:

    def __init__(
        self,
        seeds,
        results_dir="results",
        scoring_metric="f1",
    ):
        self.results_dir = results_dir
        self.scoring_metric = scoring_metric
        self.best_models = {}
        self.tuning_results = {}
        self.test_results = {}
        self.seeds = seeds

        os.makedirs(results_dir, exist_ok=True)
        self._load_checkpoint()

    def _compute_all_metrics(self, y_true, y_pred, y_pred_proba):
        metrics = {
            "recall": recall_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "f2": fbeta_score(y_true, y_pred, beta=2),
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
            "pr_auc": average_precision_score(y_true, y_pred_proba),
        }
        return metrics

    def _concat_rows(self, first, second):
        if isinstance(first, (pd.DataFrame, pd.Series)):
            return pd.concat([first, second], axis=0).reset_index(drop=True)
        return np.concatenate([first, second], axis=0)

    def tune_model(self, X_train, y_train, X_val, y_val, classifier_name):
        if classifier_name in self.tuning_results:
            return self.tuning_results[classifier_name]

        param_grid = HYPERPARAMETERS[classifier_name]
        seeds = self.seeds

        X_search = self._concat_rows(X_train, X_val)
        y_search = self._concat_rows(y_train, y_val)
        val_split = PredefinedSplit(
            test_fold=np.concatenate(
                [
                    np.full(len(y_train), -1, dtype=int),
                    np.zeros(len(y_val), dtype=int),
                ]
            )
        )

        backend_scorer = SCORING_REGISTRY.get(self.scoring_metric, self.scoring_metric)

        params_to_scores = {}
        params_lookup = {}
        for seed in seeds:
            base_model = ClassifierFactory.get_classifier(
                classifier_name, random_state=seed
            )
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=val_split,
                scoring=backend_scorer,
                n_jobs=-1,
                verbose=0,
                refit=False,
            )
            grid_search.fit(X_search, y_search)

            for params, score in zip(
                grid_search.cv_results_["params"],
                grid_search.cv_results_["mean_test_score"],
            ):
                key = json.dumps(params, sort_keys=True)
                params_lookup[key] = params
                params_to_scores.setdefault(key, []).append(float(score))

        best_key = max(
            params_to_scores,
            key=lambda key: (
                np.mean(params_to_scores[key]),
                -np.std(params_to_scores[key]),
            ),
        )
        best_params = params_lookup[best_key]
        selected_scores = params_to_scores[best_key]

        result = {
            "best_params": best_params,
            "best_val_score_mean": float(np.mean(selected_scores)),
            "best_val_score_std": float(np.std(selected_scores)),
            "val_score": self.scoring_metric,
            "seeds": list(self.seeds),
        }
        self.tuning_results[classifier_name] = result
        self._save_tuning_results()
        return result

    def tune_all_models(self, X_train, y_train, X_val, y_val):
        classifier_names = list(HYPERPARAMETERS.keys())
        pbar = tqdm(classifier_names, desc="Models")
        for classifier_name in pbar:
            pbar.set_postfix({"model": classifier_name})
            self.tune_model(X_train, y_train, X_val, y_val, classifier_name)

    def evaluate_on_test_set(self, X_train, y_train, X_val, y_val, X_test, y_test):
        if not self.tuning_results:
            raise ValueError(
                "No tuning results available. Run tune_model/tune_all_models first."
            )

        X_fit = self._concat_rows(X_train, X_val)
        y_fit = self._concat_rows(y_train, y_val)
        aggregated_test_results = {}

        for classifier_name, tuning_result in self.tuning_results.items():
            best_params = tuning_result["best_params"]
            metric_runs = {
                "recall": [],
                "precision": [],
                "f1": [],
                "roc_auc": [],
                "pr_auc": [],
            }

            for seed in self.seeds:
                model = ClassifierFactory.get_classifier(
                    classifier_name, random_state=seed
                )
                model.set_params(**best_params)
                model.fit(X_fit, y_fit)

                y_test_pred = model.predict(X_test)
                y_test_pred_proba = model.predict_proba(X_test)[:, 1]
                metrics = self._compute_all_metrics(
                    y_test, y_test_pred, y_test_pred_proba
                )

                for metric_name, value in metrics.items():
                    metric_runs[metric_name].append(float(value))

            aggregated_metrics = {}
            for metric_name, values in metric_runs.items():
                aggregated_metrics[f"{metric_name}_mean"] = float(np.mean(values))
                aggregated_metrics[f"{metric_name}_std"] = float(np.std(values))

            aggregated_metrics["seeds"] = list(self.seeds)
            aggregated_metrics["best_params"] = best_params
            aggregated_test_results[classifier_name] = aggregated_metrics

        self.test_results = aggregated_test_results
        self._save_test_results(aggregated_test_results)

        results_df = pd.DataFrame(aggregated_test_results).T
        mean_col = f"{self.scoring_metric}_mean"
        std_col = f"{self.scoring_metric}_std"
        rank_col = f"{self.scoring_metric}_mean_minus_std"
        results_df[rank_col] = results_df[mean_col] - results_df[std_col]
        results_df = results_df.sort_values(rank_col, ascending=False)

        return results_df

    def get_best_model(self, metric="f1"):
        if not self.test_results:
            print("No test results available. Run evaluate_on_test_set first.")
            return None

        metric_mean_key = f"{metric}_mean"
        metric_std_key = f"{metric}_std"
        best_model_name = max(
            self.test_results,
            key=lambda x: self.test_results[x][metric_mean_key]
            - self.test_results[x][metric_std_key],
        )
        return best_model_name, self.tuning_results[best_model_name]["best_params"]

    def _load_checkpoint(self):
        tuning_file = os.path.join(self.results_dir, "tuning_results.json")
        if os.path.exists(tuning_file):
            with open(tuning_file, "r") as f:
                self.tuning_results = json.load(f)

        test_file = os.path.join(self.results_dir, "test_results.json")
        if os.path.exists(test_file):
            with open(test_file, "r") as f:
                self.test_results = json.load(f)

    def _save_tuning_results(self):
        filepath = os.path.join(self.results_dir, "tuning_results.json")
        with open(filepath, "w") as f:
            json.dump(self.tuning_results, f, indent=2)

    def _save_test_results(self, test_results):
        filepath = os.path.join(self.results_dir, "test_results.json")
        with open(filepath, "w") as f:
            json.dump(test_results, f, indent=2)
