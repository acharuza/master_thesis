from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


class ClassifierFactory:

    DEFAULT_PARAMS = {
        "RandomForest": {"n_estimators": 100, "random_state": 123},
        "ExtraTrees": {"n_estimators": 100, "random_state": 123},
        "XGBoost": {"n_estimators": 100, "random_state": 123},
        "LGBM": {"n_estimators": 100, "random_state": 123},
        "CatBoost": {
            "iterations": 100,
            "random_seed": 123,
            "verbose": 0,
            "allow_writing_files": False,
        },
        "LogisticRegression": {"max_iter": 1000, "random_state": 123},
        "SVC": {"probability": False, "random_state": 123},
    }

    @staticmethod
    def get_classifier(name, **kwargs):
        classifiers = {
            "RandomForest": RandomForestClassifier,
            "ExtraTrees": ExtraTreesClassifier,
            "XGBoost": XGBClassifier,
            "LGBM": LGBMClassifier,
            "CatBoost": CatBoostClassifier,
            "LogisticRegression": LogisticRegression,
            "SVC": SVC,
        }

        if name not in classifiers:
            raise ValueError(f"Classifier '{name}' is not supported.")

        params = ClassifierFactory.DEFAULT_PARAMS.get(name, {}).copy()
        params.update(kwargs)

        # CatBoost fix
        if name == "CatBoost" and "random_state" in params:
            params["random_seed"] = params.pop("random_state")

        return classifiers[name](**params)
