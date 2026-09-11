from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"

WAVEFORMS_PATH = DATA_DIR / "EchoNext_test_waveforms.npy"
METADATA_PATH = DATA_DIR / "echonext_metadata_100k.csv"
TABULAR_PATH = DATA_DIR / "EchoNext_test_tabular_features.npy"

CNN_LSTM_PRETRAIN_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "cnn_lstm_fusion_pretrain_123" / "best.pt"
)

CNN_LSTM_FINETUNE_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "cnn_lstm_fusion_finetune_123" / "best.pt"
)

PRETRAIN_CONFIG_PATH = PROJECT_ROOT / "configs" / "25_cnn_lstm_fusion_pretrain_123.yaml"
FINETUNE_CONFIG_PATH = PROJECT_ROOT / "configs" / "30_cnn_lstm_fusion_finetune_123.yaml"

ISOTONIC_SCALER_PATH = (
    PROJECT_ROOT / "calibration" / "isotonic_cnn_lstm_seed_123.joblib"
)

TABULAR_FEATURES = [
    "sex",
    "ventricular_rate",
    "atrial_rate",
    "pr_interval",
    "qrs_duration",
    "qt_corrected",
    "age_at_ecg",
]

ECG_LENGTH = 2500
N_LEADS = 12

THRESHOLDS = {
    "shd_moderate_or_greater_flag": 0.4,
    "aortic_regurgitation_moderate_or_greater_flag": 0.05,
    "aortic_stenosis_moderate_or_greater_flag": 0.05,
    "lvef_lte_45_flag": 0.3,
    "lvwt_gte_13_flag": 0.2,
    "mitral_regurgitation_moderate_or_greater_flag": 0.05,
    "pasp_gte_45_flag": 0.1,
    "pericardial_effusion_moderate_large_flag": 0.05,
    "pulmonary_regurgitation_moderate_or_greater_flag": 0.05,
    "rv_systolic_dysfunction_moderate_or_greater_flag": 0.15,
    "tr_max_gte_32_flag": 0.05,
    "tricuspid_regurgitation_moderate_or_greater_flag": 0.1,
}

DISEASE_NAME_MAPPING = {
    "Composite structural heart disease": "shd_moderate_or_greater_flag",
    "Aortic stenosis": "aortic_stenosis_moderate_or_greater_flag",
    "Aortic regurgitation": "aortic_regurgitation_moderate_or_greater_flag",
    "LVEF ≤ 45%": "lvef_lte_45_flag",
    "LVWT ≥ 13 mm": "lvwt_gte_13_flag",
    "Mitral regurgitation": "mitral_regurgitation_moderate_or_greater_flag",
    "PASP ≥ 45 mmHg": "pasp_gte_45_flag",
    "Pericardial effusion": "pericardial_effusion_moderate_large_flag",
    "Pulmonary regurgitation": "pulmonary_regurgitation_moderate_or_greater_flag",
    "Right ventricular systolic dysfunction": "rv_systolic_dysfunction_moderate_or_greater_flag",
    "TR Vmax ≥ 3.2 m/s": "tr_max_gte_32_flag",
    "Tricuspid regurgitation": "tricuspid_regurgitation_moderate_or_greater_flag",
}
