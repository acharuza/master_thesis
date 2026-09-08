import os
import warnings
import joblib
import numpy as np
import pandas as pd
import neurokit2 as nk
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from .constants import COLUMNS


def is_lead_corrupted(
    signal, sampling_rate=250, window_seconds=2, variance_threshold=1e-4
):
    """
    Checks if a preprocessed, normalized lead has an extended flatline.
    """
    window_size = window_seconds * sampling_rate

    if len(signal) < window_size:
        return True

    for i in range(0, len(signal) - window_size, sampling_rate):
        window = signal[i : i + window_size]

        if np.var(window) < variance_threshold:
            return True

    return False


def _extract_and_process_single_sample(ecg_data, sample_idx):
    try:
        single_ecg = ecg_data[sample_idx]
        candidate_leads = []

        try:
            candidate_leads.append(np.asarray(single_ecg[:, 1], dtype=float))
        except Exception:
            pass
        try:
            candidate_leads.append(np.asarray(single_ecg[:, 0], dtype=float))
        except Exception:
            pass
        try:
            candidate_leads.append(np.asarray(single_ecg[:, 2], dtype=float))
        except Exception:
            pass
        try:
            candidate_leads.append(np.asarray(single_ecg, dtype=float).mean(axis=1))
        except Exception:
            candidate_leads.append(np.asarray(single_ecg, dtype=float).ravel())

        valid_leads = []
        for lead in candidate_leads:
            try:
                lead = np.ravel(lead)
                lead = np.nan_to_num(lead)
                if is_lead_corrupted(lead):
                    continue
                valid_leads.append(lead)
            except Exception:
                continue

        if len(valid_leads) == 0:
            fallback = _fallback_features(np.zeros(2500))
            fallback["sample_idx"] = sample_idx
            fallback["error"] = "all_leads_corrupted"
            return fallback, "all_leads_corrupted"

        best_lead = None
        best_score = -np.inf
        for lead in valid_leads:
            try:
                score = _lead_quality_score(lead)
                if score > best_score:
                    best_score = score
                    best_lead = lead
            except Exception:
                continue

        if best_lead is None:
            fallback = _fallback_features(np.zeros(2500))
            fallback["sample_idx"] = sample_idx
            fallback["error"] = "lead_selection_failed"
            return fallback, "lead_selection_failed"

        features_df = _process_lead(best_lead)
        features_df["sample_idx"] = sample_idx

        return features_df, None

    except Exception as exc:
        fallback = _fallback_features(np.zeros(2500))
        fallback["sample_idx"] = sample_idx
        fallback["error"] = f"critical: {type(exc).__name__}: {exc}"
        return fallback, str(exc)


def _lead_quality_score(signal, sampling_rate=250):
    try:
        signal = np.nan_to_num(signal)
        var = np.var(signal)
        _, info = nk.ecg_peaks(signal, sampling_rate=sampling_rate)
        rpeaks = info["ECG_R_Peaks"]
        peak_density = len(rpeaks) / (len(signal) / sampling_rate)
        dyn_range = np.max(signal) - np.min(signal)
        score = (0.5 * var) + (0.3 * peak_density) + (0.2 * dyn_range)
        return score
    except Exception:
        return -1


def _process_lead(lead_signal, sampling_rate=250):
    features = {}

    try:
        signal_clean = nk.ecg_clean(lead_signal, sampling_rate=sampling_rate)
    except Exception:
        signal_clean = lead_signal
    try:
        peaks, info = nk.ecg_peaks(signal_clean, sampling_rate=sampling_rate)
        rpeaks = info["ECG_R_Peaks"]
    except Exception:
        return _fallback_features(signal_clean)

    if len(rpeaks) < 3:
        return _fallback_features(signal_clean)

    try:
        rr = np.diff(rpeaks) / sampling_rate
        features["rr_mean"] = np.mean(rr)
        features["rr_std"] = np.std(rr)
        features["rmssd"] = (
            np.sqrt(np.mean(np.diff(rr) ** 2)) if len(rr) > 1 else np.nan
        )
    except Exception:
        features["rr_mean"] = np.nan
        features["rr_std"] = np.nan
        features["rmssd"] = np.nan

    try:
        window = int(0.2 * sampling_rate)
        beats = []
        for r in rpeaks:
            start = r - window
            end = r + window
            if start >= 0 and end < len(signal_clean):
                beats.append(signal_clean[start:end])
        if len(beats) > 0:
            beats = np.array(beats)
            r_amp = np.max(beats, axis=1)
            qrs_energy = np.sum(beats**2, axis=1)

            features["r_amp_mean"] = np.mean(r_amp)
            features["r_amp_std"] = np.std(r_amp)
            features["qrs_energy_mean"] = np.mean(qrs_energy)
            features["qrs_energy_std"] = np.std(qrs_energy)
    except Exception:
        pass

    try:
        features["signal_std"] = np.std(signal_clean)
        features["signal_skew"] = pd.Series(signal_clean).skew()
        features["signal_kurtosis"] = pd.Series(signal_clean).kurt()
    except Exception:
        features["signal_std"] = np.nan
        features["signal_skew"] = np.nan
        features["signal_kurtosis"] = np.nan

    try:
        fft = np.fft.rfft(signal_clean)
        power = np.abs(fft) ** 2
        features["fft_energy"] = np.mean(power)
    except Exception:
        features["fft_energy"] = np.nan

    return pd.DataFrame([features])


def _fallback_features(signal):
    return pd.DataFrame(
        [
            {
                "rr_mean": np.nan,
                "rr_std": np.nan,
                "rmssd": np.nan,
                "r_amp_mean": np.nan,
                "r_amp_std": np.nan,
                "qrs_energy_mean": np.nan,
                "qrs_energy_std": np.nan,
                "signal_std": np.nan,
                "signal_skew": np.nan,
                "signal_kurtosis": np.nan,
                "fft_energy": np.nan,
            }
        ]
    )


def extract_features_batch(
    file_path, n_jobs=-1, tabular_features_path=None, metadata_path=None, split=None
):
    """
    Extract ECG features from waveform data and merge with tabular features
    and targets while maintaining strict sequence alignment across datasets.
    """
    raw_data = np.load(file_path, mmap_mode="r")
    ecg_data = raw_data.squeeze() if len(raw_data.shape) == 4 else raw_data
    num_samples = ecg_data.shape[0]

    results = Parallel(n_jobs=n_jobs)(
        delayed(_extract_and_process_single_sample)(ecg_data, idx)
        for idx in tqdm(range(num_samples), desc="Extracting features")
    )

    tabular_features = None
    tabular_col_names = None
    if tabular_features_path is not None:
        tabular_features = np.load(tabular_features_path)
        if len(tabular_features) != num_samples:
            warnings.warn(
                f"Tabular features count ({len(tabular_features)}) "
                f"doesn't match ECG data ({num_samples}). Skipping merge."
            )
            tabular_features = None
        else:
            tabular_col_names = COLUMNS

    metadata_df = None
    target_col_names = None
    if metadata_path is not None:
        full_metadata = pd.read_csv(metadata_path, index_col=0)
        if split is not None:
            metadata_df = full_metadata[full_metadata["split"] == split].sort_index()
        else:
            metadata_df = full_metadata.sort_index()

        if len(metadata_df) != num_samples:
            warnings.warn(
                f"Metadata count ({len(metadata_df)}) "
                f"doesn't match ECG data ({num_samples}). Skipping targets."
            )
            metadata_df = None
        else:
            target_col_names = [
                col for col in metadata_df.columns if "flag" in col.lower()
            ]

    columns = None
    for r, _ in results:
        if r is not None:
            columns = [c for c in r.columns if c not in ["sample_idx", "error"]]
            break

    if columns is None:
        raise ValueError("No valid ECG samples processed.")

    corrupted_samples = 0
    valid_samples = 0
    failed_samples = 0
    final_results = []

    for idx, (r, error) in enumerate(results):
        if error == "all_leads_corrupted":
            corrupted_samples += 1
        elif error is not None:
            failed_samples += 1
        else:
            valid_samples += 1

        if error is not None or r is None:
            r = pd.DataFrame([[np.nan] * len(columns)], columns=columns)
        else:
            r = r.drop(columns=["sample_idx", "error"], errors="ignore")

        if metadata_df is not None:
            metadata_row = metadata_df.iloc[idx : idx + 1]
            r = pd.concat(
                [
                    r.reset_index(drop=True),
                    metadata_row[target_col_names].reset_index(drop=True),
                ],
                axis=1,
            )

        if tabular_features is not None:
            tabular_row = pd.DataFrame(
                [tabular_features[idx]], columns=tabular_col_names
            )
            r = pd.concat(
                [r.reset_index(drop=True), tabular_row.reset_index(drop=True)], axis=1
            )

        final_results.append(r)

    print("\n========== ECG DATA QUALITY REPORT ==========")
    print(f"Total processed files : {num_samples}")
    print(f"Valid waveforms       : {valid_samples}")
    print(f"Flatlined (NaN filled): {corrupted_samples}")
    print(f"Internal calculation errors : {failed_samples}")
    print("============================================\n")

    result_df = pd.concat(final_results, ignore_index=True)
    return result_df


def standardize_features(
    features_df,
    fit=True,
    scaler_path="../data/feature_scaler.joblib",
    exclude_cols=None,
):
    """
    Standardize extracted ECG features while safeguarding categorical ids,
    pre-scaled tabular statistics, and classification target flag columns.
    """
    numeric_df = features_df.select_dtypes(include=[np.number])

    if exclude_cols is None:
        exclude_cols = []
    else:
        exclude_cols = list(exclude_cols)

    flag_columns = [col for col in numeric_df.columns if "flag" in col.lower()]
    for flag_col in flag_columns:
        if flag_col not in exclude_cols:
            exclude_cols.append(flag_col)

    cols_to_scale = [c for c in numeric_df.columns if c not in exclude_cols]
    cols_to_preserve = [c for c in numeric_df.columns if c in exclude_cols]

    features_to_scale = numeric_df[cols_to_scale]
    preserved_features = numeric_df[cols_to_preserve]

    if fit:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features_to_scale)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        scaled = scaler.transform(features_to_scale)

    scaled_df = pd.DataFrame(scaled, columns=cols_to_scale)
    result = pd.concat([scaled_df, preserved_features], axis=1)

    return result[numeric_df.columns]
