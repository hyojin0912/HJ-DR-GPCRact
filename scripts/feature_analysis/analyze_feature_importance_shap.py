#!/usr/bin/env python
# coding: utf-8

"""
analyze_feature_importance_shap.py: Perform SHAP analysis to evaluate feature importance
for GPCR drug activity prediction across Dual-state DR, Single-state DR, and Predicted DR scenarios.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tensorflow.keras.models import load_model
from sklearn.metrics import balanced_accuracy_score
import ast
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
np.random.seed(2)
tf.random.set_seed(2)

# GPU configuration
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    logger.info("GPU configured with memory growth enabled.")

# Constants
SAMPLE_SIZE = 1000  # Number of samples for SHAP computation
DISP_THRESHOLD_AGO = 2.0  # Å threshold for DR-Ago
DISP_THRESHOLD_ANT = 2.0  # Å threshold for DR-Ant
DISP_THRESHOLD_STATE = 0.5  # Å threshold for DR-State
OUTPUT_DIR = "../Output/Final/SHAP_Analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Feature names
ECFP_NAMES = [f"ECFP4_Bit{i}" for i in range(1024)]
DR_NAMES = (
    [f"ESM1b_{i}" for i in range(1280)] +
    ["DR centroid x", "DR centroid y", "DR centroid z", "Mean(DR dist)", "SD(DR dist)", "Number of DRs"]
)
FEATURE_NAMES = ECFP_NAMES + DR_NAMES  # 1024 + 1286 = 2310 features

# Scenario configurations
SCENARIOS = {
    "Dual-state DR": {
        "train_csv": "../Final/train.csv",
        "test_csv": "../Final/test.csv",
        "dr_pkl": f"../Final_Data/Feature/Final_gpcr_cand_DR1286_{DISP_THRESHOLD_AGO}_{DISP_THRESHOLD_ANT}_{DISP_THRESHOLD_STATE}.pkl",
        "model_path": "../Output_Models/Final/AF2/model"
    },
    "Single-state DR": {
        "train_csv": "../Final/1side_train.csv",
        "test_csv": "../Final/1side_test.csv",
        "dr_pkl": f"../Final_Data/Feature/Final_1side_gpcr_cand_DR1286_{DISP_THRESHOLD_AGO}_{DISP_THRESHOLD_ANT}_{DISP_THRESHOLD_STATE}.pkl",
        "model_path": "../Output_Models/Final/1side/model"
    },
    "Predicted DR": {
        "train_csv": "../Final/DR_pred_train.csv",
        "test_csv": "../Final/DR_pred_test.csv",
        "dr_pkl": "../Final_Data/Feature/DR_pred/DR_pred_gpcr_DR1286.pkl",
        "model_path": "../Output_Models/EGNN_DR/Predicted/model_seed0"
    }
}
ECFP_PKL = "../Final_Data/Feature/Final_gpcr_cand_Cmp_ECFP4.pkl"

def encode_label(label_str: str) -> int:
    """Encode labels: agonist -> 1, antagonist -> 0."""
    return 1 if label_str.lower() == "agonist" else 0 if label_str.lower() == "antagonist" else -1

def to_float_list(x):
    """Convert string or list to float list."""
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return ast.literal_eval(x)
    raise ValueError(f"Invalid input type: {type(x)}")

def load_and_merge_data(train_csv: str, test_csv: str, ecfp_pkl: str, dr_pkl: str) -> tuple:
    """Load and merge ECFP4 and DR1286 features with train/test data."""
    logger.info(f"Loading data: train={train_csv}, test={test_csv}")
    
    # Load data
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    ecfp_df = pd.read_pickle(ecfp_pkl)
    dr_df = pd.read_pickle(dr_pkl)

    # Convert string features to lists
    ecfp_df["ECFP4"] = ecfp_df["ECFP4"].apply(to_float_list)
    dr_df["DR1286"] = dr_df["DR1286"].apply(to_float_list)

    # Merge features
    train_df = train_df.merge(ecfp_df, how="left", on="Ikey").merge(dr_df, how="left", on="AC")
    test_df = test_df.merge(ecfp_df, how="left", on="Ikey").merge(dr_df, how="left", on="AC")

    # Convert lists to numpy arrays
    train_df["ECFP4"] = train_df["ECFP4"].apply(lambda x: np.asarray(x, dtype=np.float32))
    train_df["DR1286"] = train_df["DR1286"].apply(lambda x: np.asarray(x, dtype=np.float32))
    test_df["ECFP4"] = test_df["ECFP4"].apply(lambda x: np.asarray(x, dtype=np.float32))
    test_df["DR1286"] = test_df["DR1286"].apply(lambda x: np.asarray(x, dtype=np.float32))

    # Encode labels
    train_df["Label"] = train_df["Label"].apply(encode_label)
    test_df["Label"] = test_df["Label"].apply(encode_label)

    # Validate data
    assert train_df["ECFP4"].isna().sum() == 0 and train_df["DR1286"].isna().sum() == 0, "Missing features in train"
    assert test_df["ECFP4"].isna().sum() == 0 and test_df["DR1286"].isna().sum() == 0, "Missing features in test"
    assert not train_df["Label"].eq(-1).any() and not test_df["Label"].eq(-1).any(), "Invalid labels detected"

    # Extract features and labels
    X_ecfp4_train = np.stack(train_df["ECFP4"].values)
    X_dr_train = np.stack(train_df["DR1286"].values)
    y_train = train_df["Label"].values.astype(np.float32)

    X_ecfp4_test = np.stack(test_df["ECFP4"].values)
    X_dr_test = np.stack(test_df["DR1286"].values)
    y_test = test_df["Label"].values.astype(np.float32)

    return X_ecfp4_train, X_dr_train, y_train, X_ecfp4_test, X_dr_test, y_test

def compute_shap_values(model, X_ecfp4_train: np.ndarray, X_dr_train: np.ndarray,
                        X_ecfp4_test: np.ndarray, X_dr_test: np.ndarray, sample_size: int) -> np.ndarray:
    """Compute SHAP values using DeepExplainer."""
    logger.info(f"Computing SHAP values with sample_size={sample_size}")

    # Sample background and test data
    X_bg = shap.sample(np.hstack([X_ecfp4_train, X_dr_train]), sample_size)
    X_test = np.hstack([X_ecfp4_test[:sample_size], X_dr_test[:sample_size]])

    # Create explainer
    explainer = shap.DeepExplainer(model, [X_ecfp4_train[:sample_size], X_dr_train[:sample_size]])

    # Compute SHAP values
    shap_values = explainer.shap_values([X_ecfp4_test[:sample_size], X_dr_test[:sample_size]])

    # Flatten SHAP values
    shap_values_single = shap_values[0] if isinstance(shap_values[0], list) else shap_values
    shap_vals = np.hstack(shap_values_single) if isinstance(shap_values_single, list) else shap_values_single

    logger.info(f"SHAP values computed, shape: {shap_vals.shape}")
    return shap_vals, X_test

def plot_shap_summary(shap_vals: np.ndarray, X_test: np.ndarray, scenario: str):
    """Generate SHAP summary beeswarm plot (Figure 2a for Predicted DR)."""
    if scenario != "Predicted DR":
        return  # Only generate for Predicted DR

    logger.info("Generating SHAP summary plot (Figure 2a)")

    plt.figure(figsize=(10/2.54, 12/2.54))  # 10 cm x 12 cm
    shap.summary_plot(
        shap_vals,
        X_test,
        feature_names=FEATURE_NAMES,
        max_display=10,
        show=False,
        color_bar=False
    )

    fig = plt.gcf()
    fig.subplots_adjust(left=0.30, right=0.97, top=0.90, bottom=0.15)
    fig.suptitle('SHAP Summary Plot (Predicted DR)', fontsize=14, y=0.96)

    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_xlabel('SHAP Value (Impact on Model Output)', fontsize=12)

    output_path = os.path.join(OUTPUT_DIR, "Figure_2a_SHAP_summary.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Figure 2a: {output_path}")

def plot_shap_bar(shap_dict: dict):
    """Generate grouped bar plot for top DR features (Figure 2b)."""
    logger.info("Generating SHAP grouped bar plot (Figure 2b)")

    # Top 6 DR features (last 6 indices)
    top_indices = [-6, -5, -4, -3, -2, -1]  # Centroid x,y,z, Mean dist, SD dist, Number of DRs
    mean_shap = {name: np.abs(vals[:, top_indices]).mean(axis=0) for name, vals in shap_dict.items()}
    sem_shap = {name: np.abs(vals[:, top_indices]).std(axis=0) / np.sqrt(SAMPLE_SIZE)
                for name, vals in shap_dict.items()}

    # Plot
    fig, ax = plt.subplots(figsize=(15/2.54, 10/2.54))  # 15 cm x 10 cm
    labels = ['DR centroid x', 'DR centroid y', 'DR centroid z', 'Mean(DR dist)', 'SD(DR dist)', 'Number of DRs']
    x = np.arange(len(labels))
    width = 0.25

    colors = {'Dual-state DR': '#4a90e2', 'Single-state DR': '#66c2a5', 'Predicted DR': '#ff8c61'}
    for i, name in enumerate(['Dual-state DR', 'Single-state DR', 'Predicted DR']):
        ax.bar(x + i * width, mean_shap[name], width, yerr=sem_shap[name], capsize=3,
               label=name, color=colors[name], alpha=1)

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=45, fontsize=10, ha='right')
    ax.set_ylabel('Mean |SHAP| Value', fontsize=12)
    ax.set_xlabel('DR Feature', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "Figure_2b_SHAP_bar.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Figure 2b: {output_path}")

    # Save mean SHAP values
    mean_shap_df = pd.DataFrame(mean_shap, index=labels)
    mean_shap_df.to_csv(os.path.join(OUTPUT_DIR, "mean_shap_values.csv"))
    logger.info("Saved mean SHAP values CSV")

def plot_shap_waterfall(shap_vals: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, scenario: str):
    """Generate SHAP waterfall plots for high-confidence samples (Figure 2c for Predicted DR)."""
    if scenario != "Predicted DR":
        return  # Only generate for Predicted DR

    logger.info("Generating SHAP waterfall plots (Figure 2c)")

    def plot_force(label: int, feature: str, rank: int = 1, figsize: tuple = (10/2.54, 6/2.54)):
        """Plot SHAP force plot for the rank-th sample of class label by |SHAP| on feature."""
        feat_idx = len(ECFP_NAMES) + DR_NAMES.index(feature)
        idxs = np.where(y_test == label)[0]
        if len(idxs) == 0:
            logger.warning(f"No samples for label {label}")
            return

        abs_vals = np.abs(shap_vals[idxs, feat_idx])
        best_idx = idxs[np.argsort(abs_vals)[-rank]]

        sample_vals = shap_vals[best_idx]
        top4_idxs = np.argsort(np.abs(sample_vals))[-4:]
        vals4 = sample_vals[top4_idxs]
        data4 = X_test[best_idx, top4_idxs]
        names4 = [FEATURE_NAMES[i] for i in top4_idxs]
        base_val = float(explainer.expected_value[0])

        expl = shap.Explanation(
            values=vals4.reshape(1, -1),
            base_values=np.array([base_val]),
            data=data4.reshape(1, -1),
            feature_names=names4
        )

        plt.figure(figsize=figsize)
        shap.plots.force(expl, matplotlib=True, show=False)
        ax = plt.gca()

        for txt in ax.texts:
            s = txt.get_text()
            if 'base value' in s:
                txt.set_color('gray')
                txt.set_fontsize(6)
                txt.set_fontweight('bold')
                txt.set_position((base_val, 0.02))
            elif 'higher' in s or 'lower' in s:
                txt.set_fontsize(12)
            elif '=' in s:
                try:
                    name, value = s.split(' = ')
                    value_float = float(value)
                    formatted = f"{value_float:.2f}" if abs(value_float) >= 0.1 else f"{value_float:.1f}"
                    txt.set_text(f"{name}\n= {formatted}")
                    txt.set_fontsize(10)
                    txt.set_linespacing(1.2)
                except ValueError:
                    pass
            elif 'f(x)' in s:
                txt.set_visible(False)

        ax.tick_params(axis='x', labelsize=6)
        plt.tight_layout(pad=1.5)

        label_str = "agonist" if label == 1 else "antagonist"
        output_path = os.path.join(OUTPUT_DIR, f"Figure_2c_force_{feature.replace(' ', '_')}_{label_str}_rank{rank}.png")
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved Figure 2c: {output_path}")

    # Generate plots for agonist and antagonist
    plot_force(label=1, feature="DR centroid y", rank=2)  # Agonist, top-2
    plot_force(label=0, feature="DR centroid y", rank=1)  # Antagonist, top-1

def main():
    """Main function to perform SHAP analysis and generate figures."""
    shap_dict = {}
    test_data_dict = {}
    y_test_dict = {}

    for scenario, config in SCENARIOS.items():
        logger.info(f"Processing scenario: {scenario}")

        # Load data
        X_ecfp4_train, X_dr_train, y_train, X_ecfp4_test, X_dr_test, y_test = load_and_merge_data(
            config["train_csv"], config["test_csv"], ECFP_PKL, config["dr_pkl"]
        )

        # Load model
        try:
            model = load_model(config["model_path"])
            logger.info(f"Loaded model for {scenario}")
        except Exception as e:
            logger.error(f"Failed to load model for {scenario}: {e}")
            continue

        # Compute SHAP values
        shap_vals, X_test = compute_shap_values(
            model, X_ecfp4_train, X_dr_train, X_ecfp4_test, X_dr_test, SAMPLE_SIZE
        )

        shap_dict[scenario] = shap_vals
        test_data_dict[scenario] = X_test
        y_test_dict[scenario] = y_test[:SAMPLE_SIZE]

        # Save SHAP values
        shap_df = pd.DataFrame(shap_vals, columns=FEATURE_NAMES)
        shap_df.to_csv(os.path.join(OUTPUT_DIR, f"shap_values_{scenario.replace(' ', '_')}.csv"), index=False)
        logger.info(f"Saved SHAP values for {scenario}")

    # Generate plots
    plot_shap_summary(shap_dict["Predicted DR"], test_data_dict["Predicted DR"], "Predicted DR")
    plot_shap_bar(shap_dict)
    plot_shap_waterfall(shap_dict["Predicted DR"], test_data_dict["Predicted DR"], y_test_dict["Predicted DR"], "Predicted DR")

    logger.info("SHAP analysis completed.")

if __name__ == "__main__":
    main()