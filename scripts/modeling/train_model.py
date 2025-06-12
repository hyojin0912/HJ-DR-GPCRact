# scripts/modeling/train_model.py

import os
import argparse
import numpy as np
import pandas as pd
import ast
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def encode_label(label_str):
    """Maps 'agonist' to 1 and 'antagonist' to 0."""
    if label_str == 'agonist':
        return 1
    elif label_str == 'antagonist':
        return 0
    return -1

def to_float_list(x):
    """Converts string representation of a list to a list."""
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return ast.literal_eval(x)
    raise ValueError(f"Cannot convert type {type(x)} to list.")

def load_and_prepare_data(args):
    """Loads datasets, merges features, and prepares them for training."""
    print("[INFO] Loading and preparing data...")
    
    # Define DR feature path based on thresholds
    dr_feature_path = os.path.join(args.feature_dir, f"Final_gpcr_cand_DR1286_{args.thr_ago}_{args.thr_ant}_{args.thr_state}.pkl")
    ecfp_feature_path = os.path.join(args.feature_dir, "Final_gpcr_cand_Cmp_ECFP4.pkl")
    
    # Load data
    train_df = pd.read_csv(os.path.join(args.data_dir, "train_set_pair.csv"))
    test_df = pd.read_csv(os.path.join(args.data_dir, "test_set_pair.csv"))
    ecfp_df = pd.read_pickle(ecfp_feature_path)
    dr_df = pd.read_pickle(dr_feature_path)

    # Filter to proteins with DR features
    valid_ac_list = dr_df['AC'].unique()
    train_df = train_df[train_df['AC'].isin(valid_ac_list)].reset_index(drop=True)
    test_df = test_df[test_df['AC'].isin(valid_ac_list)].reset_index(drop=True)

    # Process features
    ecfp_df["ECFP4"] = ecfp_df["ECFP4"].apply(to_float_list)
    dr_df["DR1286"] = dr_df["DR1286"].apply(to_float_list)

    # Merge features
    def attach_features(df):
        df = df.merge(ecfp_df, how="left", on="Ikey").merge(dr_df, how="left", on="AC")
        df["ECFP4"] = df["ECFP4"].apply(lambda x: np.asarray(x, dtype=np.float32))
        df["DR1286"] = df["DR1286"].apply(lambda x: np.asarray(x, dtype=np.float32))
        return df

    train_df = attach_features(train_df)
    test_df = attach_features(test_df)

    # Encode labels
    train_df["Label"] = train_df["Label"].apply(encode_label)
    test_df["Label"] = test_df["Label"].apply(encode_label)

    # Sanity checks
    assert train_df["ECFP4"].isna().sum() == 0 and train_df["DR1286"].isna().sum() == 0, "Missing features in train set"
    assert test_df["ECFP4"].isna().sum() == 0 and test_df["DR1286"].isna().sum() == 0, "Missing features in test set"

    # Prepare numpy arrays
    X_ecfp4_train = np.stack(train_df["ECFP4"].values)
    X_dr_train = np.stack(train_df["DR1286"].values)
    y_train_all = train_df["Label"].values.astype(np.float32)

    X_ecfp4_test = np.stack(test_df["ECFP4"].values)
    X_dr_test = np.stack(test_df["DR1286"].values)
    y_test = test_df["Label"].values.astype(np.float32)
    
    # Train/validation split
    tr_idx, val_idx = train_test_split(np.arange(len(y_train_all)), test_size=0.2, random_state=args.seed, stratify=y_train_all)

    data = {
        "X_train": [X_ecfp4_train[tr_idx], X_dr_train[tr_idx]], "y_train": y_train_all[tr_idx],
        "X_val": [X_ecfp4_train[val_idx], X_dr_train[val_idx]], "y_val": y_train_all[val_idx],
        "X_test": [X_ecfp4_test, X_dr_test], "y_test": y_test,
    }
    print("[INFO] Data preparation complete.")
    return data

def build_model(learning_rate=1e-3):
    """Builds the Keras MLP model."""
    print("[INFO] Building Keras MLP model...")
    # Branch 1: ECFP4
    in_ecfp = Input(shape=(1024,), name="ECFP4_in")
    x1 = Dense(512, activation='relu')(in_ecfp)
    x1 = Dense(128, activation='relu')(x1)

    # Branch 2: DR1286
    in_dr = Input(shape=(1286,), name="DR1286_in")
    x2 = Dense(512, activation='relu')(in_dr)
    x2 = Dense(128, activation='relu')(x2)

    # Fusion
    x = Concatenate()([x1, x2])
    x = Dense(128, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[in_ecfp, in_dr], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

def main(args):
    """Main function to run the training pipeline."""
    data = load_and_prepare_data(args)
    model = build_model(args.learning_rate)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    print("[INFO] Starting model training...")
    history = model.fit(
        x=data["X_train"],
        y=data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stopping],
        verbose=2
    )
    
    print("[INFO] Training finished.")

    # Evaluate on the test set
    loss, accuracy = model.evaluate(data["X_test"], data["y_test"], verbose=0)
    print(f"\n[RESULT] Test Set Performance: Loss={loss:.4f}, Accuracy={accuracy:.4f}")

    # Save the model
    os.makedirs(args.output_dir, exist_ok=True)
    model_save_path = os.path.join(args.output_dir, args.model_name)
    model.save(model_save_path)
    print(f"[SUCCESS] Model saved to {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPCR activity prediction model.")
    parser.add_argument("--data_dir", type=str, default="../Final", help="Directory containing train/test set CSVs.")
    parser.add_argument("--feature_dir", type=str, default="../Data/Feature", help="Directory containing ECFP4 and DR1286 .pkl files.")
    parser.add_argument("--output_dir", type=str, default="../Output/Final/AF2/model", help="Directory to save the trained model.")
    parser.add_argument("--model_name", type=str, default="gpcr_activity_model.h5", help="Name for the saved model file.")
    
    # Hyperparameters from your notebook
    parser.add_argument("--thr_ago", type=float, default=2.0, help="Threshold for DR-Ago.")
    parser.add_argument("--thr_ant", type=float, default=2.0, help="Threshold for DR-Ant.")
    parser.add_argument("--thr_state", type=float, default=0.5, help="Threshold for DR-State.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")

    args = parser.parse_args()
    main(args)