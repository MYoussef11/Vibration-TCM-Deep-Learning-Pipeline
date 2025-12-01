"""
Train ML model using only top 20 features.
Creates a lightweight model for real-time streaming.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Top 20 features (from feature importance analysis)
TOP_20_FEATURES = [
    'acceleration_yg_wpd_wpd_energy_lvl3_node0',
    'acceleration_yg_freq_spectral_energy',
    'acceleration_yg_freq_band_power_low',
    'acceleration_yg_time_rms',
    'acceleration_zg_freq_spectral_energy',
    'acceleration_zg_time_rms',
    'acceleration_yg_time_mean',
    'acceleration_zg_freq_band_power_low',
    'acceleration_zg_wpd_wpd_energy_lvl3_node0',
    'acceleration_zg_time_crest_factor',
    'acceleration_zg_time_mean',
    'angular_velocity_ydps_time_mad',
    'acceleration_zg_time_skewness',
    'acceleration_zg_time_std',
    'angular_velocity_zdps_wpd_wpd_energy_lvl3_node4',
    'angular_velocity_ydps_time_rms',
    'angular_velocity_ydps_time_std',
    'acceleration_zg_time_mad',
    'acceleration_zg_wpd_wpd_energy_lvl3_node1',
    'angular_velocity_ydps_wpd_wpd_energy_lvl3_node4'
]


def load_and_prepare_data(features_csv, binary=True):
    """Load and prepare dataset."""
    df = pd.read_csv(features_csv)
    
    # Convert to binary if needed
    if binary:
        # For binary: Good vs Faulty
        # Use existing label_id if present, otherwise map from label
        if 'label_id' in df.columns:
            # Already has label_id, just use it
            pass
        else:
            # Create label_id from label column
            # Assuming: 0 = Good, 1 = Faulty
            df['label_id'] = (df['label'] != df['label'].iloc[0]).astype(int)
    
    # Drop rows with NaN in label_id
    df = df.dropna(subset=['label_id'])
    
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    
    # Load data
    print("Loading dataset...")
    df = load_and_prepare_data(args.features, binary=True)
    
    # Select only top 20 features
    print(f"\nUsing top 20 features (out of {len(df.columns)} total)")
    X = df[TOP_20_FEATURES]
    y = df['label_id'].to_numpy()
    groups = df['file'].astype('category').cat.codes.to_numpy()
    
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Classes: {np.unique(y, return_counts=True)}")
    
    # Create model (same hyperparameters as best full model)
    print("\nTraining Random Forest on top 20 features...")
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=389,
            max_depth=30,
            max_features='log2',
            min_samples_split=5,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=args.random_state
        ))
    ])
    
    # Cross-validation
    cv = GroupKFold(n_splits=5)
    predictions = cross_val_predict(model, X, y, cv=cv, groups=groups, n_jobs=-1)
    
    # Evaluate
    report = classification_report(y, predictions, output_dict=True)
    accuracy = report['accuracy']
    
    print(f"\n✅ Cross-validation accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y, predictions, target_names=['Good', 'Faulty']))
    
    # Train final model on all data
    print("\nTraining final model on full dataset...")
    model.fit(X, y)
    
    # Save everything
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = args.output_dir / "rf_top20_model.pkl"
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved: {model_path}")
    
    # Save feature names
    feature_file = args.output_dir / "top_20_feature_names.json"
    with open(feature_file, 'w') as f:
        json.dump({"features": TOP_20_FEATURES}, f, indent=2)
    print(f"✅ Features saved: {feature_file}")
    
    # Save metadata
    metadata = {
        "model_type": "RandomForest",
        "n_features": len(TOP_20_FEATURES),
        "accuracy": float(accuracy),
        "binary_classification": True,
        "feature_subset": "top_20",
        "training_date": str(pd.Timestamp.now())
    }
    metadata_file = args.output_dir / "top20_model_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved: {metadata_file}")
    
    print(f"\n🎉 Done! Model ready for streaming inference.")
    print(f"   Accuracy: {accuracy:.1%}")
    print(f"   Features: {len(TOP_20_FEATURES)}")


if __name__ == "__main__":
    main()
