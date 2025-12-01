"""
Extract top features from binary ML model.
"""
import joblib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load model
model_path = PROJECT_ROOT / "reports" / "phase3" / "ml_binary" / "rf_best_estimator.pkl"
model = joblib.load(model_path)

# Load feature names
with open(PROJECT_ROOT / "reports" / "phase3" / "ml_binary" / "feature_names.json") as f:
    feature_names = json.load(f)['features']

# Get feature importances
importances = model.named_steps['clf'].feature_importances_

# Create sorted list
feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

# Print top 20
print("Top 20 Features by Importance:\n")
print(f"{'Rank':<6} {'Feature':<55} {'Importance':<12}")
print("="*75)
for i, (feat, imp) in enumerate(feature_importance[:20], 1):
    print(f"{i:<6} {feat:<55} {imp:<12.6f}")

# Save to file
output = PROJECT_ROOT / "reports" / "phase3" / "ml_binary" / "top_20_features.txt"
with open(output, 'w') as f:
    f.write("Top 20 Features for Binary ML Model\n")
    f.write("="*75 + "\n\n")
    for i, (feat, imp) in enumerate(feature_importance[:20], 1):
        f.write(f"{i}. {feat}: {imp:.6f}\n")

print(f"\nSaved to: {output}")
