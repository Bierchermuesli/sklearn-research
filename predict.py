import pandas as pd
import joblib
import argparse
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description="Predict network attacks using a trained model.")
parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase output verbosity. Use multiple times for more verbosity.")
parser.add_argument("-d", "--data", type=str, help="Path to the test data CSV file.", default="test_random.csv")
parser.add_argument("-r", "--result", type=str, help="Path to save the output results CSV file.", default="result.csv")
parser.add_argument("-a", "--all-features", action="store_true", default=False, help="Use all featrues which are in the dataset...")
parser.add_argument("-m", "--model", type=str, help="Model to use", default="model_perceptron.pkl")
parser.add_argument("-p", "--preprocessor", type=str, help="Preprocessor to use", default="preprocessor.pkl")
parser.add_argument("-e", "--encoder", type=str, help="Encoder to use", default="label_encoder_44.pkl")
args = parser.parse_args()

# Load the new data
df = pd.read_csv(args.data)

# Load the preprocessor and model
preprocessor = joblib.load(args.preprocessor)
model = joblib.load(args.model)
label_encoder = joblib.load(args.encoder)

print(f"Rows: {len(df)}")
print("Label Encoder Classes we have:")
print(f"\n- ".join(label_encoder.classes_))

# Drop the original labels if they exist
if {"label", "type"}.issubset(df.columns):
    # TON_IOT
    if args.verbose > 2:
        print("-" * 80 + "\n#Expected Labels:\n")
        print(df["type"])

    original_attack_type = df["type"].copy()
    df = df.drop(["label", "type"], axis=1)

    print("Attack_type labels removed")

    if not args.all_features:
        # we only care about  Connection and Statistical activity + labels
        df.drop(df.filter(regex="dns_*").columns, axis=1, inplace=True)
        df.drop(df.filter(regex="http_*").columns, axis=1, inplace=True)
        df.drop(df.filter(regex="ssl_*").columns, axis=1, inplace=True)
        df.drop(df.filter(regex="weird_*").columns, axis=1, inplace=True)

    print("fix and normalize values")
    int_rows = ["src_bytes", "dst_bytes", "dst_port", "src_port", "missed_bytes", "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes"]
    float_rows = ["duration", "dst_bytes"]
    for col in int_rows:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in float_rows:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)  # Use 0.0 for floats

elif {"Attack_type"}.issubset(df.columns):
    # RT_IOT2022
    if args.verbose > 2:
        print("-" * 80 + "\n#Expected Labels:\n")
        print(df["Attack_type"])

    original_attack_type = df["Attack_type"].copy()
    df = df.drop(["Attack_type"], axis=1)
    print("FYI - Attack_type labels removed")

    # drop unneeded features
    if not args.all_features:
        df.drop(["no"], inplace=True)

# Apply the preprocessor to the new data
X_new_processed = preprocessor.transform(df)

# do the predictions
predictions_numeric = model.predict(X_new_processed)

# Decode binary labels to original labels
predicted_labels = label_encoder.inverse_transform(predictions_numeric)


# Create a DataFrame for the results
df["PREDICT"] = predicted_labels
df["PREDICT_INT"] = predictions_numeric

# re attach the original label
df["ORG_LABEL"] = original_attack_type

print("-" * 80 + "\n# Prediction stats\n")
if args.verbose > 1:
    print(df[["PREDICT_INT", "PREDICT"]])

print(df.groupby("PREDICT")["PREDICT"].count().to_string())

print(f"\nResults saved to {args.result}")
df.to_csv(args.result, index=False)

accuracy = accuracy_score(original_attack_type, df["PREDICT"])
print(f"Accuracy of predictions: {accuracy:.4f}")
