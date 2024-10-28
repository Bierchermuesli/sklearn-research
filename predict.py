import pandas as pd
import joblib
import argparse
from sklearn.metrics import accuracy_score
from datetime import datetime
from rich import print
from rich.console import Console

parser = argparse.ArgumentParser(description="Predict network attacks using a trained model.")
parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase output verbosity. Use multiple times for more verbosity.")
parser.add_argument("-d", "--data", type=str, help="Path to the test data CSV file.", default="test_random.csv")
parser.add_argument("-r", "--result", type=str, help="Path to save the output results CSV file.", default="result.csv")
parser.add_argument("-a", "--all-features", action="store_true", default=False, help="Use all featrues which are in the dataset...")
parser.add_argument("-m", "--model", type=str, help="Model to use", default="model_perceptron.pkl")
parser.add_argument("-p", "--preprocessor", type=str, help="Preprocessor to use", default="preprocessor.pkl")
parser.add_argument("-e", "--encoder", type=str, help="Encoder to use", default="label_encoder_44.pkl")
args = parser.parse_args()


console = Console()

# Load the new data
df = pd.read_csv(args.data,low_memory=False)

# Load the preprocessor and model
preprocessor = joblib.load(args.preprocessor)
model = joblib.load(args.model)
label_encoder = joblib.load(args.encoder)


# Do some per-dataset cleaning
if {"label", "type"}.issubset(df.columns):
    # TON_IOT

    df.rename(columns={"label": "LABEL_BOOL", "type": "LABEL"}, inplace=True)

    if not args.all_features:
        # we only care about  Connection and Statistical activity + labels
        df.drop(df.filter(regex="dns_*").columns, axis=1, inplace=True)
        df.drop(df.filter(regex="http_*").columns, axis=1, inplace=True)
        df.drop(df.filter(regex="ssl_*").columns, axis=1, inplace=True)
        df.drop(df.filter(regex="weird_*").columns, axis=1, inplace=True)

    console.print("Fix dataypes and normalize values")
    # Specifing the dtype on pd.read_csv but we dont know the type at that time
    col_int = ["src_bytes", "dst_bytes", "dst_port", "src_port", "missed_bytes", "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes"]
    col_float = ["duration", "dst_bytes"]
    col_str = ["src_ip", "dst_ip","proto","service","conn_state","LABEL"]
    col_bool = ['LABEL_BOOL']  # Replace with your boolean column names

    for col in col_int:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in col_float:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0) 
    for col in col_str:
        df[col] = df[col].fillna("").astype(str)       
    for col in col_bool:
        df[col] = df[col].astype(bool)


elif {"Attack_type"}.issubset(df.columns):
    # RT_IOT2022
    df.rename(columns={"Attack_type": "LABEL"}, inplace=True)

    # drop unneeded features
    if not args.all_features:
        df.drop(["no"], inplace=True)

if args.verbose > 2:
    console.print("Data Types")
    console.print(df.dtypes.to_string(index=False))

if args.verbose > 1:
    console.print("\nLabled Normal/Evil Ratio:")
    ratio = df['LABEL_BOOL'].value_counts().reset_index()
    ratio.columns = ['LABEL_BOOL', 'Count']
    ratio['Percentage'] = (ratio['Count'] / ratio['Count'].sum()) * 100
    console.print(ratio.to_string(index=False))
    console.print("\nLabeled Type Ratio:")
    ratio = df['LABEL'].value_counts().reset_index()
    ratio.columns = ['LABEL', 'Count']
    ratio['Percentage'] = (ratio['Count'] / ratio['Count'].sum()) * 100
    console.print(ratio.to_string(index=False))


original_attack_type = df["LABEL"].copy()
df = df.drop(["LABEL", "LABEL_BOOL"], axis=1)
console.line()
console.print("[red]✔[/red]  Labels removed!")

console.print(f"\nRows: {len(df)} loaded")
if args.verbose > 0:
    console.print(f"Labels Encoded: {len(label_encoder.classes_)}")
    if args.verbose > 1:
        console.print(" - " + f"\n - ".join(label_encoder.classes_))


with console.status("[bold green] Predict the data...") as status:
        
    _start = datetime.now()
    # Apply the preprocessor to the new data
    X_new_processed = preprocessor.transform(df)

    # do the predictions
    predictions_numeric = model.predict(X_new_processed)

    # Decode binary labels to original labels
    predicted_labels = label_encoder.inverse_transform(predictions_numeric)

    console.print(f"duration: {round((datetime.now()-_start).total_seconds(), 1)}s")    

# Create a DataFrame for the results
df["PREDICT"] = predicted_labels
df["PREDICT_INT"] = predictions_numeric

# re attach the original label
df["ORG_LABEL"] = original_attack_type

if args.verbose > 1:
    console.print("\nPrediction Summary and Ratio:")
    ratio = df['PREDICT'].value_counts().reset_index()
    ratio.columns = ['PREDICT', 'Count']
    ratio['Percentage'] = (ratio['Count'] / ratio['Count'].sum()) * 100
    console.print(ratio.to_string(index=False))

console.print(f"\nResults saved to {args.result}")
df.to_csv(args.result, index=False)

accuracy = accuracy_score(original_attack_type, df["PREDICT"])
console.print(f"Accuracy of predictions: {accuracy:.4f}")
