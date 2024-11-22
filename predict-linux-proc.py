import pandas as pd
import joblib
import argparse
from sklearn.metrics import accuracy_score
from datetime import datetime
from rich import print
from rich.console import Console

parser = argparse.ArgumentParser(description="Predict network attacks using a trained model.")
parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase output verbosity. Use multiple times for more verbosity.")
parser.add_argument("-d", "--data", type=str, help="Path to the test data CSV file.", default="datasets/ToN/Linux_process_1.csv")
parser.add_argument("-r", "--result", type=str, help="Path to save the output results CSV file.", default="result.csv")
parser.add_argument("-m", "--model", type=str, help="Model to use", default="model_xgboost-linux-proc.pkl")
parser.add_argument("-p", "--preprocessor", type=str, help="Preprocessor to use", default="preprocessor-linux-proc.pkl")
parser.add_argument("-e", "--encoder", type=str, help="Encoder to use", default="label_encoder-linux-proc.pkl")
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

    #nothing else to clean!
        
if args.verbose > 2:
    console.print("Data Types")
    console.print(df.dtypes.to_string(index=True))

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

    console.print("\nData Types:")
    console.print(df.dtypes.to_string(index=True))

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

