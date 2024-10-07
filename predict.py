import pandas as pd
import joblib
import argparse

parser = argparse.ArgumentParser(description="Predict network attacks using a trained model.")
parser.add_argument('-v', '--verbose', action='count', default=0,
                    help='Increase output verbosity. Use multiple times for more verbosity.')
parser.add_argument('-t','--test_data', type=str,
                    help='Path to the test data CSV file.',default="test_random.csv")   
parser.add_argument("-r",'--result', type=str,
                    help='Path to save the output results CSV file.',default="result.csv")
parser.add_argument("-m",'--model', type=str,
                    help='Model to use',default="model_perceptron.pkl")
args = parser.parse_args()

# Load the new data
new_data = pd.read_csv(args.test_data)

# Load the preprocessor and model
preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load(args.model)
label_encoder = joblib.load('label_encoder.pkl')

# Drop the original 'Attack_type' if it exists
if 'Attack_type' in new_data.columns:
    if args.verbose > 2:
        print("-"*80+"\n#Expected Labels:\n")
        print(new_data['Attack_type'])

    original_attack_type = new_data['Attack_type']
    new_data = new_data.drop(['Attack_type'], axis=1)
    print("FYI - Attack_type labels removed")

# Apply the preprocessor to the new data
X_new_processed = preprocessor.transform(new_data)

# do the predictions
predictions_numeric = model.predict(X_new_processed)

# Decode binary labels to original labels
predicted_labels = label_encoder.inverse_transform(predictions_numeric)

# Create a DataFrame for the results
new_data['Predicted_Binary_Label'] = predicted_labels
new_data['Predicted_Binary'] = predictions_numeric

print("-"*80+"\n# Prediction stats\n")
if args.verbose > 1:
    print(new_data[['Predicted_Binary_Label', 'Predicted_Binary']])

print(new_data.groupby('Predicted_Binary_Label')['Predicted_Binary_Label'].count().to_string())

print(f"\nResults are also saved to {args.result}")
new_data.to_csv(args.result, index=False)