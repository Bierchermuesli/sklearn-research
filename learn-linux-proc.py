import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
import joblib
import argparse
from rich.console import Console
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

console = Console()

models = ["perceptron", "randomforest", "ensemble","xgboost"]

parser = argparse.ArgumentParser(description="Predict network attacks using a trained model.")
parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase output verbosity. Use multiple times for more verbosity.")
parser.add_argument("-d", "--data-set", type=str, help="Path to the data set CSV file.", default="trainsets/Train_Test_Linux_process.csv")
parser.add_argument("-m", "--models", choices=models, default=models, nargs="+", help="Models to generate")
args = parser.parse_args()


# Load the dataset
df = pd.read_csv(args.data_set)
console.print(f" Features: {len(df.columns)}")


# do some data normlization. regognize the datasets by its fields...
if {"label", "type"}.issubset(df.columns):
    # TON_IOT
    df.rename(columns={"label": "LABEL_BOOL", "type": "LABEL"}, inplace=True)

    # Drop columns that contain only empty strings or zeros like PTRP
    df = df.loc[:, (df != 0).any(axis=0)]
    df = df.loc[:, (df != "").any(axis=0)]

    #nothing else to clean!

else:
    console.print("Dataset unknown")


if args.verbose > 2:
    console.print("Data Types")
    console.print(df.dtypes.to_string(index=True))

print(args.verbose)
if args.verbose > 1:
    console.print("\nNormal/Evil Ratio:")
    ratio = df['LABEL_BOOL'].value_counts().reset_index()
    ratio.columns = ['LABEL_BOOL', 'Count']
    ratio['Percentage'] = (ratio['Count'] / ratio['Count'].sum()) * 100
    console.print(ratio.to_string(index=False))

    console.print("\nAttack Type Ratio:")
    ratio = df['LABEL'].value_counts().reset_index()
    ratio.columns = ['LABEL', 'Count']
    ratio['Percentage'] = (ratio['Count'] / ratio['Count'].sum()) * 100
    console.print(ratio.to_string(index=False))

    console.print(f"Stats\n Rows: {len(df)}")
    console.print(f" Features: {len(df.columns)}")



X = df.drop(["LABEL", "LABEL_BOOL"], axis=1)
y = df["LABEL"]

# Encode the binary labels
label = LabelEncoder()
y_encoded = label.fit_transform(y)
joblib.dump(label, "label_encoder-linux-proc.pkl")

console.print(f"\nLabels: {len(label.classes_)}")
console.print(f"- " + "\n- ".join(label.classes_))

# Select numerical features for correlation matrix
numerical_columns = X.select_dtypes(exclude=["object"]).columns
correlation_matrix = X[numerical_columns].corr()

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap="BuPu", square=True)
plt.title("Feature Correlation Matrix")
plt.show()

# Apply OneHotEncoder to categorical features and scale numerical features
categorical_columns = X.select_dtypes(include=["object"]).columns
preprocessor = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(), numerical_columns),
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ]
)

# Process the features
X_processed = preprocessor.fit_transform(X)

# Save the preprocessor for future predictions
joblib.dump(preprocessor, "preprocessor-linux-proc.pkl")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.4, random_state=42)

for model_name in args.models:

    if model_name == "perceptron":
        # Train a Perceptron model
        from sklearn.linear_model import Perceptron

        with console.status(f"[bold green]Working on {model_name}...") as status:
            status.update(f"Prepping {model_name}")
            _start = datetime.now()
            model = Perceptron()
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            joblib.dump(model, f"model_{model_name}-linux-proc.pkl")
            console.print(f"✔ {model_name} created. Accuracy: {accuracy:.4f} - {round((datetime.now()-_start).total_seconds(), 1)}s")

    elif model_name == "randomforest":
        # Train a RandomForest model
        from sklearn.ensemble import RandomForestClassifier

        with console.status(f"[bold green]Working on {model_name}...") as status:
            _start = datetime.now()
            model = RandomForestClassifier() # no improove with class_weight='balanced' 
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            joblib.dump(model, f"model_{model_name}-linux-proc.pkl")
            console.print(f"✔ {model_name} created. Accuracy: {accuracy:.4f} - {round((datetime.now()-_start).total_seconds(), 1)}s")

    elif model_name == "ensemble":
        # Train a ensemble model with a VotingClassifier
        from sklearn.linear_model import Perceptron, LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import VotingClassifier, RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier


        # Define individual models with updated max_iter
        models = [
            ("Random Forest", RandomForestClassifier()),
            ("Decision Tree", DecisionTreeClassifier()),
            ("KNN", KNeighborsClassifier()),
            ("Logistic Regression", LogisticRegression(max_iter=300)),  # Updated max_iter
            ("MLP Classifier", MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)),  # Updated max_iter
            ("Perceptron", Perceptron(max_iter=300))  # Updated max_iter
        ]

        ensemble_model = VotingClassifier(estimators=models, voting="hard")
        
        with console.status(f"[bold green]Working on {model_name}...") as status:
            _start = datetime.now()
            # Initialize VotingClassifier with the list of models
            model = VotingClassifier(estimators=models, voting="hard") # no improove with soft 
            # Train the VotingClassifier
            model.fit(X_train, y_train)
            # Make predictions using the VotingClassifier
            y_pred = model.predict(X_test)
            # Evaluate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            joblib.dump(model, f"model_{model_name}-linux-proc.pkl")
            console.print(f"✔ {model_name} created. Accuracy: {accuracy:.4f} - {round((datetime.now()-_start).total_seconds(), 1)}s")

    elif model_name == "xgboost":
        import xgboost as xgb

        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            max_depth=6,
            n_estimators=100,
            learning_rate=0.1
        )
        with console.status(f"[bold green]Working on {model_name}...") as status:
            _start = datetime.now()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            joblib.dump(model, f"model_{model_name}-linux-proc.pkl")
            console.print(f"✔ {model_name} created. Accuracy: {accuracy:.4f} - {round((datetime.now()-_start).total_seconds(), 1)}s")