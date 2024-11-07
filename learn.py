import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
import joblib
import argparse
from rich.console import Console
from datetime import datetime

console = Console()

models = ["perceptron", "randomforest", "ensemble","xgboost"]

parser = argparse.ArgumentParser(description="Predict network attacks using a trained model.")
parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase output verbosity. Use multiple times for more verbosity.")
parser.add_argument("-d", "--data-set", type=str, help="Path to the data set CSV file.", default="dataset/Ton_IoT_train_test_network.csv")
parser.add_argument("-a", "--all-features", action="store_true", default=False, help="Use all featrues which are in the dataset...")
parser.add_argument("-m", "--models", choices=models, default=models, nargs="+", help="Models to generate")
args = parser.parse_args()



# Load the dataset
df = pd.read_csv(args.data_set)

# do some data normlization. regognize the datasets by its fields...
if {"label", "type"}.issubset(df.columns):
    # TON_IOT
    df.rename(columns={"label": "LABEL_BOOL", "type": "LABEL"}, inplace=True)

    # categorical columns
    categorical_columns = ["conn_state", "proto"]

    console.print("Fix dataypes and normalize values")
    # Specifing the dtype on pd.read_csv but we dont know the type at that time
    if not args.all_features:
        # we only care about  Connection and Statistical activity + labels
        df.drop(df.filter(regex="dns_*").columns, axis=1, inplace=True)
        df.drop(df.filter(regex="http_*").columns, axis=1, inplace=True)
        df.drop(df.filter(regex="ssl_*").columns, axis=1, inplace=True)
        df.drop(df.filter(regex="weird_*").columns, axis=1, inplace=True)

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
    # RT_IOT
    df.rename(columns={"Attack_type": "LABEL"}, inplace=True)

    # Relabel for binary classification and keep original attack types
    normal_list = ["MQTT_Publish", "Thing_Speak", "Wipro_bulb", "Amazon-Alexa", "TV"]
    attack_list = ["DOS_SYN_Hping", "ARP_poisioning", "NMAP_UDP_SCAN", "NMAP_XMAS_TREE_SCAN", "NMAP_OS_DETECTION", "NMAP_TCP_scan", "DDOS_Slowloris", "Metasploit_Brute_Force_SSH", "NMAP_FIN_SCAN"]
    df["LABEL_BOOL"] = df["LABEL"].apply(lambda x: "Normal" if x in normal_list else "Attack" if x in attack_list else x)

    # categorical columns
    categorical_columns = ["proto", "service"]

    if not args.all_features:
        # drop unneeded features
        df.drop(["no"], inplace=True)

else:
    console.print("Dataset unknown")

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
if args.verbose > 1:
    console.print(" - " + "\n - ".join(df.columns.tolist()))


# use features as x
X = df.drop(["LABEL", "LABEL_BOOL"], axis=1)
#  The Labels for training
y = df["LABEL"]


# Encode the binary labels
label = LabelEncoder()
y_encoded = label.fit_transform(y)
joblib.dump(label, "label_encoder.pkl")

console.print(f"\nLabels: {len(label.classes_)}")
console.print(f"- " + "\n- ".join(label.classes_))

# Apply OneHotEncoder to categorical features and scale numerical features
preprocessor = ColumnTransformer(transformers=[("num", MinMaxScaler(), X.select_dtypes(exclude=["object"]).columns), ("cat", OneHotEncoder(), categorical_columns)])

X_processed = preprocessor.fit_transform(X)

# Save the preprocessor for future predictions
joblib.dump(preprocessor, "preprocessor.pkl")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)

console.print("Create models:")
console.line()

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
            joblib.dump(model, f"model_{model_name}.pkl")
            console.print(f"✔ {model_name} created. Accuracy: {accuracy:.4f} - {round((datetime.now()-_start).total_seconds(), 1)}s")

    if model_name == "randomforest":
        # Train a RandomForest model
        from sklearn.ensemble import RandomForestClassifier

        with console.status(f"[bold green]Working on {model_name}...") as status:
            _start = datetime.now()
            model = RandomForestClassifier() # no improove with class_weight='balanced' 
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            joblib.dump(model, f"model_{model_name}.pkl")
            console.print(f"✔ {model_name} created. Accuracy: {accuracy:.4f} - {round((datetime.now()-_start).total_seconds(), 1)}s")

    if model_name == "ensemble":
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
            joblib.dump(model, f"model_{model_name}.pkl")
            console.print(f"✔ {model_name} created. Accuracy: {accuracy:.4f} - {round((datetime.now()-_start).total_seconds(), 1)}s")

    if model_name == "xgboost":
        import xgboost as xgb
        _start = datetime.now()
        # Load and preprocess your data
        # data = pd.read_csv("your_dataset.csv")
        # X = data.drop("target_column", axis=1)  # Replace 'target_column' with your target label column
        # y = data["target_column"]

        # Encode labels if needed (assuming binary classification: 'Attack' and 'Normal')
        # label_encoder = LabelEncoder()
        # y = label_encoder.fit_transform(y)  # Converts 'Attack'/'Normal' to 1/0

        # # Train-test split
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize the XGBoost classifier with imbalance handling
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            max_depth=6,
            n_estimators=100,
            learning_rate=0.1
        )
        # Train the model
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        joblib.dump(model, f"model_{model_name}.pkl")
        console.print(f"✔ {model_name} created. Accuracy: {accuracy:.4f} - {round((datetime.now()-_start).total_seconds(), 1)}s")