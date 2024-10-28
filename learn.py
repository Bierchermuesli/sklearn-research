import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import joblib
from yaspin import yaspin
import argparse

import time
from datetime import datetime

models = ["perceptron", "randomforest", "ensemble"]

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
    df.rename(columns={"label": "LABEL_BIN", "type": "LABEL"}, inplace=True)

    # categorical columns
    categorical_columns = ["conn_state", "proto"]

    if not args.all_features:
        # we only care about  Connection and Statistical activity + labels
        df.drop(df.filter(regex="dns_*").columns, axis=1, inplace=True)
        df.drop(df.filter(regex="http_*").columns, axis=1, inplace=True)
        df.drop(df.filter(regex="ssl_*").columns, axis=1, inplace=True)
        df.drop(df.filter(regex="weird_*").columns, axis=1, inplace=True)

elif {"Attack_type"}.issubset(df.columns):
    # RT_IOT
    df.rename(columns={"Attack_type": "LABEL"}, inplace=True)

    # Relabel for binary classification and keep original attack types
    normal_list = ["MQTT_Publish", "Thing_Speak", "Wipro_bulb", "Amazon-Alexa", "TV"]
    attack_list = ["DOS_SYN_Hping", "ARP_poisioning", "NMAP_UDP_SCAN", "NMAP_XMAS_TREE_SCAN", "NMAP_OS_DETECTION", "NMAP_TCP_scan", "DDOS_Slowloris", "Metasploit_Brute_Force_SSH", "NMAP_FIN_SCAN"]
    df["LABEL_BIN"] = df["LABEL"].apply(lambda x: "Normal" if x in normal_list else "Attack" if x in attack_list else x)

    # categorical columns
    categorical_columns = ["proto", "service"]

    if not args.all_features:
        # drop unneeded features
        df.drop(["no"], inplace=True)

else:
    print("Dataset unknown")

print(f"Rows: {len(df)}")
print(f"Features: {len(df.columns)}")
if args.verbose > 1:
    print("\n - ".join(df.columns.tolist()))


# use features as x
X = df.drop(["LABEL", "LABEL_BIN"], axis=1)
#  The Labels for training
y = df["LABEL"]


# Encode the binary labels
label = LabelEncoder()
y_encoded = label.fit_transform(y)
joblib.dump(label, "label_encoder.pkl")

print(f"Labels: {len(label.classes_)}")
print(f"\n- ".join(label.classes_))

# Apply OneHotEncoder to categorical features and scale numerical features
preprocessor = ColumnTransformer(transformers=[("num", MinMaxScaler(), X.select_dtypes(exclude=["object"]).columns), ("cat", OneHotEncoder(), categorical_columns)])

X_processed = preprocessor.fit_transform(X)

# Save the preprocessor for future predictions
joblib.dump(preprocessor, "preprocessor.pkl")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)

for model in args.models:

    if model == "perceptron":
        # Train a Perceptron model
        from sklearn.linear_model import Perceptron

        with yaspin(text=f"Create {model} Model", color="yellow") as sp:
            _start = datetime.now()
            sp.write(f"Prepping {model}")
            model = Perceptron()
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            sp.write(f"Accuracy: {accuracy:.4f}")
            sp.write(f"Saving model to model_{model}")
            joblib.dump(model, f"model_{model}")
            sp.ok(f"✔ {round((datetime.now()-_start).total_seconds(), 1)}s")

    if model == "randomforest":
        # Train a RandomForest model
        target = "model_randomforest.pkl"
        from sklearn.ensemble import RandomForestClassifier

        with yaspin(text=f"Create {model} Model", color="yellow") as sp:
            _start = datetime.now()
            sp.write(f"Prepping {model}")
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            sp.write(f"Accuracy: {accuracy:.4f}")
            sp.write(f"Saving model to {target}")
            joblib.dump(model, target)
            sp.ok(f"✔ {round((datetime.now()-_start).total_seconds(), 1)}s")

    if model == "ensemble":
        # lets add mooar more models
        from sklearn.ensemble import VotingClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import accuracy_score

        random_forest = RandomForestClassifier()
        decision_tree = DecisionTreeClassifier()
        knn = KNeighborsClassifier()
        logistic_reg = LogisticRegression(max_iter=1000)
        mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, 50), activation="relu", solver="adam", random_state=42)

        # Create a list of models for VotingClassifier
        models = [
            ("Random Forest", random_forest),
            ("Decision Tree", decision_tree),
            ("KNN", knn),
            ("Logistic Regression", logistic_reg),
            ("MLP Classifier", mlp_classifier),
        ]

        target = "model_ensemble.pkl"
        with yaspin(text=f"Create {model} Model", color="yellow") as sp:
            _start = datetime.now()
            sp.write(f"Prepping {model}")
            # Initialize VotingClassifier with the list of models
            model = VotingClassifier(estimators=models, voting="hard")
            # Train the VotingClassifier
            model.fit(X_train, y_train)
            # Make predictions using the VotingClassifier
            y_pred = model.predict(X_test)
            # Evaluate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            sp.write(f"Accuracy: {accuracy:.6f}")
            joblib.dump(model, target)
            sp.write(f"Saving model to {target}")
            sp.ok(f"✔ {round((datetime.now()-_start).total_seconds(), 1)}s")
