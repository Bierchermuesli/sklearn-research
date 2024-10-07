import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import joblib
from yaspin import yaspin
import argparse


parser = argparse.ArgumentParser(description="Predict network attacks using a trained model.")
parser.add_argument('-v', '--verbose', action='count', default=0,
                    help='Increase output verbosity. Use multiple times for more verbosity.')
parser.add_argument('-d','--data-set', type=str,
                    help='Path to the data set CSV file.',default="dataset/RT_IOT2022.csv")   
args = parser.parse_args()



# Load the dataset
df = pd.read_csv(args.data_set)

# Relabel for binary classification and keep original attack types
normal_list = ["MQTT_Publish", "Thing_Speak", "Wipro_bulb", "Amazon-Alexa"]
attack_list = ["DOS_SYN_Hping", "ARP_poisioning", "NMAP_UDP_SCAN", "NMAP_XMAS_TREE_SCAN", 
               "NMAP_OS_DETECTION", "NMAP_TCP_scan", "DDOS_Slowloris", 
               "Metasploit_Brute_Force_SSH", "NMAP_FIN_SCAN"]

#create a binary label mapping
df['Binary_Label'] = df['Attack_type'].apply(lambda x: 'Normal' if x in normal_list else 'Attack' if x in attack_list else x)

# Define features and labels
X = df.drop(['no','Attack_type', 'Binary_Label'], axis=1)  # Drop original and binary label columns
y = df['Binary_Label']  # Use the binary label for training

# Encode the binary labels
label = LabelEncoder()
y_encoded = label.fit_transform(y)
joblib.dump(label, 'label_encoder.pkl')

# Identify categorical columns for one-hot encoding
categorical_columns = ['proto', 'service']

# Apply OneHotEncoder to categorical features and scale numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), X.select_dtypes(exclude=['object']).columns),
        ('cat', OneHotEncoder(), categorical_columns)
    ])

X_processed = preprocessor.fit_transform(X)

# Save the preprocessor for future predictions
joblib.dump(preprocessor, 'preprocessor.pkl')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_encoded, test_size=0.2, random_state=42
)

# Train a Perceptron model
target = 'model_perceptron.pkl'
from sklearn.linear_model import Perceptron
with yaspin(text="Create Perceptron Model", color="yellow") as sp:
    sp.write("Prepping the RandomForest")
    model = Perceptron()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    sp.write(f"Accuracy: {accuracy:.4f}")
    sp.write(f"Saving model to {target}")
    joblib.dump(model, target)
    sp.ok("✔")

# Train a RandomForest model
target = 'model_randomforest.pkl'
from sklearn.ensemble import RandomForestClassifier
with yaspin(text="Create RandomForest Model", color="yellow") as sp:
    sp.write("Prepping the RandomForest")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    sp.write(f"Accuracy: {accuracy:.4f}")
    sp.write(f"Saving model to {target}")
    joblib.dump(model, target)
    sp.ok("✔")


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
with yaspin(text="Create Ensemble Model", color="yellow") as sp:   
    sp.write("Prepping the VotingClassifier")
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
    sp.ok("✔")
