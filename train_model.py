import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Path to your dataset
DATASET_PATH = "data/features/actions_dataset.csv"
MODEL_PATH = "models/action_classifier.pkl"

def main():
    # Load dataset
    print("📂 Loading dataset...")
    data = pd.read_csv(DATASET_PATH)

    if data.shape[0] < 10:
        print("⚠️ Warning: Very small dataset. Model may not be accurate.")

    # Assume last column is the label
    X = data.iloc[:, :-1]   # features
    y = data.iloc[:, -1]    # labels

    num_classes = y.nunique()
    total_samples = len(y)

    print(f"📊 Total samples: {total_samples}")
    print(f"🏷️ Number of classes: {num_classes}")

    # Decide whether to use stratify or not
    test_size = 0.3
    test_samples = int(total_samples * test_size)

    if test_samples < num_classes:
        print("⚠️ Dataset too small for stratified split. Using normal split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    else:
        print("✅ Using stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

    # Create model
    print("🧠 Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("✅ Training complete!")
    print(f"📊 Accuracy: {acc * 100:.2f}%")
    print("\n📄 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Create models folder if not exists
    os.makedirs("models", exist_ok=True)

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
