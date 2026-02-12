import joblib
import numpy as np

class ActionClassifier:
    def __init__(self, model_path="models/action_classifier.pkl"):
        self.model = joblib.load(model_path)

    def predict(self, features):
        X = np.array(features).reshape(1, -1)
        prediction = self.model.predict(X)[0]

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            confidence = max(proba[0])
        else:
            confidence = 1.0

        return prediction, confidence
