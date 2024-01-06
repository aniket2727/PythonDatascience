from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import numpy as np

class AdaBoost:
    def __init__(self, n_classifiers=50):
        self.n_classifiers = n_classifiers
        self.classifiers = []
        self.alphas = []

    def fit(self, X, y):
        n_samples, _ = X.shape
        # Initialize weights
        w = np.ones(n_samples) / n_samples

        for t in range(self.n_classifiers):
            # Train weak learner (Decision Stump in this case)
            classifier = DecisionTreeClassifier(max_depth=1)
            classifier.fit(X, y, sample_weight=w)
            
            # Make predictions
            predictions = classifier.predict(X)
            
            # Compute weighted error
            err = np.sum(w * (predictions != y))
            
            # Compute classifier weight (alpha)
            alpha = 0.5 * np.log((1 - err) / max(err, 1e-10))

            # Update sample weights
            w *= np.exp(-alpha * y * predictions)
            
            # Normalize weights
            w /= np.sum(w)

            # Save classifier and alpha
            self.classifiers.append(classifier)
            self.alphas.append(alpha)

    def predict(self, X):
        # Make predictions using all weak classifiers
        weak_preds = np.array([classifier.predict(X) for classifier in self.classifiers])

        # Combine predictions using alpha values
        final_pred = np.sign(np.dot(self.alphas, weak_preds))

        return final_pred

# Example usage:
# Create a synthetic dataset
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Train AdaBoost
adaboost = AdaBoost(n_classifiers=50)
adaboost.fit(X, y)

# Make predictions
new_sample = np.array([[3, 3]])
prediction = adaboost.predict(new_sample)
print(f"Prediction for {new_sample}: {prediction[0]}")

# Evaluate accuracy on the training set
train_predictions = adaboost.predict(X)
accuracy = accuracy_score(y, train_predictions)
print(f"Accuracy on the training set: {accuracy}")
