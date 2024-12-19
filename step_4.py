# Import necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the preprocessed data
import pandas as pd
X_train = pd.read_csv('X_train_preprocessed.csv')
X_test = pd.read_csv('X_test_preprocessed.csv')
y_train = pd.read_csv('y_train_preprocessed.csv').values.ravel()
y_test = pd.read_csv('y_test_preprocessed.csv').values.ravel()

# Initialize Logistic Regression model
model = LogisticRegression(max_iter=2000, random_state=42)

# Train the model on the training data
print("Training Logistic Regression model...")
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("\nModel Evaluation Metrics:")
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the model's predictions for analysis
predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions.to_csv('logistic_regression_predictions.csv', index=False)

print("\nBaseline Logistic Regression model completed.")