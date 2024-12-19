# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the preprocessed data
import pandas as pd
X_train = pd.read_csv('X_train_preprocessed.csv')
X_test = pd.read_csv('X_test_preprocessed.csv')
y_train = pd.read_csv('y_train_preprocessed.csv').values.ravel()
y_test = pd.read_csv('y_test_preprocessed.csv').values.ravel()

# Train a Random Forest Classifier
print("Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluate the Random Forest model
print("\nRandom Forest Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# Train an XGBoost Classifier
print("\nTraining XGBoost Classifier...")
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the XGBoost model
print("\nXGBoost Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

# Save model predictions
rf_predictions = pd.DataFrame({'Actual': y_test, 'Predicted_RF': y_pred_rf})
xgb_predictions = pd.DataFrame({'Actual': y_test, 'Predicted_XGB': y_pred_xgb})
rf_predictions.to_csv('random_forest_predictions.csv', index=False)
xgb_predictions.to_csv('xgboost_predictions.csv', index=False)

print("\nAdvanced models completed and predictions saved.")