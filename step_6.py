# Import necessary libraries
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

# Load preprocessed data
X_train = pd.read_csv('X_train_preprocessed.csv')
X_test = pd.read_csv('X_test_preprocessed.csv')
y_train = pd.read_csv('y_train_preprocessed.csv').values.ravel()
y_test = pd.read_csv('y_test_preprocessed.csv').values.ravel()


# Define the objective function for optuna
def objective(trial):
    # Define the hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }

    # Train the model
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


# Create and run the optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)  # Adjust n_trials as needed

# Retrieve the best hyperparameters and best score
best_params = study.best_params
best_accuracy = study.best_value
print("Best Parameters:", best_params)
print("Best Accuracy:", best_accuracy)

# Train the best model on the full training set
best_model = XGBClassifier(**best_params)
best_model.fit(X_train, y_train)

# Evaluate the best model on the test set
y_pred_test = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)

# Output evaluation results
print("\nTest Set Evaluation:")
print("Test Accuracy:", test_accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))

# Save predictions to CSV
predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test})
predictions.to_csv('optuna_xgboost_predictions.csv', index=False)

print("\nOptimized XGBoost model completed and predictions saved.")


# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# Load preprocessed data
X_train = pd.read_csv('X_train_preprocessed.csv')
X_test = pd.read_csv('X_test_preprocessed.csv')
y_train = pd.read_csv('y_train_preprocessed.csv').values.ravel()
y_test = pd.read_csv('y_test_preprocessed.csv').values.ravel()

# Define Random Forest hyperparameter grid
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Perform RandomizedSearchCV for Random Forest
print("Optimizing Random Forest...")
rf_model = RandomForestClassifier(random_state=42)
rf_random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=rf_param_grid,
    n_iter=50,
    scoring='accuracy',
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    verbose=2,
    n_jobs=-1,
    random_state=42
)
rf_random_search.fit(X_train, y_train)

# Best parameters and score for Random Forest
best_rf_params = rf_random_search.best_params_
print("Best Random Forest Parameters:", best_rf_params)
print("Best Random Forest Accuracy:", rf_random_search.best_score_)

# Train the optimized Random Forest model
optimized_rf = rf_random_search.best_estimator_
y_pred_rf = optimized_rf.predict(X_test)

# Evaluate the optimized Random Forest model
print("\nOptimized Random Forest Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))