# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
# Load the cleaned dataset
data = pd.read_csv('cardio_cleaned.csv')

# Define features (X) and target variable (y)
X = data.drop(['id', 'cardio'], axis=1)  # Drop irrelevant features and target column
y = data['cardio']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'BMI']
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Check the class distribution after SMOTE
print("Class Distribution After SMOTE:\n", y_train.value_counts())

# Save the preprocessed data (optional)
X_train.to_csv('X_train_preprocessed.csv', index=False)
X_test.to_csv('X_test_preprocessed.csv', index=False)
y_train.to_csv('y_train_preprocessed.csv', index=False)
y_test.to_csv('y_test_preprocessed.csv', index=False)

print("Preprocessed data saved as CSV files.")