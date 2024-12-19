# Import necessary libraries
import pandas as pd

# Load the dataset
data = pd.read_csv('C:\\Users\\olegr\\Downloads\\cardio_train.csv', delimiter=';')

# Check the dataset structure
print("Dataset Shape:", data.shape)  # Confirm number of rows and columns
print("Columns:", data.columns)  # List column names


# Convert age from days to years
data['age'] = data['age'] / 365.25

# Create a new feature for BMI
data['BMI'] = data['weight'] / ((data['height'] / 100) ** 2)

# Check for missing values
print("\nMissing Values:\n", data.isnull().sum())  # Ensure no missing data

# Check for duplicate rows
print("\nDuplicate Rows:", data.duplicated().sum())  # Ensure no duplicate records

# Save the updated dataset (optional)
data.to_csv('cardio_cleaned.csv', index=False)
print("\nCleaned dataset saved as 'cardio_cleaned.csv'")