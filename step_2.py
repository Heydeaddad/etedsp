# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
data = pd.read_csv('cardio_cleaned.csv')

# Analyze feature distributions
print("Feature Distributions:")

# Plot histograms for numerical features
numerical_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'BMI']
data[numerical_features].hist(bins=20, figsize=(15, 10))
plt.suptitle('Distribution of Numerical Features')
plt.show()

# Plot bar plots for categorical features
categorical_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
for feature in categorical_features:
    sns.countplot(data=data, x=feature, hue='cardio', palette='viridis')
    plt.title(f'{feature.capitalize()} Distribution by Cardiovascular Disease')
    plt.xlabel(feature.capitalize())
    plt.ylabel('Count')
    plt.legend(title='Cardio', loc='upper right', labels=['No Disease', 'Disease'])
    plt.show()

# Check correlations
print("\nCorrelation Matrix:")
correlation_matrix = data.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap')
plt.show()

# Highlight correlation with target variable
correlation_with_target = correlation_matrix['cardio'].sort_values(ascending=False)
print("Correlation with Target (Cardio):\n", correlation_with_target)

# Analyze relationships
# Pair plot for selected features and target variable
selected_features = ['age', 'BMI', 'ap_hi', 'ap_lo', 'cardio']
sns.pairplot(data[selected_features], hue='cardio', diag_kind='kde', palette='coolwarm')
plt.suptitle('Pair Plot of Selected Features and Cardio')
plt.show()

# Boxplot for systolic blood pressure by cardiovascular disease
plt.figure(figsize=(8, 6))
sns.boxplot(data=data, x='cardio', y='ap_hi', palette='viridis')
plt.title('Systolic Blood Pressure by Cardiovascular Disease')
plt.xlabel('Cardio (0 = No Disease, 1 = Disease)')
plt.ylabel('Systolic Blood Pressure (ap_hi)')
plt.show()