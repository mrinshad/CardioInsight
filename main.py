# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load datasets
df1 = pd.read_csv('./Datasets/cardio_data_processed.csv')
df2 = pd.read_csv('./Datasets/cardio_train.csv')
df3 = pd.read_csv('./Datasets/cardio_train1.csv')
df4 = pd.read_csv('./Datasets/data.csv')

# Concatenate datasets into a single DataFrame
df = pd.concat([df1, df2, df3, df4], axis=0, ignore_index=True)

# Define the features and target variable
features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'active']
target = 'cardio'

# Extract features and target variable
X = df[features]
y = df[target]

# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X, columns=['gender', 'cholesterol', 'gluc', 'smoke', 'active'])

# Handle missing values in features using imputation
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Handle missing values in the target variable
imputer_y = SimpleImputer(strategy='mean')
y_imputed = imputer_y.fit_transform(y.values.reshape(-1, 1))  # Reshape for 2D array

# Convert back to a Pandas Series
y_imputed = pd.Series(y_imputed.flatten(), name=target)

# Convert target variable to binary format
threshold = 0.5  # You can adjust this threshold based on your needs
y_binary = (y_imputed > threshold).astype(int)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_binary, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a Logistic Regression model
model = LogisticRegression(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Display results
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')
