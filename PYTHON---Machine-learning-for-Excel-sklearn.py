# Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the Excel file
# Replace 'your_excel_file.xlsx' with the path to your Excel file
df = pd.read_excel('your_excel_file.xlsx')

# Step 2: Data Preprocessing
# For example, filling missing values or encoding categorical variables
# This depends on your data

# Example: Dropping rows with missing values
df = df.dropna()

# Example: Encoding categorical columns, if any
# df['categorical_column'] = df['categorical_column'].astype('category').cat.codes

# Step 3: Split the data into features (X) and labels (y)
# Replace 'target_column' with the name of the column you're predicting
X = df.drop('target_column', axis=1)  # Features
y = df['target_column']  # Target variable

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Initialize and Train the model
clf = DecisionTreeClassifier()  # You can replace this with any ML model
clf.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = clf.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Optional: Predict on new data
# new_data = pd.DataFrame({...})  # Replace with actual data for prediction
# new_predictions = clf.predict(new_data)
