import pandas as pd
import re

# 1. Load the dataset
# Replace 'spam_dataset.csv' with your file name
data = pd.read_csv('fraud_email_.csv', engine='python', on_bad_lines='skip')

# 2. Clean the text data
def clean_text(text):
    if pd.isnull(text):  # Handle missing values
        return ''
    # Remove special characters like '=' and '\n'
    text = re.sub(r'=', '', text)
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    return text.strip()

data['Text'] = data['Text'].apply(clean_text)

# 3. Standardize labels
# Ensure 'Class' column is numeric (0 for non-spam, 1 for spam)
data['Class'] = data['Class'].astype(str).str.strip()  # Remove extra spaces
data['Class'] = data['Class'].replace({'1': 1, '0': 0}).astype(int)  # Map to binary

# 4. Drop rows with missing or invalid data
data = data.dropna(subset=['Text', 'Class'])  # Remove rows where 'Text' or 'Class' is NaN

# 5. Verify the cleaned dataset
print("Dataset Info:")
print(data.info())
print("\nSample Data:")
print(data.head())

# Save the cleaned data to a new CSV file
data.to_csv('cleaned_fraud_dataset.csv', index=False)
