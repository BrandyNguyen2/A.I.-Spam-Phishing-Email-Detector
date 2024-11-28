import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

try:
    # Step 1: Read the CSV file
    data = pd.read_csv('fraud_email_.csv')
    
    # Check if the DataFrame is empty
    if data.empty:
        raise ValueError("The CSV file is empty")

    # Print the columns to verify
    print("Columns in the CSV file:", data.columns)

    # Step 2: Preprocess the data
    # Assuming the CSV has columns 'email_text' and 'label'
    if 'email_text' not in data.columns or 'label' not in data.columns:
        raise ValueError("Required columns 'email_text' or 'label' are missing in the CSV file")

    X = data['email_text']
    y = data['label']

    # Convert text to vectors
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    # Step 3: Train the model
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

    # Step 4: Save the model and vectorizer
    with open('spam_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('vectorizer.pkl', 'wb') as vec_file:
        pickle.dump(vectorizer, vec_file)

except pd.errors.EmptyDataError:
    print("Error: The CSV file is empty or does not contain any columns to parse.")
except FileNotFoundError:
    print("Error: The CSV file was not found.")
except ValueError as ve:
    print(f"ValueError: {ve}")
except Exception as e:
    print(f"An error occurred: {e}")