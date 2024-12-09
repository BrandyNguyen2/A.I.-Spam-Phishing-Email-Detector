import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the dataset
data = pd.read_csv('cleaned_fraud_dataset.csv')

# Check for missing values
print("Missing values in Text column:", data['Text'].isnull().sum())

# Handle missing values
data['Text'] = data['Text'].fillna('')  # Replace NaN with empty string
data['Class'] = data['Class'].fillna(0)  # Ensure no missing labels

# 2. Convert labels to binary
data['Class'] = data['Class'].map({1: 1, 0: 0})  # Ensure labels are 0 or 1
X = data['Text']
y = data['Class']

# 3. Convert text to feature vectors
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# 6. Predict on the test data
y_pred = model.predict(X_test)

# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

def test_custom_input(model, vectorizer):
    while True:
        user_input = input("Enter a message to classify (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Exiting.")
            break
        # Transform the user input using the fitted vectorizer
        input_vectorized = vectorizer.transform([user_input])
        # Predict using the trained model
        prediction = model.predict(input_vectorized)
        if prediction[0] == 1:
            print("Prediciton: Not Spam")
        print("Prediction: Spam")

# Call the function to test custom input
test_custom_input(model, vectorizer)
