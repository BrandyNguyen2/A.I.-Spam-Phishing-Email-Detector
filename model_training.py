import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample

def model_predict():
    # 1. Load the dataset
    data = pd.read_csv('cleaned_fraud_dataset.csv')

    # Check for missing values and fill them
    data['Text'] = data['Text'].fillna('')
    data['Class'] = data['Class'].fillna(0)

    # 2. Handle data imbalance
    spam = data[data['Class'] == 1]
    not_spam = data[data['Class'] == 0]
    spam_upsampled = resample(spam, 
                              replace=True, 
                              n_samples=len(not_spam), 
                              random_state=42)
    data = pd.concat([not_spam, spam_upsampled])

    # 3. Extract features and labels
    X = data['Text']
    y = data['Class']

    # 4. Vectorize the text data
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(X)

    # 5. Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Train the model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # 7. Evaluate the model
    y_pred = model.predict(X_test)
    print("\nModel Stats")
    print("--------------------------------------------------------")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("--------------------------------------------------------")

    return vectorizer, model

def test_custom_input(model, vectorizer):
    while True:
        user_input = input("Enter a message to classify (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        input_vectorized = vectorizer.transform([user_input])
        prediction = model.predict(input_vectorized)
        print("Prediction:", "Spam" if prediction[0] == 1 else "Not Spam")
