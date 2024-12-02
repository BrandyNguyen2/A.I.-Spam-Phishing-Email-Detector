from model_training import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data_path = 'fraud_email_.csv'
data = pd.read_csv(data_path)

# Preprocess the dataset
data['email_text'] = data['email_text'].str.lower()
data['email_text'] = data['email_text'].str.replace(r'[^\w\s]', '', regex=True)

# Split data
X = data['email_text']
y = data['label']  # Assuming 'label' column has 1 for spam, 0 for ham
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate model
y_pred = model.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save the model and vectorizer
with open('spam_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

# Load model and vectorizer in Flask
app = Flask(__name__)
with open('spam_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    email_text = data.get('email_text', '')

    # Preprocess and predict
    email_vector = vectorizer.transform([email_text])
    prediction = model.predict(email_vector)
    label = 'spam' if prediction[0] == 1 else 'ham'

    return jsonify({'label': label})

if __name__ == '__main__':
    app.run(debug=True)
