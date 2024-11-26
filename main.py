from flask import Flask, request, jsonify
import pickle

# Placeholder code from chatgpt spam detection

# Save the trained model and vectorizer
with open('spam_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

# Load model and vectorizer
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
