import os
import pickle
import base64
import json
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, request, jsonify

import model_training as mt

vectorizer, model = mt.model_predict()

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

# Function to authenticate with Gmail API
def authenticate_gmail():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('gmail', 'v1', credentials=creds)
    return service

# Function to extract the body of the email
def extract_email_body(message, msg_id):
    try:
        # Extract email payload
        payload = message['payload']
        headers = payload.get('headers', [])
        
        # Find the subject
        for header in headers:
            if header['name'] == 'Subject':
                print(f"\tSubject: {header['value']}")  # Debug: Subject line

        # Extract email body
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    return part['body']['data']
        elif 'body' in payload:
            return payload['body']['data']
        
        return "No body found."
    except Exception as e:
        print(f"Error extracting email body for message ID {msg_id}: {e}")
        return "Error in extraction."


# Function to forward the email content to your spam detector
def forward_to_spam_detector(email_body):
    # Here, call your spam detection model and forward the result
    prediction = model.predict(vectorizer.transform([email_body]))
    label = 'Spam' if prediction[0] == 1 else 'Not Spam'
    print(f"\tPrediction: {label}")
    return label

# Function to list unread emails
def list_emails(service):
    try:
        print("\nChecking for unread messages...\n")
        results = service.users().messages().list(userId='me', labelIds=['INBOX'], q="is:unread").execute()
        messages = results.get('messages', [])
        
        if not messages:
            print("No unread messages found.\n")
            return

        print(f"Found {len(messages)} unread message(s).") 
        
        for msg in messages:
            msg_id = msg['id']
            message = service.users().messages().get(userId='me', id=msg_id).execute()
            print(f"Processing message...") 
            
            email_body = extract_email_body(message, msg_id)
            
            forward_to_spam_detector(email_body)
            
            # Mark the message as read
            service.users().messages().modify(userId='me', id=msg_id, body={'removeLabelIds': ['UNREAD']}).execute()
            print(f"Marked message as read.\n")  # Debug print

    except Exception as e:
        print(f"Error in list_emails: {e}")

# Flask app setup
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    email_text = data.get('email_text', '')
    
    # Use your model to predict
    email_vector = vectorizer.transform([email_text])
    prediction = model.predict(email_vector)
    label = 'spam' if prediction[0] == 1 else 'ham'
    
    return jsonify({'label': label})

if __name__ == '__main__':
    # Load your model and vectorizer here (or from files)
    with open('spam_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)

    # Authenticate and list emails in a separate thread if needed for background processing
    service = authenticate_gmail()
    list_emails(service)  # Or call this periodically as needed

    app.run(debug=False)
