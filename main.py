import os
import pickle
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from flask import Flask, request, jsonify, render_template

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

def extract_email_subject(message):
    headers = message['payload']['headers']
    for header in headers:
        if header['name'] == 'Subject':
            return header['value']
    return "No Subject"


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

# Flask app setup
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
# Function to list unread emails
def predict():
    service = authenticate_gmail()
    try:
        print("\nChecking for unread messages...\n")
        results = service.users().messages().list(userId='me', labelIds=['INBOX'], q="is:unread").execute()
        messages = results.get('messages', [])
        
        if not messages:
            print("No unread messages found.\n")
            return jsonify({'messages' : []})

        print(f"Found {len(messages)} unread message(s).") 

        email_results = []
        
        for msg in messages:
            msg_id = msg['id']
            message = service.users().messages().get(userId='me', id=msg_id).execute()
            print(f"Processing message...") 
            
            email_subject = extract_email_subject(message)
            email_body = extract_email_body(message, msg_id)
            
            label = forward_to_spam_detector(email_body)
            email_results.append({'subject':email_subject, 'label':label})
            
            # Mark the message as read
            service.users().messages().modify(userId='me', id=msg_id, body={'removeLabelIds': ['UNREAD']}).execute()
            print(f"Marked message as read.\n")  # Debug print
        
        return jsonify({'messages':email_results})

    except Exception as e:
        print(f"Error in list_emails: {e}")

if __name__ == '__main__':
    # Load your model and vectorizer here (or from files)
    with open('spam_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)

    app.run(debug=False)