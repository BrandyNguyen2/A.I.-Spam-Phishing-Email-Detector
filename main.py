from flask import Flask, jsonify, render_template
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import os
import base64
from email import message_from_bytes
import model_training as mt

app = Flask(__name__)

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

# Load the model and vectorizer
with open('spam_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

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

def extract_email_body(message):
    payload = message['payload']
    body = ''
    if 'parts' in payload:
        for part in payload['parts']:
            if part.get('mimeType') == 'text/plain':
                body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
    elif 'body' in payload and 'data' in payload['body']:
        body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
    return body.strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_inbox', methods=['POST'])
def check_inbox():
    service = authenticate_gmail()
    try:
        results = service.users().messages().list(userId='me', labelIds=['INBOX'], q="is:unread").execute()
        messages = results.get('messages', [])
        
        response = {'messages': []}
        for msg in messages:
            msg_id = msg['id']
            message = service.users().messages().get(userId='me', id=msg_id).execute()
            subject = next((header['value'] for header in message['payload']['headers'] if header['name'] == 'Subject'), 'No Subject')
            email_body = extract_email_body(message)
            prediction = model.predict(vectorizer.transform([email_body]))[0]
            label = 'Spam' if prediction == 1 else 'Not Spam'
            response['messages'].append({'subject': subject, 'prediction': label})
            service.users().messages().modify(userId='me', id=msg_id, body={'removeLabelIds': ['UNREAD']}).execute()

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
