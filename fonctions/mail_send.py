# import re

# def is_valid_email(email):
#     # Utilisation d'une regex pour vérifier si l'email est au bon format
#     regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
#     return re.match(regex, email) is not None

# import os.path
# import base64
# from email.message import EmailMessage
# import google.auth
# from googleapiclient.discovery import build
# from googleapiclient.errors import HttpError
# from google.oauth2.credentials import Credentials
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request

# # Si vous modifiez ces portées, supprimez le fichier token.json
# SCOPES = ['https://www.googleapis.com/auth/gmail.compose']

# def gmail_authenticate():
#     """Authentification OAuth2 pour Gmail API."""
#     creds = None
#     # Le fichier token.json stocke les tokens d'accès et de rafraîchissement de l'utilisateur
#     if os.path.exists('token.json'):
#         creds = Credentials.from_authorized_user_file('token.json', SCOPES)
#     # Si il n'y a pas de token valide, il faut s'authentifier
#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file(
#                 'credentials.json', SCOPES)
#             creds = flow.run_local_server(port=0)
#         # Sauvegarder le token pour la prochaine exécution
#         with open('token.json', 'w') as token:
#             token.write(creds.to_json())
#     return creds

# def gmail_create_draft(sender_email, recipient_email, subject, newsletter_content):
#     """Créer un brouillon dans Gmail."""
#     creds = gmail_authenticate()

#     try:
#         # Créer le client Gmail API
#         service = build("gmail", "v1", credentials=creds)

#         message = EmailMessage()

#         # Contenu du mail
#         message.set_content(newsletter_content)

#         message["To"] = recipient_email
#         message["From"] = sender_email
#         message["Subject"] = subject

#         # Encoder le message au format base64
#         encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

#         create_message = {"message": {"raw": encoded_message}}

#         # Créer le brouillon dans Gmail
#         draft = (
#             service.users()
#             .drafts()
#             .create(userId="me", body=create_message)
#             .execute()
#         )

#         # print(f'Draft created with id: {draft["id"]}\nDraft message: {draft["message"]}')
#         return draft

#     except HttpError as error:
#         print(f"An error occurred: {error}")
#         return None

# recipient_email = st.text_input("Entrez l'adresse e-mail pour recevoir la newsletter :")

# Définir les informations du mail
# sender_email = "joris.entrepro@gmail.com"
# recipient_email = recipient_email
# subject = "Votre Newsletter"
# newsletter_content = """
# Bonjour,

# Voici les dernières actualités de notre newsletter !

# Cordialement,
# L'équipe.
# """
