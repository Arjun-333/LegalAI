import os
import io
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from pathlib import Path

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def download_pdfs_from_drive(folder_id, local_dir):
    """Downloads all PDFs from a specific Google Drive folder."""
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('drive', 'v3', credentials=creds)
    
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # Query for files in the folder
    query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
    results = service.files().list(q=query, fields="nextPageToken, files(id, name)").execute()
    items = results.get('files', [])

    if not items:
        print('No files found.')
        return

    print(f"Found {len(items)} PDFs in Drive folder.")
    for item in items:
        file_id = item['id']
        file_name = item['name']
        file_path = local_dir / file_name
        
        if file_path.exists():
            print(f"Skipping {file_name} (already exists)")
            continue

        print(f"Downloading {file_name}...")
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            
        with open(file_path, 'wb') as f:
            f.write(fh.getbuffer())

if __name__ == '__main__':
    # You need to put your Folder ID here
    # Example: https://drive.google.com/drive/folders/1A2B3C... -> ID is 1A2B3C...
    DRIVE_FOLDER_ID = "YOUR_FOLDER_ID_HERE" 
    LOCAL_DESTINATION = "./PDF_FROM_DRIVE"
    
    if DRIVE_FOLDER_ID == "YOUR_FOLDER_ID_HERE":
        print("Please edit the script to set your DRIVE_FOLDER_ID")
    else:
        download_pdfs_from_drive(DRIVE_FOLDER_ID, LOCAL_DESTINATION)
