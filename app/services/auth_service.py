"""
Auth Service — verifies Firebase ID tokens from the frontend.
The frontend sends the Google Sign-In ID token in every request header.
We verify it server-side using Firebase Admin SDK.

Supports two credential modes:
1. File-based: Set FIREBASE_CREDENTIALS_PATH to a JSON file path (local dev)
2. Env-based: Set FIREBASE_CREDENTIALS_JSON to the full JSON string (Railway deployment)
"""

import os
import json
import firebase_admin
from firebase_admin import credentials, auth, firestore

# Initialize Firebase Admin once
if not firebase_admin._apps:
    # Prefer env var (better for Railway/cloud), fall back to file
    cred_json_str = os.environ.get("FIREBASE_CREDENTIALS_JSON")
    if cred_json_str:
        cred_dict = json.loads(cred_json_str)
        cred = credentials.Certificate(cred_dict)
    else:
        cred_path = os.environ.get("FIREBASE_CREDENTIALS_PATH", "firebase-credentials.json")
        cred = credentials.Certificate(cred_path)
    
    firebase_admin.initialize_app(cred)

db = firestore.client()


def verify_token(id_token: str) -> str:
    """
    Verifies a Firebase ID token and returns the user_id (uid).
    Raises an exception if token is invalid/expired.
    """
    decoded = auth.verify_id_token(id_token)
    return decoded["uid"]


def get_firestore_client():
    return db
