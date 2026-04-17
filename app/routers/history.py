from fastapi import APIRouter, HTTPException, Header
from typing import Optional
from app.services.auth_service import verify_token, get_firestore_client

router = APIRouter()


@router.get("/history/sessions")
async def get_sessions(authorization: Optional[str] = Header(None)):
    """Returns all conversation sessions for the authenticated user."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing auth token")

    token = authorization.split(" ")[1]
    try:
        user_id = verify_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid auth token")

    db = get_firestore_client()
    sessions_ref = (
        db.collection("conversations")
        .document(user_id)
        .collection("sessions")
    )
    sessions = sessions_ref.order_by("updated_at", direction="DESCENDING").limit(20).get()

    result = []
    for s in sessions:
        data = s.to_dict()
        data["session_id"] = s.id
        result.append(data)

    return {"sessions": result}


@router.get("/history/sessions/{session_id}")
async def get_session_messages(session_id: str, authorization: Optional[str] = Header(None)):
    """Returns all messages for a specific session."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing auth token")

    token = authorization.split(" ")[1]
    try:
        user_id = verify_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid auth token")

    db = get_firestore_client()
    messages_ref = (
        db.collection("conversations")
        .document(user_id)
        .collection("sessions")
        .document(session_id)
        .collection("messages")
    )
    messages = messages_ref.order_by("timestamp").get()

    return {
        "session_id": session_id,
        "messages": [
            {"role": m.to_dict()["role"], "content": m.to_dict()["content"]}
            for m in messages
        ]
    }
