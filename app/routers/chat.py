from fastapi import APIRouter, HTTPException, Header
from typing import Optional
from app.models.schemas import ChatRequest, ChatResponse, BookingState
from app.services.gemini_service import get_gemini_response
from app.services.booking_service import create_fd_booking
from app.services.auth_service import verify_token, get_firestore_client
from datetime import datetime
import json

router = APIRouter()


def update_booking_state(current_state: Optional[dict], update: dict) -> dict:
    """Merges new entity values into booking state."""
    if not current_state:
        current_state = {
            "principal_amount": None,
            "tenor_months": None,
            "pan_number": None,
            "nominee_name": None,
            "selected_fd_id": None,
            "stage": "collecting"
        }
    current_state.update({k: v for k, v in update.items() if v is not None})

    # Check if all 4 entities are collected
    required = ["principal_amount", "tenor_months", "pan_number", "nominee_name"]
    if all(current_state.get(f) for f in required):
        current_state["stage"] = "confirming"

    return current_state


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    authorization: Optional[str] = Header(None)
):
    # Verify auth token
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing auth token")

    token = authorization.split(" ")[1]
    try:
        user_id = verify_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid auth token")

    # Get Gemini response
    result = await get_gemini_response(
        message=request.message,
        language=request.language,
        history=request.history,
        booking_state=request.booking_state
    )

    # --- Booking state machine ---
    # Start with whatever state the frontend sent us
    booking_state = request.booking_state

    # Merge any newly extracted entities from this message
    if result["booking_update"]:
        booking_state = update_booking_state(booking_state, result["booking_update"])

    # Generate receipt if all 4 fields are present.
    # This check runs UNCONDITIONALLY — not gated on booking_update being present.
    # Handles the common case where Gemini 2.5 Flash collects the last entity
    # but skips the <booking_update> XML tag on the confirmation message.
    booking_receipt = None
    required = ["principal_amount", "tenor_months", "pan_number", "nominee_name"]
    if (
        booking_state
        and all(booking_state.get(f) for f in required)
        and booking_state.get("stage") != "booked"
    ):
        try:
            booking_receipt = create_fd_booking(
                principal_amount=booking_state["principal_amount"],
                tenor_months=booking_state["tenor_months"],
                pan_number=booking_state["pan_number"],
                nominee_name=booking_state["nominee_name"],
                fd_id=booking_state.get("selected_fd_id", "fd_002"),
                user_id=user_id
            )
            booking_state["stage"] = "booked"
        except Exception as e:
            print(f"[Booking] Warning: could not create booking receipt: {e}")

    # --- Save to Firestore ---
    try:
        db = get_firestore_client()
        session_ref = (
            db.collection("conversations")
            .document(user_id)
            .collection("sessions")
            .document(request.session_id)
        )

        messages_ref = session_ref.collection("messages")

        # Save user message
        messages_ref.add({
            "role": "user",
            "content": request.message,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Save assistant reply — persist receipt and cards so history loads them
        assistant_doc = {
            "role": "assistant",
            "content": result["reply"],
            "timestamp": datetime.utcnow().isoformat()
        }
        if booking_receipt:
            assistant_doc["booking_receipt"] = json.dumps(booking_receipt)
        if result.get("fd_cards"):
            assistant_doc["fd_cards"] = json.dumps(result["fd_cards"])
        if result.get("retrieved_context_used"):
            assistant_doc["retrieved_context_used"] = True

        messages_ref.add(assistant_doc)

        # Update session metadata
        session_ref.set({
            "title": request.message[:50] + "..." if len(request.message) > 50 else request.message,
            "updated_at": datetime.utcnow().isoformat(),
            "language": request.language
        }, merge=True)
    except Exception as e:
        print(f"[Firestore] Warning: could not save message: {e}")

    return ChatResponse(
        reply=result["reply"],
        booking_state=BookingState(**booking_state) if booking_state else None,
        fd_cards=result["fd_cards"],
        booking_receipt=booking_receipt,
        retrieved_context_used=result["retrieved_context_used"]
    )
