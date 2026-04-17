from fastapi import APIRouter, HTTPException, Header
from typing import Optional
from app.models.schemas import ChatRequest, ChatResponse, BookingState
from app.services.gemini_service import get_gemini_response
from app.services.booking_service import create_fd_booking
from app.services.auth_service import verify_token, get_firestore_client
from datetime import datetime

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

    # Update booking state if entities were extracted
    booking_state = request.booking_state
    booking_receipt = None

    if result["booking_update"]:
        booking_state = update_booking_state(booking_state, result["booking_update"])

        # If all entities collected, create booking
        if booking_state.get("stage") == "confirming":
            booking_receipt = create_fd_booking(
                principal_amount=booking_state["principal_amount"],
                tenor_months=booking_state["tenor_months"],
                pan_number=booking_state["pan_number"],
                nominee_name=booking_state["nominee_name"],
                fd_id=booking_state.get("selected_fd_id", "fd_002"),
                user_id=user_id
            )
            booking_state["stage"] = "booked"

    # Save message to Firestore
    try:
        db = get_firestore_client()
        session_ref = (
            db.collection("conversations")
            .document(user_id)
            .collection("sessions")
            .document(request.session_id)
        )

        # Save both user message and assistant reply
        messages_ref = session_ref.collection("messages")
        messages_ref.add({
            "role": "user",
            "content": request.message,
            "timestamp": datetime.utcnow().isoformat()
        })
        messages_ref.add({
            "role": "assistant",
            "content": result["reply"],
            "timestamp": datetime.utcnow().isoformat()
        })

        # Update session metadata
        session_ref.set({
            "title": request.message[:50] + "..." if len(request.message) > 50 else request.message,
            "updated_at": datetime.utcnow().isoformat(),
            "language": request.language
        }, merge=True)
    except Exception as e:
        # Non-fatal — don't fail the response if Firestore has issues
        print(f"[Firestore] Warning: could not save message: {e}")

    return ChatResponse(
        reply=result["reply"],
        booking_state=BookingState(**booking_state) if booking_state else None,
        fd_cards=result["fd_cards"],
        booking_receipt=booking_receipt,
        retrieved_context_used=result["retrieved_context_used"]
    )
