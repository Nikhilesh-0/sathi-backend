from pydantic import BaseModel
from typing import Optional, List


class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str
    language: str  # "hindi", "punjabi", "bengali"
    history: List[Message]
    booking_state: Optional[dict] = None  # tracks extracted entities


class BookingState(BaseModel):
    principal_amount: Optional[float] = None
    tenor_months: Optional[int] = None
    pan_number: Optional[str] = None
    nominee_name: Optional[str] = None
    selected_fd_id: Optional[str] = None
    stage: str = "advisory"  # advisory → collecting → confirming → booked


class ChatResponse(BaseModel):
    reply: str
    booking_state: Optional[BookingState] = None
    fd_cards: Optional[List[dict]] = None  # FD products to render as cards
    booking_receipt: Optional[dict] = None
    retrieved_context_used: bool = False


class HistoryRequest(BaseModel):
    user_id: str
    session_id: str
    messages: List[Message]


class UserSession(BaseModel):
    session_id: str
    title: str
    created_at: str
    language: str
