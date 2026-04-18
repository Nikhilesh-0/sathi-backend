"""
Gemini Service — handles all LLM interactions for Sathi.

Key design decisions:
- System prompt is constructed fresh each call (includes FD data + retrieved context)
- Tool calling is implemented via structured output parsing from markers in response
- Language instruction is embedded in the system prompt
- Booking state is tracked server-side and passed in from the frontend
"""

import json
import os
import re
from typing import List, Optional
import google.generativeai as genai
from app.models.schemas import Message, BookingState
from app.services.faiss_service import get_context_for_query

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Load FD products once
_FD_DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/fd_products.json")
with open(_FD_DATA_PATH, "r") as f:
    FD_PRODUCTS = json.load(f)

LANGUAGE_INSTRUCTIONS = {
    "hindi": "ALWAYS respond in Hindi (Devanagari script). Use simple, everyday Hindi — not formal or bureaucratic. Avoid English jargon; when you must use a financial term, say it in English and immediately explain it in Hindi.",
    "punjabi": "ALWAYS respond in Punjabi (Gurmukhi script). Use simple, everyday Punjabi — warm and conversational. Avoid English jargon; when you must use a financial term, say it in English and immediately explain it in Punjabi.",
    "bengali": "ALWAYS respond in Bengali (Bengali script). Use simple, everyday Bengali — friendly and approachable. Avoid English jargon; when you must use a financial term, say it in English and immediately explain it in Bengali.",
}

BOOKING_ENTITY_PROMPT = """
You are also responsible for collecting booking information. Current booking state:
{booking_state_json}

Missing fields: {missing_fields}

If the user provides any of these values in their message, extract them and include them in your response as a JSON block at the very end of your message, like this:
<booking_update>
{{"principal_amount": 50000, "tenor_months": 12, "pan_number": "ABCDE1234F", "nominee_name": "Priya Sharma"}}
</booking_update>

Only include fields that were just provided. Do not fabricate values. If all 4 fields are now collected, congratulate the user warmly and tell them you're creating their FD booking.

PAN format: 5 uppercase letters, 4 digits, 1 uppercase letter (e.g., ABCDE1234F). If user provides invalid PAN, ask them to re-check.
"""


def build_system_prompt(language: str, retrieved_chunks: List[str], booking_state: Optional[dict]) -> str:
    fd_json_str = json.dumps(FD_PRODUCTS, indent=2, ensure_ascii=False)
    lang_instruction = LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["hindi"])

    context_section = ""
    if retrieved_chunks:
        context_section = f"""
## RETRIEVED KNOWLEDGE (use this to answer conceptual questions accurately):
{chr(10).join(f'- {chunk}' for chunk in retrieved_chunks)}
"""

    booking_section = ""
    if booking_state and booking_state.get("stage") in ("collecting", "confirming"):
        state = BookingState(**booking_state)
        missing = []
        if not state.principal_amount:
            missing.append("principal_amount (amount in ₹ they want to invest)")
        if not state.tenor_months:
            missing.append("tenor_months (duration in months)")
        if not state.pan_number:
            missing.append("pan_number (their PAN card number)")
        if not state.nominee_name:
            missing.append("nominee_name (name of their nominee)")

        booking_section = BOOKING_ENTITY_PROMPT.format(
            booking_state_json=json.dumps(booking_state, ensure_ascii=False),
            missing_fields=", ".join(missing) if missing else "None — all collected!"
        )

    return f"""You are Sathi (साथी / ਸਾਥੀ / সাথী), a warm and trusted FD (Fixed Deposit) advisor for Indian users. Your name means "companion" or "friend." You speak like a knowledgeable family friend, not a bank employee or financial robot.

## LANGUAGE RULE (MOST IMPORTANT):
{lang_instruction}

## YOUR PERSONA:
- Warm, patient, never condescending
- You celebrate small savings ("₹10,000 is a great start!")  
- You use relatable comparisons ("8.5% matlab ₹10,000 lagao, saal baad ₹10,850 milega")
- You never overwhelm — explain one thing at a time
- You proactively ask about the user's goal before recommending

## AVAILABLE FD PRODUCTS (current, accurate data — use ONLY these rates):
{fd_json_str}

{context_section}

## HOW TO HANDLE QUESTIONS:
1. Jargon questions (what is tenor, p.a., TDS, etc.) → use the retrieved knowledge above to explain in simple terms with an example
2. "Which FD is best?" → First ask: What is their goal? How much money? How long? Then recommend top 2-3 options from the product list above with a comparison
3. Booking intent → Switch to booking collection mode (collect 4 entities: amount, tenor, PAN, nominee)
4. General financial questions → Answer helpfully but stay focused on FDs

## FD RECOMMENDATION FORMAT:
When recommending FDs, end your message with a special marker so the frontend can render cards:
<fd_recommendations>
["fd_001", "fd_007", "fd_011"]
</fd_recommendations>
(Use the id field from the FD products list)

## NEVER:
- Invent interest rates not in the product list
- Give tax or legal advice beyond what's in the knowledge base
- Recommend FDs outside the provided list
- Switch languages mid-conversation

{booking_section}

Remember: You are talking to someone in a tier-2/3 Indian city who may never have invested before. Be their Sathi.
"""


def is_conceptual_question(message: str) -> bool:
    """
    Heuristic to decide whether to use FAISS retrieval.
    Booking/amount questions don't need RAG. Jargon/concept questions do.
    """
    conceptual_keywords = [
        "kya hai", "matlab", "matalab", "samjhao", "batao", "explain",
        "what is", "what are", "how does", "TDS", "DICGC", "cumulative",
        "non-cumulative", "tenor", "maturity", "nominee", "KYC", "PAN",
        "penalty", "withdrawal", "insurance", "tax", "15G", "15H",
        "ki hai", "hunda hai", "ki hoe", "কি", "কেন", "কিভাবে",
        "kaise", "kyun", "why", "kaisa", "compare", "difference",
        "safer", "safe", "risky", "better", "best", "choose",
    ]
    message_lower = message.lower()
    return any(kw.lower() in message_lower for kw in conceptual_keywords)


def parse_booking_update(response_text: str) -> Optional[dict]:
    """Extract booking update JSON from model response if present."""
    match = re.search(r'<booking_update>(.*?)</booking_update>', response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except Exception:
            return None
    return None


def parse_fd_recommendations(response_text: str) -> Optional[List[dict]]:
    """Extract FD IDs from response and return full product objects."""
    match = re.search(r'<fd_recommendations>(.*?)</fd_recommendations>', response_text, re.DOTALL)
    if match:
        try:
            ids = json.loads(match.group(1).strip())
            fd_map = {fd["id"]: fd for fd in FD_PRODUCTS}
            return [fd_map[id] for id in ids if id in fd_map]
        except Exception:
            return None
    return None


def clean_response(text: str) -> str:
    """Remove internal markers from the response before sending to frontend."""
    text = re.sub(r'<booking_update>.*?</booking_update>', '', text, flags=re.DOTALL)
    text = re.sub(r'<fd_recommendations>.*?</fd_recommendations>', '', text, flags=re.DOTALL)
    return text.strip()


async def get_gemini_response(
    message: str,
    language: str,
    history: List[Message],
    booking_state: Optional[dict]
) -> dict:
    """
    Main function called by the chat router.
    Returns dict with: reply, booking_update, fd_cards, retrieved_context_used
    """
    # Step 1: Decide if we need RAG
    use_rag = is_conceptual_question(message)
    retrieved_chunks = []
    if use_rag:
        retrieved_chunks, _ = get_context_for_query(message)

    # Step 2: Build system prompt
    system_prompt = build_system_prompt(language, retrieved_chunks, booking_state)

    # Step 3: Build conversation history for Gemini
    gemini_history = []
    for msg in history[-10:]:  # Last 10 messages for context window efficiency
        gemini_history.append({
            "role": "user" if msg.role == "user" else "model",
            "parts": [msg.content]
        })

    # Step 4: Call Gemini
    gemini_model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system_prompt
    )

    chat = gemini_model.start_chat(history=gemini_history)
    response = chat.send_message(message)
    raw_text = response.text

    # Step 5: Parse structured outputs
    booking_update = parse_booking_update(raw_text)
    fd_cards = parse_fd_recommendations(raw_text)
    clean_reply = clean_response(raw_text)

    return {
        "reply": clean_reply,
        "booking_update": booking_update,
        "fd_cards": fd_cards,
        "retrieved_context_used": use_rag and len(retrieved_chunks) > 0
    }
