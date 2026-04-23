"""
Gemini Service — handles all LLM interactions for Sathi.

Key design decisions:
- System prompt is constructed fresh each call (includes FD data + retrieved context)
- Entity extraction for booking is a SEPARATE silent Gemini call (not XML tags in
  the main response). This is reliable with Gemini 2.5 Flash which often skips
  XML tags when it writes a natural conversational summary.
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

BOOKING_COLLECTION_PROMPT = """
You are also collecting FD booking details. Current booking state:
{booking_state_json}

Missing fields: {missing_fields}

Ask for ONE missing field at a time in a warm, friendly way. Do NOT ask for multiple fields in one message.

PAN format: 5 uppercase letters, 4 digits, 1 uppercase letter (e.g., ABCDE1234F). If user provides invalid PAN, ask them to re-check politely.

Once all 4 fields are collected, warmly confirm the booking details and tell the user their FD is being processed.
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
        state_obj = BookingState(**booking_state)
        missing = []
        if not state_obj.principal_amount:
            missing.append("principal_amount (amount in ₹ they want to invest)")
        if not state_obj.tenor_months:
            missing.append("tenor_months (duration in months)")
        if not state_obj.pan_number:
            missing.append("pan_number (their PAN card number)")
        if not state_obj.nominee_name:
            missing.append("nominee_name (name of their nominee)")

        booking_section = BOOKING_COLLECTION_PROMPT.format(
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


def is_booking_message(message: str, booking_state: Optional[dict]) -> bool:
    """Returns True if we are in an active booking collection flow."""
    if booking_state and booking_state.get("stage") in ("collecting", "confirming"):
        return True
    booking_triggers = [
        "book", "booking", "invest", "lagana", "lagaana", "karna chahta",
        "karna chahunda", "korte chai", "shuru", "start",
    ]
    msg_lower = message.lower()
    return any(t in msg_lower for t in booking_triggers)


async def extract_booking_entities(
    message: str,
    booking_state: dict,
    language: str
) -> Optional[dict]:
    """
    Separate silent Gemini call that ONLY extracts booking entities from the
    user's message. Returns a dict of newly provided fields, or None.

    By keeping this completely separate from the conversational response we
    guarantee entity extraction even when the main response doesn't include
    any XML tags (which Gemini 2.5 Flash often skips).
    """
    current_json = json.dumps(booking_state, ensure_ascii=False)

    extraction_prompt = f"""You are a data extraction assistant. Extract booking information from the user's message.

Current booking state: {current_json}

User message: "{message}"

Extract ONLY the fields the user just provided in this message. Return a JSON object with only the newly provided fields.
Valid fields:
- principal_amount: number (the amount in rupees they want to invest)
- tenor_months: number (duration in months, e.g. 12 for 1 year, 24 for 2 years)
- pan_number: string (10-character PAN, 5 uppercase letters + 4 digits + 1 uppercase letter)
- nominee_name: string (the name of their nominee/family member)

Rules:
- Only include fields explicitly provided in this message
- If a field was already collected (not null in current state), do not re-extract it
- If no new fields are provided, return {{}}
- Return ONLY valid JSON, no explanation, no markdown

Examples:
User: "50000 rupees lagana chahta hoon" → {{"principal_amount": 50000}}
User: "ek saal ke liye" → {{"tenor_months": 12}}
User: "ABCDE1234F hai mera PAN" → {{"pan_number": "ABCDE1234F"}}
User: "nominee mera beta Rahul hai" → {{"nominee_name": "Rahul"}}
User: "haan sab sahi hai" → {{}}
"""

    try:
        extraction_model = genai.GenerativeModel(model_name="gemini-2.5-flash")
        response = extraction_model.generate_content(extraction_prompt)
        raw = response.text.strip()

        # Strip markdown code fences if model wraps in them
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'^```\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        raw = raw.strip()

        extracted = json.loads(raw)
        if not isinstance(extracted, dict) or not extracted:
            return None
        return extracted
    except Exception as e:
        print(f"[Extraction] Warning: entity extraction failed: {e}")
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

    # Step 4: Call Gemini for conversational reply
    gemini_model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system_prompt
    )

    chat = gemini_model.start_chat(history=gemini_history)
    response = chat.send_message(message)
    raw_text = response.text

    # Step 5: Parse FD recommendations from main response (still XML-based, works fine)
    fd_cards = parse_fd_recommendations(raw_text)
    clean_reply = clean_response(raw_text)

    # Step 6: Extract booking entities via a SEPARATE silent call.
    # This is completely independent of the conversational reply — the extraction
    # model only reads the user's message and returns pure JSON. This is reliable
    # even when the main response contains no XML tags at all.
    booking_update = None
    in_booking_flow = is_booking_message(message, booking_state)
    if in_booking_flow and booking_state and booking_state.get("stage") != "booked":
        booking_update = await extract_booking_entities(message, booking_state, language)

    return {
        "reply": clean_reply,
        "booking_update": booking_update,
        "fd_cards": fd_cards,
        "retrieved_context_used": use_rag and len(retrieved_chunks) > 0
    }
