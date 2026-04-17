"""
Booking Service — handles FD booking creation and receipt generation.
In production this would call Blostem's actual API. Here we generate
a realistic mock receipt that looks and feels like a real FD confirmation.
"""

import random
import string
import json
import os
from datetime import datetime, timedelta

_FD_DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/fd_products.json")
with open(_FD_DATA_PATH, "r") as f:
    FD_PRODUCTS = json.load(f)


def generate_reference_number() -> str:
    """Generate a realistic FD reference number."""
    prefix = "SATHI"
    year = datetime.now().year
    random_part = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    return f"{prefix}{year}{random_part}"


def calculate_maturity(principal: float, rate: float, tenor_months: int) -> float:
    """Compound interest calculation (quarterly compounding, standard for FDs)."""
    n = 4  # quarterly
    t = tenor_months / 12
    r = rate / 100
    maturity = principal * (1 + r / n) ** (n * t)
    return round(maturity, 2)


def create_fd_booking(
    principal_amount: float,
    tenor_months: int,
    pan_number: str,
    nominee_name: str,
    fd_id: str,
    user_id: str
) -> dict:
    """
    Creates a mock FD booking and returns a receipt payload.
    The frontend renders this as a styled receipt card.
    """
    fd_map = {fd["id"]: fd for fd in FD_PRODUCTS}
    # Default to highest rate FD if specific one not selected
    fd = fd_map.get(fd_id, FD_PRODUCTS[1])

    booking_date = datetime.now()
    maturity_date = booking_date + timedelta(days=tenor_months * 30.44)
    maturity_amount = calculate_maturity(principal_amount, fd["interest_rate"], tenor_months)
    interest_earned = round(maturity_amount - principal_amount, 2)

    # TDS applicable if interest > 40000/year
    annual_interest = principal_amount * (fd["interest_rate"] / 100)
    tds_applicable = annual_interest > 40000
    tds_amount = round(interest_earned * 0.10, 2) if tds_applicable else 0

    # Determine tenor display string
    if tenor_months < 12:
        tenor_display = f"{tenor_months} months"
    elif tenor_months == 12:
        tenor_display = "1 year"
    elif tenor_months % 12 == 0:
        tenor_display = f"{tenor_months // 12} years"
    else:
        years = tenor_months // 12
        months = tenor_months % 12
        tenor_display = f"{years} year{'s' if years > 1 else ''} {months} months"

    # Mask PAN — show first 3 and last character only
    masked_pan = pan_number[:3] + "XX" + pan_number[5:8] + "X" + pan_number[9] if len(pan_number) == 10 else pan_number

    return {
        "reference_number": generate_reference_number(),
        "status": "CONFIRMED",
        "bank_name": fd["bank_name"],
        "bank_type": fd["bank_type"],
        "principal_amount": principal_amount,
        "interest_rate": fd["interest_rate"],
        "tenor_months": tenor_months,
        "tenor_display": tenor_display,
        "maturity_amount": maturity_amount,
        "interest_earned": interest_earned,
        "tds_applicable": tds_applicable,
        "tds_amount": tds_amount,
        "net_maturity": round(maturity_amount - tds_amount, 2),
        "booking_date": booking_date.strftime("%d %b %Y"),
        "maturity_date": maturity_date.strftime("%d %b %Y"),
        "pan_number": masked_pan,
        "nominee_name": nominee_name,
        "dicgc_insured": fd["dicgc_insured"],
        "user_id": user_id,
    }
