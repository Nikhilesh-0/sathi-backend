from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import chat, history

app = FastAPI(title="Sathi FD Advisor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten to your Vercel URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api/v1")
app.include_router(history.router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok", "service": "Sathi FD Advisor"}
