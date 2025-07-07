# backend/models/schemas.py

from pydantic import BaseModel

# For GPT chat
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

# For chart generation
class ChartRequest(BaseModel):
    prompt: str  # What user wants: e.g., "Compare sales per year"

class ChartResponse(BaseModel):
    image_url: str  # URL/path to generated chart image
