import os
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from services.chat_service import (
    process_uploaded_csv,
    get_text_reply,
    should_generate_chart   # ✅ ADDED THIS
)
from services.chart_service import (
    get_chart_code_from_gpt,
    run_generated_chart_code
)
from models.schemas import ChatRequest, ChatResponse, ChartRequest, ChartResponse

router = APIRouter()

# Health check
@router.get("/test")
def test_chat_api():
    return {"message": "Chat route working fine!"}

# Upload CSV file
@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
    
    try:
        result = await process_uploaded_csv(file)
        return {"message": "File uploaded and processed successfully", "columns": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Ask GPT any general question
@router.post("/ask", response_model=ChatResponse)
async def ask_gpt(request: ChatRequest):
    reply = get_text_reply(request.message)
    return ChatResponse(reply=reply)

# Generate chart from user query
@router.post("/generate-chart", response_model=ChartResponse)
async def generate_chart(request: ChartRequest):
    try:
        # Step 0: Use GPT to decide if chart is required
        if not should_generate_chart(request.prompt):
            raise HTTPException(status_code=400, detail="Prompt is not suitable for chart generation.")

        # Load latest uploaded CSV from backend/data/
        latest_file = sorted(
            [f for f in os.listdir("backend/data") if f.endswith(".csv")],
            key=lambda x: os.path.getmtime(os.path.join("backend/data", x)),
            reverse=True
        )[0]

        df = pd.read_csv(os.path.join("backend/data", latest_file))
        df_preview = df.head().to_string(index=False)
        columns = df.columns.tolist()

        # Step 1: Ask GPT to write chart code
        code = get_chart_code_from_gpt(request.prompt, df_preview, columns)

        # Step 2: Run generated code and save chart image
        image_path = run_generated_chart_code(code, df)

        if not image_path or not image_path.endswith(".png"):
            raise HTTPException(status_code=500, detail=image_path or "Unknown error")

        return ChartResponse(image_url=image_path.replace("\\", "/"))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart generation failed: {str(e)}")
