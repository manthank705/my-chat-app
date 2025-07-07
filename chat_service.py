import openai
import os
import uuid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import re
import pandas as pd
from fastapi import UploadFile

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

def should_generate_chart(prompt: str) -> bool:
    """
    Ask GPT if the prompt is requesting a chart or not.
    Returns True if a chart is requested, otherwise False.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You're an AI that classifies prompts as chart requests or general queries. Reply only 'yes' if the prompt clearly asks for data visualization, otherwise reply 'no'."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=3,
    )

    reply = response["choices"][0]["message"]["content"].strip().lower()
    return "yes" in reply

def get_chart_code_from_gpt(prompt: str, df_preview: str, columns: list) -> str:
    """
    Ask GPT to generate Python code for a chart based on the user prompt and file columns.
    """
    system_prompt = (
        "You are a data analyst. Generate Python code using matplotlib to create a chart.\n"
        "Don't use plt.show(). Instead, save the chart as 'output.png' using plt.savefig().\n"
        "Assume 'df' is a Pandas DataFrame already loaded from the uploaded CSV file.\n"
        "ONLY return valid executable Python code — no text, no explanation, no markdown.\n"
    )

    user_prompt = (
        f"The DataFrame 'df' has the following columns: {columns}\n"
        f"Here's a preview of the data:\n{df_preview}\n"
        f"Now generate the chart for this request:\n{prompt}"
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.5,
        max_tokens=700
    )

    raw_code = response["choices"][0]["message"]["content"]
    return clean_code(raw_code)

def clean_code(code: str) -> str:
    """
    Remove markdown formatting and GPT-inserted plt.savefig calls.
    """
    cleaned = re.sub(r"```(?:python)?", "", code, flags=re.IGNORECASE).strip()
    cleaned = cleaned.replace("```", "").strip()
    cleaned = re.sub(r"plt\.savefig\(.+\)", "", cleaned)
    return cleaned

def run_generated_chart_code(code: str, df) -> str:
    """
    Safely execute generated code and save the chart image.
    Returns the file path of the saved image.
    """
    image_id = str(uuid.uuid4())
    image_path = os.path.join(UPLOAD_DIR, f"{image_id}.png")
    exec_env = {"df": df, "plt": plt}

    try:
        print("=== EXECUTING GPT CODE ===")
        print(code)

        os.makedirs(UPLOAD_DIR, exist_ok=True)
        exec(code, exec_env)
        print(f"Saving image to: {image_path}")
        plt.savefig(image_path)
        plt.clf()

        if os.path.exists(image_path):
            print(f"✅ Image saved successfully: {image_path}")
            return os.path.basename(image_path)
        else:
            print(f"❌ Image NOT saved: {image_path}")
            return None

    except Exception as e:
        return f"❌ Error in generated code: {e}"

def get_text_reply(prompt: str) -> str:
    """
    Returns a natural language reply for general prompts.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You're a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=200
    )
    return response["choices"][0]["message"]["content"].strip()

async def process_uploaded_csv(file: UploadFile):
    """
    Saves the uploaded CSV file and returns its column names.
    """
    contents = await file.read()
    filename = file.filename
    save_path = os.path.join(UPLOAD_DIR, filename)

    with open(save_path, "wb") as f:
        f.write(contents)

    df = pd.read_csv(save_path)
    return df.columns.tolist()
