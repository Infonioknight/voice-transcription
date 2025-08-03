from fastapi import FastAPI, Request
import aiohttp
import os
import whisper
from langgraph.graph import StateGraph
from langgraph.graph import END
import google.generativeai as genai
from typing import TypedDict, Optional
import csv, json
from dotenv import load_dotenv

# --- Config ---
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_KEY)

TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

app = FastAPI()
VOICE_DIR = "voice_notes"
TRANSCRIPT_DIR = "transcriptions"
CSV_PATH = "data/output.csv"

os.makedirs(VOICE_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

model = whisper.load_model("base")  # options: tiny, base, small, medium, large
print("[✓] Whisper model loaded")

# --- Step 1: Save Audio ---
async def save_audio(state):
    print("[→] save_audio() starting...")
    file_path = state["file_path"]
    media_url = state["media_url"]
    print(f"[•] Downloading from {media_url} to {file_path}")
    try:
        async with aiohttp.ClientSession(auth=aiohttp.BasicAuth(state["sid"], state["token"])) as session:
            async with session.get(media_url) as resp:
                if resp.status == 200:
                    with open(file_path, "wb") as f:
                        f.write(await resp.read())
                    print(f"[✓] Audio saved to {file_path}")
                else:
                    print(f"[×] Failed to download: HTTP {resp.status}")
    except Exception as e:
        print(f"[×] Exception in save_audio: {e}")
    return state

# --- Step 2: Transcribe ---
async def transcribe_audio(state):
    print("[→] transcribe_audio() starting...")
    audio_path = state["file_path"]
    message_id = state["message_sid"]
    transcript_path = os.path.join(TRANSCRIPT_DIR, f"{message_id}.txt")

    try:
        print(f"[•] Transcribing {audio_path}")
        result = model.transcribe(audio_path)
        with open(transcript_path, "w") as f:
            f.write(result["text"])
        print(f"[✓] Transcription saved to {transcript_path}")
        state["transcript_path"] = transcript_path
        state["transcript"] = result["text"]
    except Exception as e:
        print(f"[×] Exception in transcribe_audio: {e}")
    return state

# --- Step 3: Gemini call ---
async def query_gemini(state):
    print("[→] query_gemini() starting...")
    prompt = f"""You are an expert analyst who, when given a transcript of a conversation, can perfectly extract the necessary details and fields with utmost accuracy, based off a pre-provided set of requirements/templates. 

    From the transcript: {state['transcript']}, extract all of the following:
    1. Name of the person
    2. Their phone number
    3. Shoulder width 
    4. Waist
    5. Height

    Once you extract all of these, give them to me as an output, strictly in JSON format. Ensure your output has no before text or after text, just exclusively the JSON I request from you.

    Output:
    ```json
    {{
        'Name': 'Name of the person'
        'Phone': 'Phone number'
        'Shoulder_Width': 'shoulder width'
        'Waist': 'waist'
        'Height': 'height'
    }}
    ```
    I repeat once more for utmost emphasis, I do no want any extra text, explanation or justifications from you, just exclusively my output json.

"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        print(f"[•] Sending to Gemini: {prompt[:50]}...")
        response = model.generate_content(prompt)
        state["gemini_response"] = response.text.strip()
        print(f"[✓] Gemini response: {state['gemini_response'][:50]}...")
    except Exception as e:
        print(f"[×] Exception in query_gemini: {e}")
    return state

async def store_information(state):
    print("→ store_information()")
    gemini_response = state.get("gemini_response", "")
    if not gemini_response:
        print("⚠️ No Gemini data to store")
        return state

    try:
        data = json.loads(gemini_response[8:-4])
    except json.JSONDecodeError:
        print("❌ Failed to parse Gemini JSON:", gemini_response)
        return state

    # Normalize keys to lowercase
    normalized = {k.lower(): v for k, v in data.items()}

    # Ensure consistent column order
    fieldnames = sorted(normalized.keys())

    # Create CSV file if not exists (with header) or append
    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(normalized)

    print(f"✱ Stored row in {CSV_PATH}: {normalized}")
    return state

# --- LangGraph schema ---
class State(TypedDict):
    media_url: str
    file_path: str
    message_sid: str
    sid: str
    token: str
    transcript_path: Optional[str]
    transcript: Optional[str]
    gemini_response: Optional[str]

graph = StateGraph(State)
graph.add_node("save_audio", save_audio, is_async=True)
graph.add_node("transcribe", transcribe_audio, is_async=True)
graph.add_node("gemini", query_gemini, is_async=True)
graph.add_node("store_information", store_information, is_async=True)

graph.set_entry_point("save_audio")
graph.add_edge("save_audio", "transcribe")
graph.add_edge("transcribe", "gemini")
graph.add_edge("gemini", "store_information")
graph.set_finish_point("store_information")

flow = graph.compile()

# --- FastAPI endpoint ---
@app.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    print("[→] Incoming WhatsApp request...")
    form = await request.form()
    
    media_url = form.get("MediaUrl0")
    media_type = form.get("MediaContentType0")
    message_sid = form.get("MessageSid")

    print(f"[•] Media URL: {media_url}")
    print(f"[•] Media Type: {media_type}")
    print(f"[•] Message SID: {message_sid}")

    if media_url and media_type and media_type.startswith("audio"):
        file_ext = media_type.split("/")[-1]
        file_path = os.path.join(VOICE_DIR, f"{message_sid}.{file_ext}")
        
        input_state = {
            "media_url": media_url,
            "file_path": file_path,
            "message_sid": message_sid,
            "sid": TWILIO_SID,
            "token": TWILIO_AUTH_TOKEN
        }

        print("[→] Running LangGraph flow...")
        try:
            result = await flow.ainvoke(input_state)
            print("[✓] All nodes have run")
            return {
                "transcript_file": result.get("transcript_path"),
                "gemini_response": result.get("gemini_response")
            }
        except Exception as e:
            print(f"[×] Exception during LangGraph run: {e}")
            return {"error": "LangGraph failed", "details": str(e)}

    print("[!] No valid audio file found")
    return {"error": "No valid audio file found"}