"""D-BOT: Doctor's Bot for Operational Trackers.

An AI-powered medical assistant that helps doctors and healthcare
professionals record, analyze, and retrieve patient data.

Workflow
--------
1. **Record**   — PyAudio captures a short clip from the server microphone.
2. **Transcribe** — Groq Whisper converts the speech into text.
3. **Summarize**  — Google Gemini extracts structured patient details.
4. **Persist**    — The structured record is appended to a CSV dataset.
5. **Assist**     — A Llama chat model answers questions about stored records.

All external credentials are read from ``config.Config``, which loads them
from the ``.env`` file (never hard-coded in source).
"""

from __future__ import annotations

import uuid
import wave
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
from google import genai
import pyaudio
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from groq import Groq

from config import Config

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)

gemini_client = genai.Client(api_key=Config.GEMINI_API_KEY)

groq_client = Groq(api_key=Config.GROQ_API_KEY)

CSV_COLUMNS = [
    "UUID",
    "Audio Input",
    "Summary",
    "Patient Name",
    "Age",
    "Gender",
    "Estimated Disease",
    "Symptoms",
    "Patient History",
    "Date of Diagnosis",
    "Timestamp",
]

# Fields that appear in the Gemini summary and are stored in the CSV.
PARSE_FIELDS = (
    "Patient Name",
    "Age",
    "Gender",
    "Estimated Disease",
    "Symptoms",
    "Patient History",
)


def _ensure_dataset_exists() -> None:
    """Create the CSV dataset with the proper schema if it is missing."""
    if not Path(Config.CSV_FILE).exists():
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(Config.CSV_FILE, index=False)


_ensure_dataset_exists()

# ---------------------------------------------------------------------------
# Audio capture
# ---------------------------------------------------------------------------


def record_audio(filename: str, duration: int = Config.PATIENT_RECORD_SECONDS) -> str:
    """Record microphone input and save it as a WAV file.

    Args:
        filename: Destination path for the recorded audio.
        duration: Number of seconds to record.

    Returns:
        The path to the saved WAV file.
    """
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=Config.AUDIO_FORMAT,
        channels=Config.AUDIO_CHANNELS,
        rate=Config.AUDIO_RATE,
        input=True,
        frames_per_buffer=Config.AUDIO_CHUNK,
    )

    frames = []
    try:
        for _ in range(0, int(Config.AUDIO_RATE / Config.AUDIO_CHUNK * duration)):
            frames.append(stream.read(Config.AUDIO_CHUNK))
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

    with wave.open(filename, "wb") as wav_file:
        wav_file.setnchannels(Config.AUDIO_CHANNELS)
        wav_file.setsampwidth(audio.get_sample_size(Config.AUDIO_FORMAT))
        wav_file.setframerate(Config.AUDIO_RATE)
        wav_file.writeframes(b"".join(frames))

    return filename


# ---------------------------------------------------------------------------
# Transcription and summarization
# ---------------------------------------------------------------------------


def transcribe_audio(filename: str) -> str:
    """Convert a WAV file into text using Groq Whisper.

    Args:
        filename: Path to the audio file to transcribe.

    Returns:
        The trimmed transcription text.
    """
    with open(filename, "rb") as audio_file:
        response = groq_client.audio.transcriptions.create(
            file=(filename, audio_file.read()),
            model=Config.TRANSCRIPTION_MODEL,
            response_format="verbose_json",
        )
    return response.text.strip()


def summarize_text(text: str) -> str:
    """Extract structured patient details from a transcription using Gemini.

    Args:
        text: The transcribed patient consultation.

    Returns:
        A human-readable summary containing labelled patient details.
    """
    prompt = (
        "Summarize the following medical text and extract patient details:\n"
        "- Patient Name\n"
        "- Age\n"
        "- Gender\n"
        "- Estimated Disease\n"
        "- Symptoms\n"
        "- Patient History\n\n"
        f"{text}"
    )
    return gemini_client.models.generate_content(
        model=Config.GEMINI_MODEL, contents=prompt
    ).text.strip()


# ---------------------------------------------------------------------------
# Data extraction and persistence
# ---------------------------------------------------------------------------


def extract_patient_details(summary: str) -> Dict[str, str]:
    """Parse labelled fields out of the Gemini summary.

    Values are matched by label (e.g. ``Patient Name:``). Any markdown
    formatting, such as the ``**`` bold markers Gemini tends to emit, is
    stripped from the extracted value.

    Args:
        summary: The raw summary text produced by ``summarize_text``.

    Returns:
        A mapping of field label to extracted value, defaulting to
        ``"Unknown"`` when a field is absent.
    """
    details: Dict[str, str] = {label: "Unknown" for label in PARSE_FIELDS}
    for line in summary.splitlines():
        for label in PARSE_FIELDS:
            if line.lower().startswith(label.lower()):
                value = line.split(":", 1)[-1].strip().strip("*").strip()
                if value:
                    details[label] = value
                break
    return details


def save_to_csv(audio_text: str, summary: str) -> Dict[str, str]:
    """Append a patient record to the CSV dataset.

    Args:
        audio_text: The transcribed patient consultation.
        summary: The Gemini summary containing the structured details.

    Returns:
        A JSON-safe dict describing the saved patient record, used by the
        frontend to render the result.
    """
    details = extract_patient_details(summary)
    now = datetime.now()

    record = pd.DataFrame(
        [
            {
                "UUID": str(uuid.uuid4()),
                "Audio Input": audio_text,
                "Summary": summary,
                "Patient Name": details["Patient Name"],
                "Age": details["Age"],
                "Gender": details["Gender"],
                "Estimated Disease": details["Estimated Disease"],
                "Symptoms": details["Symptoms"],
                "Patient History": details["Patient History"],
                "Date of Diagnosis": now.strftime("%Y-%m-%d"),
                "Timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            }
        ]
    )

    dataset = pd.read_csv(Config.CSV_FILE)
    dataset = pd.concat([dataset, record], ignore_index=True)
    dataset.to_csv(Config.CSV_FILE, index=False)

    return {
        "uuid": record.loc[0, "UUID"],
        "name": details["Patient Name"],
        "age": details["Age"],
        "gender": details["Gender"],
        "disease": details["Estimated Disease"],
        "symptoms": details["Symptoms"],
    }


# ---------------------------------------------------------------------------
# Chat assistant
# ---------------------------------------------------------------------------


def _load_dataset() -> pd.DataFrame:
    """Read the CSV dataset, guarding against a missing file."""
    if not Path(Config.CSV_FILE).exists():
        _ensure_dataset_exists()
    return pd.read_csv(Config.CSV_FILE)


def generate_response(system_prompt: str, user_question: str, dataset: pd.DataFrame) -> str:
    """Answer a question using only the patient records relevant to it.

    Keyword-based relevance filtering is applied first; if nothing matches,
    the most recent records are used as context instead.

    Args:
        system_prompt: The assistant's system instructions.
        user_question: The user's question.
        dataset: The patient dataset to draw context from.

    Returns:
        The assistant's answer as plain text.
    """
    keywords = [word.lower() for word in user_question.split() if word]
    mask = dataset.apply(
        lambda row: any(keyword in row.to_string().lower() for keyword in keywords),
        axis=1,
    )
    relevant = dataset[mask]

    if relevant.empty:
        relevant = dataset.tail(3)

    context = relevant[["Patient Name", "Age", "Estimated Disease", "Symptoms"]].to_string(index=False)

    full_prompt = f"""{system_prompt}

Patient context:
{context}

Question:
{user_question}"""

    response = groq_client.chat.completions.create(
        model=Config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt},
        ],
        temperature=Config.AI_TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def home():
    """Serve the web interface."""
    return render_template("index.html")


@app.route("/health")
def health():
    """Simple health check for uptime monitoring."""
    return jsonify({"status": "ok"})


@app.route("/record_patient", methods=["POST"])
def record_patient():
    """Record a patient consultation and store the extracted details."""
    try:
        record_audio(Config.PATIENT_AUDIO_FILE)
        audio_text = transcribe_audio(Config.PATIENT_AUDIO_FILE)
        summary = summarize_text(audio_text)
        return jsonify(save_to_csv(audio_text, summary))
    except Exception as exc:  # noqa: BLE001 - surface any failure to the client
        return jsonify({"error": str(exc)}), 500


@app.route("/record_chat", methods=["POST"])
def record_chat():
    """Record a spoken question and answer it with the assistant."""
    try:
        record_audio(Config.CHAT_AUDIO_FILE, duration=Config.CHAT_RECORD_SECONDS)
        user_question = transcribe_audio(Config.CHAT_AUDIO_FILE)
        answer = generate_response(
            "You are a helpful AI medical assistant.",
            user_question,
            _load_dataset(),
        )
        return jsonify({"question": user_question, "answer": answer})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.route("/chat_text", methods=["POST"])
def chat_text():
    """Answer a typed question with the assistant."""
    try:
        user_input = request.json.get("message", "")
        answer = generate_response(
            "You are a helpful AI medical assistant.",
            user_input,
            _load_dataset(),
        )
        return jsonify({"answer": answer})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.FLASK_DEBUG)