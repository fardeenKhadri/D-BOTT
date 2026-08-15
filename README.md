# D-BOT — Doctor's Bot for Operational Trackers

AI-powered medical assistant that records patient consultations, extracts
structured patient details with AI, stores them for future reference, and
answers clinical questions about the stored records — all from a clean web
interface.

## Features

- **Voice-to-text patient recording** — captures a consultation from the
  microphone and transcribes it with Whisper.
- **AI medical summarization** — Gemini extracts patient name, age, gender,
  estimated disease, symptoms, and history.
- **CSV data storage** — every consultation is persisted with a unique ID and
  timestamp for future retrieval.
- **Smart clinical chat** — ask questions in text or by voice; relevant
  patient records are used as context.
- **Dark-mode interface** — responsive Bootstrap-based UI.

## Architecture

```
Browser / Flask UI                    Server (Flask / PyAudio)
┌──────────────────────┐   /record_patient  ┌──────────────────────────────────────┐
│  index.html          │ ─────────────────▶ │ record_audio  → PyAudio (WAV)        │
│  (jQuery + Bootstrap)│                   │ transcribe    → Groq Whisper          │
│                      │ ◀───────────────── │ summarize     → Google Gemini         │
└──────────────────────┘    JSON result     │ persist       → patient_data.csv      │
                                            └──────────────────────────────────────┘
```

| Step | Responsibility | Tool |
|------|----------------|------|
| Audio capture | Record microphone to WAV | PyAudio |
| Speech-to-text | Transcribe the WAV | Groq Whisper |
| Data extraction | Summarize + extract patient fields | Google Gemini |
| Chat reasoning | Answer questions on patient data | Groq Llama 3.3 |
| Persistence | Append records to CSV | pandas |

## Project structure

```
D-BOT/
├── app.py              # Flask application (routes + business logic)
├── config.py           # Centralised configuration (reads .env)
├── requirements.txt    # Python dependencies
├── .env.example        # Template for environment variables
├── .env                # Local secrets (git-ignored, never committed)
├── templates/
│   └── index.html      # Web interface
├── static/
│   ├── style.css       # Interface styles
│   └── care.png        # Brand logo / favicon
└── patient_data.csv    # Patient dataset (schema created automatically)
```

## Getting started

### Prerequisites

- Python 3.9+
- A working microphone on the machine running the server
- API keys from [Groq](https://groq.com/) and [Google AI Studio](https://ai.google.dev/)

### Installation

```sh
git clone https://github.com/fardeenKhadri/D-BOTT.git
cd D-BOTT

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

Copy the `.env.example` template and add your API keys:

```sh
cp .env.example .env
```

```dotenv
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
```

Optional variables are listed in `.env.example`. The `.env` file is
git-ignored and must never be committed.

### Running

```sh
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

For production, use a WSGI server:

```sh
gunicorn app:app
```

## Usage

1. **Add patient data** — click the button, speak into the microphone, and the AI
   extracts the patient details into the dataset.
2. **Ask D-BOT (voice)** — click the mic button and ask a question out loud.
3. **Ask D-BOT (text)** — type a question in the chat box and hit send.

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Serve the web interface |
| `GET`  | `/health` | Service health check |
| `POST` | `/record_patient` | Record, transcribe, summarize, and store a patient consultation |
| `POST` | `/record_chat` | Record a spoken question and return an answer |
| `POST` | `/chat_text` | Submit a JSON `{"message": "..."}` and return an answer |

## Security note

This project is a demonstration and is **not** HIPAA-compliant. Patient data
is stored in plain CSV, and the server records audio via its local microphone.
Rotate the API keys in `.env` regularly, and never commit the `.env` file.

## Technologies

- **Backend** — Flask
- **Audio** — PyAudio, Wave
- **AI** — Google Gemini, Groq (Whisper, Llama 3.3)
- **Data** — pandas
- **Frontend** — Bootstrap, jQuery