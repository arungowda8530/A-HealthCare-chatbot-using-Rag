import io
from gtts import gTTS
from fastapi.responses import StreamingResponse
from fastapi import HTTPException

# Supported languages in gTTS
LANGUAGE_MAP = {
    "english": "en",
    "hindi": "hi",
    "kannada": "kn",
    "tamil": "ta",
    "telugu": "te",
    "malayalam": "ml",
}

def generate_audio(text: str, language: str = "english"):
    """Generate audio from text in the specified language using gTTS."""
    try:
        # Validate language input
        lang_code = LANGUAGE_MAP.get(language.lower())
        if not lang_code:
            raise ValueError(f"Unsupported language: {language}. Choose from {list(LANGUAGE_MAP.keys())}")

        # Create gTTS object
        tts = gTTS(text=text, lang=lang_code, slow=False)

        # Save to an in-memory buffer
        audio_buffer = io.BytesIO()
        tts.save("temp.mp3")  # Save as MP3
        with open("temp.mp3", "rb") as f:
            audio_buffer.write(f.read())

        # Prepare buffer for streaming
        audio_buffer.seek(0)
        return audio_buffer

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation error: {str(e)}")
