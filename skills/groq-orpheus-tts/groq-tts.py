import os
import sys
import json
import requests
import subprocess
import time

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 groq-tts.py \"Text\" output.mp3 [voice] [lang: ar|en]")
        sys.exit(1)

    text = sys.argv[1]
    out_mp3 = sys.argv[2]
    voice = sys.argv[3] if len(sys.argv) > 3 else "fahad"
    lang = sys.argv[4] if len(sys.argv) > 4 else "ar"
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        # Fallback for local testing if not in env
        # But for the skill, it must come from env
        print("Error: GROQ_API_KEY environment variable is not set.")
        sys.exit(1)

    temp_wav = f"/tmp/groq_temp_{int(time.time())}.wav"

    model = "canopylabs/orpheus-arabic-saudi"
    if lang == "en":
        model = "canopylabs/orpheus-v1-english"
        if voice == "fahad":
            voice = "troy"

    url = "https://api.groq.com/openai/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": "wav"
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            with open(temp_wav, "wb") as f:
                f.write(response.content)
        else:
            print(f"FAILED: API returned status {response.status_code}")
            print(response.text)
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1)

    if os.path.exists(temp_wav) and os.path.getsize(temp_wav) > 0:
        try:
            # Convert to MP3
            subprocess.run(["ffmpeg", "-i", temp_wav, "-acodec", "libmp3lame", "-y", out_mp3], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            os.remove(temp_wav)
            print(f"SUCCESS: {out_mp3}")
        except Exception as e:
            print(f"ERROR during conversion: {str(e)}")
            sys.exit(1)
    else:
        print("FAILED: Temporary WAV file is empty.")
        sys.exit(1)

if __name__ == "__main__":
    main()
