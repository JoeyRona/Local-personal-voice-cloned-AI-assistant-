"""
Voice assistant that answers in a cloned voice. Runs fully on your own PC,
on the CPU only (no graphics card needed).

    you speak  ->  Whisper (speech to text)
               ->  Ollama  (the "brain", a small local language model)
               ->  Pocket TTS (speaks the reply in the cloned voice)

Only clone a voice with that person's clear permission. Have your friend
record the sample themselves, knowing what it is for.

Run it:            python voice_assistant.py
Pick a mic:        python voice_assistant.py --list-devices
                   python voice_assistant.py --mic 3 --speaker 5
"""

import os

# Windows: Whisper and PyTorch each ship the same Intel maths library, and
# loading both normally crashes with "OMP: Error #15". This allows it.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse
import json
import queue
import re
import sys
import threading
import time
from math import gcd
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
from scipy.signal import resample_poly

# ----------------------------- settings ------------------------------------

VOICE_SAMPLE = "friend.wav"   # your friend's recording, in the same folder
LLM_MODEL = "llama3.2"        # low on memory or too slow? try "llama3.2:1b"
WHISPER_MODEL = "base.en"     # "tiny.en" is faster, "small.en" is more accurate
OLLAMA_URL = "http://127.0.0.1:11434"

# Change this to give the assistant a name and personality.
PERSONALITY = (
    "You are a friendly, funny voice assistant. Your replies are read out "
    "loud, so talk the way people talk: one to three short sentences, plain "
    "words, no lists, no markdown, no emoji."
)

STREAM_AUDIO = True   # set to False if the speech stutters or breaks up
MEMORY_TURNS = 8      # how many past exchanges the assistant remembers

# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
WHISPER_RATE = 16000

VOICE_ACCESS_HELP = """
Voice cloning isn't unlocked yet. It is free, but you have to agree to the
model's terms once:

  1. Make a free account at https://huggingface.co
  2. Open https://huggingface.co/kyutai/pocket-tts and accept the terms
  3. Make a token at https://huggingface.co/settings/tokens  (type: Read)
  4. In this terminal run:   hf auth login    and paste the token
  5. Run this script again
"""


def resample(audio, src_rate, dst_rate):
    """Change the sample rate of a mono float32 signal."""
    if src_rate == dst_rate or len(audio) == 0:
        return audio.astype(np.float32)
    g = gcd(int(src_rate), int(dst_rate))
    return resample_poly(audio, int(dst_rate) // g, int(src_rate) // g).astype(np.float32)


# ----------------------------- hearing -------------------------------------

def record(mic):
    """Record from the microphone until Enter is pressed. Returns 16 kHz mono."""
    rate = int(sd.query_devices(mic, "input")["default_samplerate"])
    chunks = []

    def on_audio(indata, frames, time_info, status):
        chunks.append(indata.copy())

    with sd.InputStream(device=mic, samplerate=rate, channels=1,
                        dtype="float32", callback=on_audio):
        input("  Listening... press Enter when you've finished talking. ")

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return resample(np.concatenate(chunks)[:, 0], rate, WHISPER_RATE)


def transcribe(whisper, audio):
    language = "en" if WHISPER_MODEL.endswith(".en") else None
    segments, _ = whisper.transcribe(
        audio, language=language, beam_size=1, vad_filter=True,
        condition_on_previous_text=False,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


# ----------------------------- thinking ------------------------------------

def check_ollama():
    """Make sure Ollama is running and the model is downloaded, then load it."""
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()
    except requests.RequestException:
        sys.exit("\nCan't reach Ollama. Install it from https://ollama.com, "
                 "make sure it is running, then try again.")
    installed = [m["name"] for m in tags.get("models", [])]
    wanted = LLM_MODEL if ":" in LLM_MODEL else LLM_MODEL + ":latest"
    if wanted not in installed:
        sys.exit(f"\nThe model '{LLM_MODEL}' isn't downloaded yet. Run:\n\n"
                 f"    ollama pull {LLM_MODEL}\n")
    # An empty chat just loads the model into memory so the first reply is quick.
    requests.post(f"{OLLAMA_URL}/api/chat", timeout=300,
                  json={"model": LLM_MODEL, "messages": [], "keep_alive": "30m"})


def ask(messages):
    """Send the conversation to the model and yield its reply piece by piece."""
    response = requests.post(
        f"{OLLAMA_URL}/api/chat", stream=True, timeout=(5, 300),
        json={"model": LLM_MODEL, "messages": messages, "stream": True,
              "keep_alive": "30m"},
    )
    response.raise_for_status()
    for line in response.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        if "error" in data:
            raise requests.RequestException(data["error"])
        piece = data.get("message", {}).get("content", "")
        if piece:
            yield piece
        if data.get("done"):
            break


SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
NOT_AN_ENDING = re.compile(r"\b(?:e\.g|i\.e|Mr|Mrs|Ms|Dr|St|vs)\.$")
MIN_SENTENCE_CHARS = 25


def pop_sentences(buffer):
    """Split finished sentences off the front of `buffer`.

    Returns (sentences, leftover). Very short sentences are kept together
    with the next one so the speech doesn't come out choppy.
    """
    sentences, start = [], 0
    for match in SENTENCE_END.finditer(buffer):
        piece = buffer[start:match.start()].strip()
        if len(piece) >= MIN_SENTENCE_CHARS and not NOT_AN_ENDING.search(piece):
            sentences.append(piece)
            start = match.end()
    return sentences, buffer[start:]


def clean_for_speech(text):
    """Remove things a voice shouldn't try to read (markdown, emoji)."""
    text = re.sub(r"[*_#`~>|\[\]]", "", text)
    text = re.sub("[\U0001F000-\U0001FAFF☀-➿️‍]", "", text)
    return re.sub(r"\s+", " ", text).strip()


# ----------------------------- speaking ------------------------------------

def voice_files(sample_path):
    """Where the recording is, and where its processed copy gets saved."""
    sample = Path(sample_path)
    if not sample.is_absolute():
        sample = HERE / sample
    cache = sample.with_suffix(".voice.safetensors")
    if not sample.exists() and not cache.exists():
        sys.exit(f"\nCan't find the voice recording: {sample}\n"
                 f"Put your friend's recording there (WAV or MP3).")
    return sample, cache


def load_voice(tts, sample, cache):
    """Turn the voice recording into a 'voice state' the model can speak with.

    This is slow the first time, so the result is saved next to the recording
    and reused until the recording changes.
    """
    from pocket_tts import export_model_state

    if cache.exists() and (not sample.exists()
                           or cache.stat().st_mtime >= sample.stat().st_mtime):
        return tts.get_state_for_audio_prompt(cache)
    if not tts.has_voice_cloning:
        sys.exit(VOICE_ACCESS_HELP)

    print("  Learning the voice from the recording (first time only)...")
    voice = tts.get_state_for_audio_prompt(sample, truncate=True)
    export_model_state(voice, cache)
    return voice


class Speaker:
    """Speaks sentences one after another on a background thread."""

    def __init__(self, tts, voice, device=None):
        self.tts, self.voice = tts, voice
        self.rate = tts.sample_rate
        self.sentences = queue.Queue()
        self.stop = threading.Event()

        # Some sound cards refuse the model's 24 kHz. Then use the card's own
        # rate and convert each sentence before playing it.
        self.out_rate = self.rate
        try:
            sd.check_output_settings(device=device, samplerate=self.rate,
                                     channels=1, dtype="float32")
        except Exception:
            self.out_rate = int(sd.query_devices(device, "output")["default_samplerate"])
        self.stream = sd.OutputStream(device=device, samplerate=self.out_rate,
                                      channels=1, dtype="float32")
        self.stream.start()
        threading.Thread(target=self._run, daemon=True).start()

    def say(self, text):
        text = clean_for_speech(text)
        if re.search(r"\w", text):
            self.sentences.put(text)

    def wait(self):
        """Block until everything queued has been spoken (Ctrl+C still works)."""
        while self.sentences.unfinished_tasks:
            time.sleep(0.05)

    def cancel(self):
        """Stop talking right now and forget anything still queued."""
        self.stop.set()
        while True:
            try:
                self.sentences.get_nowait()
                self.sentences.task_done()
            except queue.Empty:
                break
        self.wait()
        self.stop.clear()

    def _run(self):
        while True:
            text = self.sentences.get()
            try:
                if not self.stop.is_set():
                    self._speak(text)
            except Exception as error:
                print(f"\n  (couldn't say that part: {error})")
            finally:
                self.sentences.task_done()

    def _speak(self, text):
        chunks = self.tts.generate_audio_stream(self.voice, text, stop=self.stop)
        if STREAM_AUDIO and self.out_rate == self.rate:
            for chunk in chunks:              # play each piece as it is made
                if self.stop.is_set():
                    break
                self._play(chunk.numpy())
        else:                                 # make the whole sentence first
            pieces = [chunk.numpy() for chunk in chunks]
            if pieces and not self.stop.is_set():
                self._play(resample(np.concatenate(pieces), self.rate, self.out_rate))

    def _play(self, samples):
        self.stream.write(np.ascontiguousarray(samples, dtype=np.float32).reshape(-1, 1))


# ----------------------------- main loop -----------------------------------

def main():
    global LLM_MODEL

    parser = argparse.ArgumentParser(description="Local voice assistant with a cloned voice")
    parser.add_argument("--list-devices", action="store_true",
                        help="show microphones and speakers, then exit")
    parser.add_argument("--mic", type=int, help="microphone number from --list-devices")
    parser.add_argument("--speaker", type=int, help="speaker number from --list-devices")
    parser.add_argument("--voice", default=VOICE_SAMPLE, help="voice recording to clone")
    parser.add_argument("--model", default=LLM_MODEL, help="Ollama model to use")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        print("\n'>' marks the default microphone, '<' the default speaker.")
        return

    LLM_MODEL = args.model
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # don't crash on odd characters

    sample, cache = voice_files(args.voice)

    print("Starting up (the first run downloads the models, so give it a few minutes)")
    print("  Checking Ollama...")
    check_ollama()

    print("  Loading speech recognition...")
    from faster_whisper import WhisperModel
    whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    print("  Loading the voice...")
    from pocket_tts import TTSModel
    tts = TTSModel.load_model()
    voice = load_voice(tts, sample, cache)
    speaker = Speaker(tts, voice, args.speaker)

    system = {"role": "system", "content": PERSONALITY}
    history = []
    print("\nReady. Ctrl+C while it is talking cuts it off.")

    while True:
        try:
            typed = input("\nPress Enter to talk (or type a message, or 'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if typed.lower() in {"quit", "exit"}:
            break

        if typed:
            user_text = typed
        else:
            try:
                audio = record(args.mic)
            except (EOFError, KeyboardInterrupt):
                break
            except Exception as error:
                print(f"  Microphone problem: {error}\n"
                      f"  Run with --list-devices and pick one with --mic NUMBER.")
                continue
            if len(audio) < WHISPER_RATE * 0.3:
                print("  That was too short, try again.")
                continue
            user_text = transcribe(whisper, audio)
            if not user_text:
                print("  I didn't hear any words. Check the mic with --list-devices.")
                continue
            print(f"You: {user_text}")

        history.append({"role": "user", "content": user_text})
        reply, unspoken = "", ""
        print("Assistant: ", end="", flush=True)
        try:
            for piece in ask([system] + history[-2 * MEMORY_TURNS:]):
                print(piece, end="", flush=True)
                reply += piece
                unspoken += piece
                sentences, unspoken = pop_sentences(unspoken)
                for sentence in sentences:
                    speaker.say(sentence)
            speaker.say(unspoken)
            print()
            speaker.wait()
        except KeyboardInterrupt:
            speaker.cancel()
            print("\n  (stopped)")
        except requests.RequestException as error:
            speaker.cancel()
            print(f"\n  Ollama problem: {error}")
        if reply.strip():
            history.append({"role": "assistant", "content": reply})
        else:
            history.pop()

    print("\nBye!")


if __name__ == "__main__":
    main()
