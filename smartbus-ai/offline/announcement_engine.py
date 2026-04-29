"""
announcement_engine.py — Offline TTS via pyttsx3
Handles English + Kannada announcements over Raspberry Pi audio output.

Why pyttsx3?
  - 100% offline — no internet required (unlike gTTS which needs Google)
  - Pure Python, lightweight (~2MB)
  - Works with espeak-ng backend on Linux/Pi
  - Perfect for demo reliability at exhibitions

Kannada support:
  - espeak-ng has 'kn' (Kannada) voice
  - Install: sudo apt install espeak-ng espeak-ng-data
  - Quality isn't perfect but functional and offline

Smart trigger rules:
  - Only announce when bus is near OR crowd crosses threshold
  - Rate-limit: no announcement within 60s of last one per stop
  - Queue-based: never interrupt a playing announcement
"""

import threading
import time
import logging
from datetime import datetime
from queue import Queue, Empty

log = logging.getLogger('Announce')

# Minimum gap between announcements for same stop (seconds)
MIN_INTERVAL_S = 60

# Supported languages and their espeak voice codes
VOICES = {
    'en': 'en',
    'kn': 'kn',   # Kannada
}


class AnnouncementEngine:
    def __init__(self):
        self._queue     = Queue(maxsize=10)
        self._last_ts   = {}          # stop_id → last announcement time
        self._engine    = None
        self._thread    = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._init_engine()

    # ── Init pyttsx3
    def _init_engine(self):
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', 145)      # words/min — clear pace
            self._engine.setProperty('volume', 0.95)
            # Try to set a good voice
            voices = self._engine.getProperty('voices')
            for v in voices:
                if 'english' in v.name.lower() or 'en' in v.id.lower():
                    self._engine.setProperty('voice', v.id)
                    break
            log.info('pyttsx3 TTS engine initialised')
        except Exception as e:
            log.warning(f'pyttsx3 unavailable: {e}. Announcements will be logged only.')
            self._engine = None

    # ── Public API

    def speak(self, text: str, lang: str = 'en', stop_id: str = None, force: bool = False):
        """Queue an announcement. Respects rate-limiting unless force=True."""
        if stop_id and not force:
            last = self._last_ts.get(stop_id, 0)
            if time.time() - last < MIN_INTERVAL_S:
                log.debug(f'Skipping announcement for {stop_id} — rate limited')
                return
        if stop_id:
            self._last_ts[stop_id] = time.time()
        try:
            self._queue.put_nowait({'text': text, 'lang': lang})
        except Exception:
            log.warning('Announcement queue full — dropping message')

    def announce_crowd(self, stop_id: str, count: int, seats: int):
        en_text = f"Attention. Current crowd at this stop: {count} people. Estimated {seats} seats available on the next bus."
        kn_text = f"ಗಮನಿಸಿ. ಈ ನಿಲ್ದಾಣದಲ್ಲಿ {count} ಜನ ಇದ್ದಾರೆ. ಮುಂದಿನ ಬಸ್‌ನಲ್ಲಿ ಸುಮಾರು {seats} ಆಸನಗಳಿವೆ."
        self.speak(en_text, 'en', stop_id)
        self.speak(kn_text, 'kn', stop_id)

    def announce_bus_arriving(self, stop_id: str, bus_id: str, eta_min: int, seats: int):
        en_text = f"Bus {bus_id} arriving in {eta_min} minutes. {seats} seats available."
        kn_text = f"ಬಸ್ {bus_id} {eta_min} ನಿಮಿಷದಲ್ಲಿ ಬರುತ್ತದೆ. {seats} ಆಸನಗಳು ಲಭ್ಯ."
        self.speak(en_text, 'en', stop_id)
        self.speak(kn_text, 'kn', stop_id)

    def announce_overcrowding(self, stop_id: str, count: int):
        en_text = f"Warning. Overcrowding detected. {count} people at stop. Additional bus being dispatched."
        kn_text = f"ಎಚ್ಚರಿಕೆ. ಹೆಚ್ಚಿನ ಜನಸಂದಣಿ. {count} ಜನ ಇದ್ದಾರೆ. ಹೆಚ್ಚುವರಿ ಬಸ್ ಕಳುಹಿಸಲಾಗುತ್ತಿದೆ."
        self.speak(en_text, 'en', stop_id, force=True)
        self.speak(kn_text, 'kn', stop_id, force=True)

    def announce_dispatch(self, stop_id: str, result: dict):
        bus   = result.get('bus_id', 'a bus')
        eta   = result.get('eta_min', '?')
        en_text = f"Dispatcher action. Extra bus {bus} assigned. ETA {eta} minutes."
        self.speak(en_text, 'en', force=True)

    def announce_signal_received(self, stop_id: str):
        en_text = "Your bus request has been received. Depot has been notified."
        kn_text = "ನಿಮ್ಮ ಬಸ್ ವಿನಂತಿ ಸ್ವೀಕರಿಸಲಾಗಿದೆ. ಡಿಪೋಗೆ ತಿಳಿಸಲಾಗಿದೆ."
        self.speak(en_text, 'en', stop_id)
        self.speak(kn_text, 'kn', stop_id)

    # ── Worker thread (serial TTS playback)
    def _worker(self):
        while True:
            try:
                item = self._queue.get(timeout=1)
                self._play(item['text'], item['lang'])
                self._queue.task_done()
            except Empty:
                continue
            except Exception as e:
                log.error(f'TTS worker error: {e}')

    def _play(self, text: str, lang: str):
        log.info(f'[TTS/{lang}] {text}')
        if self._engine is None:
            return  # silent mode
        try:
            import pyttsx3
            # Switch voice for language if needed
            if lang == 'kn':
                voices = self._engine.getProperty('voices')
                for v in voices:
                    if 'kn' in v.id.lower() or 'kannada' in v.name.lower():
                        self._engine.setProperty('voice', v.id)
                        break
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as e:
            log.error(f'TTS playback error: {e}')
