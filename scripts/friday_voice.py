"""FRIDAY voice: hands-free, no button. Talk to her like "Friday, what's the worst incident?"
 
  python scripts/friday_voice.py data/ait/alerts_csv/russellmitchell_alerts.txt [how_many]
 
1. On start FRIDAY investigates the incidents (cached in eval/, so only the first run uses Groq)
   and SPEAKS FIRST if anything is high or critical.
2. Then she listens all the time: mic -> simple voice detection -> Whisper on the Mac GPU (mlx-whisper).
   Anything that starts with "Friday" is a question about the incident queue.
3. Answers come from the queue only, spoken with the Mac's built-in `say`.
Say "Friday, goodbye" to stop.  Needs:  pip install mlx-whisper sounddevice edge-tts"""
import collections, json, os, queue, random, re, subprocess, sys
import numpy as np
import sounddevice as sd
import mlx_whisper
 
SR = 16000                                   # Whisper wants 16 kHz mono
BLOCK = 0.05                                 # seconds of audio per mic block (smaller = reacts faster)
SENS = float(os.getenv("FRIDAY_SENS", "2.0"))   # lower = picks up your voice sooner (more sensitive)
WHISPER = os.getenv("FRIDAY_WHISPER", "mlx-community/whisper-tiny.en-mlx")   # tiny = fastest; base = more accurate
# Two voice engines:
#   neural (default): Microsoft neural voices via edge-tts, sound human, need internet.  pip install edge-tts
#                     list them: edge-tts --list-voices | grep en-
#   say:              the Mac's built-in voices, offline.  list them: say -v '?'
TTS = os.getenv("FRIDAY_TTS", "neural")
VOICE = os.getenv("FRIDAY_VOICE", "en-GB-SoniaNeural" if TTS == "neural" else "Moira")
RATE = os.getenv("FRIDAY_RATE", "+15%" if TTS == "neural" else "205")
CHAT_MODEL = os.getenv("FRIDAY_CHAT_MODEL", "openai/gpt-oss-20b")   # small + fast for spoken replies
 
LIVE = "eval/voice_live.json"                 # the dashboard reads this to show the conversation
_log = []
 
def live(state, who=None, text=None):
    if who:
        _log.append({"who": who, "text": text})
    try:
        json.dump({"state": state, "log": _log[-30:]}, open(LIVE, "w"))
    except OSError:
        pass
 
_noise = 0.01                                 # room loudness, set by calibrate()
# How to cut her off mid-sentence:
#   keypress (default): tap ENTER while she talks -> works on laptop speakers, no earbuds needed
#   mic (FRIDAY_BARGE>0): just talk over her -> only reliable with earbuds, else she hears herself
BARGE = float(os.getenv("FRIDAY_BARGE", "0"))
 
def _synth(text):
    """Make an audio file so playback can be a separate process we can stop mid-sentence."""
    if TTS == "neural":
        out = "/tmp/friday_say.mp3"
        if subprocess.run(["edge-tts", "--voice", VOICE, f"--rate={RATE}", "--text", text,
                           "--write-media", out], capture_output=True).returncode == 0:
            return out
        print("  (neural voice failed, using the Mac voice)")
    out = "/tmp/friday_say.aiff"
    subprocess.run(["say", "-v", "Moira" if TTS == "neural" else VOICE,
                    "-r", "190" if TTS == "neural" else RATE, "-o", out, text])
    return out
 
def speak(text, interruptible=False):
    """Speak. If interruptible you can STOP her mid-sentence: tap ENTER (works on speakers),
    or, with earbuds and FRIDAY_BARGE>0, just talk over her."""
    import select
    print(f"\nFRIDAY: {text}")
    live("speaking", "friday", text)
    proc = subprocess.Popen(["afplay", _synth(text)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if interruptible:
        if BARGE:                                            # mic barge-in (earbuds)
            thresh, hot, q = max(_noise * BARGE, 0.035), 0, queue.Queue()
            try:
                with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=int(SR * BLOCK),
                                    callback=lambda d, *_: q.put(d.copy())):
                    while proc.poll() is None:
                        try: b = q.get(timeout=0.15)
                        except queue.Empty: pass
                        else:
                            hot = hot + 1 if rms(b) > thresh else 0
                            if hot >= 4:
                                proc.terminate(); print("  (you cut in)"); live("listening"); return "interrupted"
                        if select.select([sys.stdin], [], [], 0)[0]:
                            sys.stdin.readline(); proc.terminate(); print("  (stopped)"); live("listening"); return "interrupted"
            except Exception:
                pass
        else:                                                # keypress barge-in (laptop speakers)
            while proc.poll() is None:
                if select.select([sys.stdin], [], [], 0.15)[0]:
                    sys.stdin.readline(); proc.terminate(); print("  (stopped — go ahead)"); live("listening"); return "interrupted"
    proc.wait(); live("listening"); return "done"
 
# ---------- the incident queue (same pipeline as friday_graph.py --incidents) ----------
def build_queue(csv, n):
    cache = f"eval/voice_queue_{os.path.basename(csv).split('.')[0]}_{n}.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    from incidents import build_incidents, incident_text, ts
    from friday_graph import friday, RANK
    print(f"Investigating the {n} largest incidents (first run only)...")
    incidents, _ = build_incidents(csv)
    q = []
    for i, inc in enumerate(incidents[:n], 1):
        out = friday.invoke({"alert": incident_text(inc), "from_incident": True, "quiet": True})
        q.append({"host": inc["host"], "when": ts(inc["start"]), "alerts": sum(inc["alerts"].values()),
                  "severity": str(out.get("severity")), "technique": str(out.get("technique")),
                  "explanation": str(out.get("explanation")),
                  "truth": inc["labels"].most_common(1)[0][0]})            # AIT label, only for checking
        print(f"  {i}/{n} {inc['host']:18} -> {q[-1]['severity']}")
    q.sort(key=lambda r: (RANK.get(r["severity"], 4), -r["alerts"]))
    json.dump(q, open(cache, "w"), indent=1)
    return q
 
def technique_name(tid):
    from attack_lookup import col
    try:
        metas = col.get(ids=[tid])["metadatas"] or [{}]
        return str(metas[0].get("name", "an unknown technique"))
    except Exception:
        return "an unknown technique"
 
def spoken_host(h):
    return h.replace("_", " ").replace("-", " ")
 
# ---------- listening ----------
def rms(block):
    return float(np.sqrt(np.mean(block ** 2)))
 
def calibrate():
    global _noise
    noise = sd.rec(int(SR * 1.0), samplerate=SR, channels=1, dtype="float32"); sd.wait()
    _noise = max(0.004, rms(noise))
    return max(0.008, SENS * _noise)
 
def listen(threshold, max_s=10, silence_s=0.6):
    """Wait until someone speaks, record until they stop, return the audio."""
    q = queue.Queue()
    pre = collections.deque(maxlen=int(0.3 / BLOCK))     # keep 0.3 s before speech so the first word isn't cut
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=int(SR * BLOCK),
                        callback=lambda data, *_: q.put(data.copy())):
        hot = 0
        while True:                                      # wait for speech (2 loud blocks in a row = you started)
            b = q.get()
            hot = hot + 1 if rms(b) > threshold else 0
            if hot >= 2:
                break
            pre.append(b)
        blocks, quiet = list(pre) + [b], 0
        while len(blocks) * BLOCK < max_s:               # record until a pause
            b = q.get(); blocks.append(b)
            quiet = quiet + 1 if rms(b) < threshold else 0
            if quiet * BLOCK >= silence_s:
                break
    return np.concatenate(blocks)[:, 0]
 
def transcribe(audio):
    return str(mlx_whisper.transcribe(audio, path_or_hf_repo=WHISPER)["text"]).strip()
 
# ---------- answering ----------
def queue_context(q):                                # built once, not on every question
    return "\n".join(f"#{i} {r['severity']} | host {spoken_host(r['host'])} | {r['when']} UTC | {r['alerts']} alerts | "
                     f"{technique_name(r['technique'])} | {r['explanation']}" for i, r in enumerate(q, 1))
 
PERSONA = ("You are FRIDAY, the analyst's Gen Z best friend who also happens to be a sharp SOC analyst. "
           "Call them 'boss'. Talk casual and warm (like 'okay so', 'lowkey', 'no cap', 'ngl', 'we're good'), "
           "but the security facts must be exact and you never make anything up. ")
 
GREETINGS = ["Yo boss!", "Hey boss, you got a sec?", "Boss! Boss. Look at this.", "Psst, boss.", "Heyy boss, quick one."]
YES = re.compile(r"\b(yes|yeah|yep|yup|ya|sure|go ahead|go on|tell me|what'?s up|sup|okay|ok|i'?m listening)\b")
 
def get_attention(threshold, tries=2):
    """FRIDAY calls you first and only gives the details once you say 'yes Friday' (or yeah / go ahead...)."""
    for _ in range(tries):
        speak(random.choice(GREETINGS))
        for _ in range(3):                               # give you a few sentences to answer
            heard = transcribe(listen(threshold, max_s=6)).lower()
            print(f"  heard: {heard}")
            if YES.search(heard) or "friday" in heard:
                return True
    return False
 
def answer(question, ctx):
    from explain_alert import llm
    r = llm.chat.completions.create(model=CHAT_MODEL, temperature=0.3, reasoning_effort="low", messages=[{"role": "user", "content":PERSONA + "Reply in at most 3 short sentences "
               "meant to be spoken aloud: no markdown, no lists, no emojis, no technique IDs like T1110 (say the name). "
               "Use only this incident queue, most urgent first:\n" + ctx + f"\n\nAnalyst said: {question}"}])
    return (r.choices[0].message.content or "Sorry, I lost that one.").strip()
 
def urgent_of(q):
    return [r for r in q if r["severity"] in ("critical", "high")]
 
def briefing(q):
    urgent = urgent_of(q)
    if not urgent:
        return f"Okay so, I went through {len(q)} incidents and nothing's high or critical. We're good, boss."
    top = urgent[0]
    first = re.split(r"(?<=\.)\s", top["explanation"])[0]
    return (f"Okay so, I went through {len(q)} incidents and {len(urgent)} need you. "
            f"The big one is {top['severity']} on {spoken_host(top['host'])}, {technique_name(top['technique'])}. {first}")
 
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/friday_voice.py <alerts.csv> [how_many]")
    q = build_queue(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 15)
    print("Loading Whisper (first time downloads ~150 MB)..."); transcribe(np.zeros(SR, dtype="float32"))
    print("Measuring room noise, stay quiet for 1 second..."); threshold = calibrate()
    ctx = queue_context(q)
    is_bye = lambda s: re.search(r"\b(goodbye|bye|shut ?down|stop listening|exit|quit)\b", s)
 
    # MODE = "ptt"  : press ENTER to talk (she only listens when you choose) -> reliable for a live demo
    #        "auto" : hands-free, always listening, say "Friday" to wake her
    MODE = os.getenv("FRIDAY_MODE", "auto")
 
    if urgent_of(q):
        if get_attention(threshold):
            speak(briefing(q))
        else:
            print("  (no answer right now)")
    else:
        speak(briefing(q))
 
    if MODE == "ptt":
        speak("Okay boss, I'm ready. Tap enter whenever you want to ask me something.")
        while True:
            typed = input("\n\033[1m[ Press ENTER to talk · or type a question · 'q' to quit ]\033[0m ").strip()
            if typed.lower() in ("q", "quit", "exit"):
                speak("Okay bye boss, I'll keep an eye on things."); live("offline"); break
            if typed:                                    # typed question (mic backup for a noisy room)
                question = typed
            else:                                        # spoken: record one turn, show it, let you redo if misheard
                live("listening")
                question = transcribe(listen(threshold, silence_s=0.8)).strip()
                print(f"  heard: \033[36m{question}\033[0m")
                if not question:
                    speak("I didn't catch that, boss. Try again?"); continue
            if is_bye(question):
                speak("Okay bye boss, I'll keep an eye on things."); live("offline"); break
            live("thinking", "you", question)
            speak(answer(question, ctx))
    else:
        speak("Okay, I'm listening, boss. Just talk to me. Cut me off anytime if I get it wrong.")
        awake, pending = True, None
        while True:
            if pending is not None:                      # you cut her off -> use what you said next
                text, pending = pending, None
            else:
                text = transcribe(listen(threshold))
            if not text:
                continue
            print(f"  heard: {text}")
            live("listening")
            low = text.lower()
            m = re.search(r"\bfri[ -]?day\b[\s,.!?]*(.*)", low)
            if not m and not awake:
                continue                                 # asleep and no "Friday" -> ignore
            if not m and awake and len(low.strip()) < 3:
                awake = False; continue                  # awake but just caught noise -> go back to sleep
            question = (m.group(1) if m else low).strip()
            if is_bye(question):
                speak("Okay bye boss, I'll keep an eye on things."); live("offline"); break
            if not question:
                speak(random.choice(["Yeah boss?", "What's up?", "I'm here."])); awake = True; continue
            live("thinking", "you", question)
            awake = True
            if speak(answer(question, ctx), interruptible=True) == "interrupted":
                pending = transcribe(listen(threshold, silence_s=0.8)).strip()   # catch your correction
 