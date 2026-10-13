import os, sqlite3, json, uuid, tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from flask import Flask, request, jsonify, render_template, g
from werkzeug.utils import secure_filename

from mood_engine import analyze_audio, predict_mood, MOODS
from music_catalog import search_itunes

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
DB_PATH = os.environ.get("MOODIFY_DB", str(DATA_DIR / "moodify.sqlite3"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "25"))
ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "flac", "m4a", "aac"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "change-this-in-production")

def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def db():
    if "db" not in g:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        g.db = conn
    return g.db

@app.teardown_appcontext
def close_db(_error=None):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY, filename TEXT NOT NULL, mood TEXT NOT NULL,
            confidence REAL NOT NULL, features_json TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS library (
            id INTEGER PRIMARY KEY AUTOINCREMENT, track_id TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL, artist TEXT NOT NULL, album TEXT DEFAULT '',
            artwork TEXT DEFAULT '', preview_url TEXT DEFAULT '', track_url TEXT DEFAULT '',
            mood TEXT DEFAULT '', added_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS plays (
            id INTEGER PRIMARY KEY AUTOINCREMENT, track_id TEXT, title TEXT NOT NULL,
            artist TEXT NOT NULL, mood TEXT DEFAULT '', played_at TEXT NOT NULL
        );
        """)

def read_catalog():
    path = DATA_DIR / "songs.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str).fillna("")
    return df.apply(lambda col: col.str.strip())

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model": "RandomForestClassifier", "moods": MOODS})

@app.get("/api/songs")
def songs():
    mood = request.args.get("mood", "").strip().lower()
    q = request.args.get("q", "").strip().lower()
    try:
        limit = min(max(int(request.args.get("limit", 100)), 1), 250)
    except ValueError:
        limit = 100
    df = read_catalog()
    if df.empty:
        return jsonify([])
    if mood and mood != "all":
        df = df[df["mood"].str.lower() == mood]
    if q:
        mask = df["title"].str.lower().str.contains(q, na=False) | df["artist"].str.lower().str.contains(q, na=False)
        df = df[mask]
    return jsonify(df.head(limit).to_dict(orient="records"))

@app.get("/api/recommendations")
def recommendations():
    mood = request.args.get("mood", "happy").strip().lower()
    if mood not in MOODS:
        return jsonify({"error": "Unknown mood", "valid_moods": MOODS}), 400
    df = read_catalog()
    if df.empty:
        return jsonify([])
    chosen = df[df["mood"].str.lower() == mood].copy()
    if chosen.empty:
        chosen = df.copy()
    return jsonify(chosen.sample(min(12, len(chosen)), random_state=None).to_dict(orient="records"))

@app.post("/api/analyze")
def analyze():
    if "audio" not in request.files:
        return jsonify({"error": "Choose an audio file first."}), 400
    file = request.files["audio"]
    if not file.filename:
        return jsonify({"error": "Choose an audio file first."}), 400
    safe_name = secure_filename(file.filename)
    ext = safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else ""
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Unsupported file type. Try WAV, MP3, OGG, FLAC, M4A or AAC."}), 400
    # Keep upload transient; do not retain user audio on disk.
    suffix = "." + ext
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(UPLOAD_DIR)) as tmp:
            file.save(tmp)
            tmp_path = tmp.name
        features = analyze_audio(tmp_path)
        mood, confidence, probabilities = predict_mood(features)
        analysis_id = str(uuid.uuid4())
        now = now_iso()
        db().execute(
            "INSERT INTO analyses (id, filename, mood, confidence, features_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (analysis_id, safe_name, mood, float(confidence), json.dumps(features), now)
        )
        db().commit()
        return jsonify({
            "id": analysis_id, "filename": safe_name, "mood": mood,
            "confidence": round(float(confidence), 3), "probabilities": probabilities,
            "features": features, "created_at": now,
            "note": "Mood is an estimate from audio features, not a definitive interpretation."
        })
    except Exception as exc:
        app.logger.exception("Audio analysis failed")
        return jsonify({"error": "Could not analyze this file. Try a short WAV or MP3 file. Details: " + str(exc)[:180]}), 422
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except OSError: pass

@app.get("/api/history")
def history():
    rows = db().execute("SELECT * FROM analyses ORDER BY created_at DESC LIMIT 100").fetchall()
    result = []
    for r in rows:
        item = dict(r)
        item["features"] = json.loads(item.pop("features_json"))
        result.append(item)
    return jsonify(result)

@app.get("/api/analytics")
def analytics():
    rows = db().execute("SELECT mood, COUNT(*) as count FROM analyses GROUP BY mood").fetchall()
    moods = {m: 0 for m in MOODS}
    for r in rows:
        moods[r["mood"]] = r["count"]
    total = sum(moods.values())
    recent = db().execute("SELECT created_at, mood, filename FROM analyses ORDER BY created_at DESC LIMIT 7").fetchall()
    return jsonify({"total_analyses": total, "mood_counts": moods, "recent": [dict(r) for r in recent]})

@app.get("/api/library")
def library():
    rows = db().execute("SELECT * FROM library ORDER BY added_at DESC").fetchall()
    return jsonify([dict(r) for r in rows])

@app.post("/api/library")
def add_library():
    item = request.get_json(silent=True) or {}
    required = ["track_id", "title", "artist"]
    if not all(item.get(k) for k in required):
        return jsonify({"error": "track_id, title and artist are required"}), 400
    db().execute("""INSERT OR IGNORE INTO library
        (track_id,title,artist,album,artwork,preview_url,track_url,mood,added_at)
        VALUES (?,?,?,?,?,?,?,?,?)""",
        (str(item["track_id"]), str(item["title"])[:200], str(item["artist"])[:200],
         str(item.get("album",""))[:200], str(item.get("artwork",""))[:1000],
         str(item.get("preview_url",""))[:1000], str(item.get("track_url",""))[:1000],
         str(item.get("mood",""))[:30], now_iso()))
    db().commit()
    return jsonify({"ok": True})

@app.delete("/api/library/<path:track_id>")
def remove_library(track_id):
    db().execute("DELETE FROM library WHERE track_id=?", (track_id,))
    db().commit()
    return jsonify({"ok": True})

@app.post("/api/plays")
def add_play():
    item = request.get_json(silent=True) or {}
    if not item.get("title") or not item.get("artist"):
        return jsonify({"error": "title and artist are required"}), 400
    db().execute("INSERT INTO plays (track_id,title,artist,mood,played_at) VALUES (?,?,?,?,?)",
                 (str(item.get("track_id","")), str(item["title"])[:200], str(item["artist"])[:200],
                  str(item.get("mood",""))[:30], now_iso()))
    db().commit()
    return jsonify({"ok": True})

@app.get("/api/plays")
def plays():
    rows = db().execute("SELECT * FROM plays ORDER BY played_at DESC LIMIT 100").fetchall()
    return jsonify([dict(r) for r in rows])

@app.get("/api/external-search")
def external_search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"results": [], "source": "iTunes Search API"})
    try:
        return jsonify({"results": search_itunes(q), "source": "iTunes Search API"})
    except Exception as exc:
        return jsonify({"error": "External search is temporarily unavailable.", "detail": str(exc)[:160], "results": []}), 502

@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": f"File too large. Maximum upload size is {MAX_UPLOAD_MB} MB."}), 413

init_db()

if __name__ == "__main__":
    # Local development only. Use gunicorn for production deployment (set HOST=0.0.0.0 only if you need other devices to connect).
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5000")), debug=os.environ.get("FLASK_DEBUG") == "1")
