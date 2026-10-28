"""EduMentor AI backend: accounts, saved snapshots, history and faculty cohort view.
Run:  python app.py   then open http://127.0.0.1:5000
Env:  EDUMENTOR_SECRET (session key), FACULTY_CODE (needed to register a faculty account; default 'faculty123' - change it)."""
import json, math, os, secrets, sqlite3
from datetime import datetime, timezone
from flask import Flask, g, jsonify, request, send_from_directory, session
from werkzeug.security import check_password_hash, generate_password_hash

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = json.load(open(os.path.join(BASE, "outputs", "web_model.json")))
SUBJECTS = ["Maths", "Programming", "DBMS", "Networks", "AI_Basics"]
LIMITS = {"assignment_avg":(0,100),"attendance_pct":(0,100),"assignment_trend":(-4,4),"coding_hours_week":(0,20),"study_hours_week":(0,30),"missed_deadlines":(0,10),**{s:(0,100) for s in SUBJECTS}}

app = Flask(__name__, static_folder=None)
app.secret_key = os.environ.get("EDUMENTOR_SECRET") or secrets.token_hex(32)
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")
DB = os.environ.get("EDUMENTOR_DB", os.path.join(BASE, "edumentor.db"))

def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB); g.db.row_factory = sqlite3.Row
        g.db.executescript("""CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, pw_hash TEXT NOT NULL, role TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS snapshots(id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id), created TEXT NOT NULL, inputs TEXT NOT NULL, risk REAL NOT NULL, level TEXT NOT NULL);""")
    return g.db
@app.teardown_appcontext
def close(_):
    d = g.pop("db", None)
    if d: d.close()

def score(inputs):
    s = dict(inputs); s["quiz_avg"] = sum(s[k] for k in SUBJECTS) / len(SUBJECTS)
    z = MODEL["intercept"] + sum(c * (s[f] - m) / sc for f, c, m, sc in zip(MODEL["features"], MODEL["coef"], MODEL["mean"], MODEL["scale"]))
    p = 1 / (1 + math.exp(-z)); return p, ("Low" if p < MODEL["t_low"] else "Medium" if p < MODEL["t_high"] else "High")
def clean(body):
    out = {}
    for k, (lo, hi) in LIMITS.items():
        try: v = float(body[k])
        except (KeyError, TypeError, ValueError): raise ValueError(f"Missing or invalid value: {k}")
        if not (lo <= v <= hi) or math.isnan(v): raise ValueError(f"{k} must be between {lo} and {hi}")
        out[k] = v
    return out
def me():
    uid = session.get("uid")
    if not uid: return None
    return db().execute("SELECT id, username, role FROM users WHERE id=?", (uid,)).fetchone()
def err(msg, code=400): return jsonify(error=msg), code
def user_json(u): return {"username": u["username"], "role": u["role"]}

@app.get("/")
def index(): return send_from_directory(os.path.join(BASE, "web"), "index.html")

@app.post("/api/register")
def register():
    b = request.get_json(silent=True) or {}; name = str(b.get("username", "")).strip(); pw = str(b.get("password", "")); role = b.get("role", "student")
    if not (3 <= len(name) <= 32) or not name.replace("_", "").replace(".", "").isalnum(): return err("Username must be 3-32 letters, digits, _ or .")
    if len(pw) < 8: return err("Password must be at least 8 characters.")
    if role not in ("student", "faculty"): return err("Invalid role.")
    if role == "faculty" and b.get("faculty_code") != os.environ.get("FACULTY_CODE", "faculty123"): return err("Wrong faculty code.", 403)
    try: cur = db().execute("INSERT INTO users(username,pw_hash,role) VALUES(?,?,?)", (name, generate_password_hash(pw), role)); db().commit()
    except sqlite3.IntegrityError: return err("That username is taken.", 409)
    session.clear(); session["uid"] = cur.lastrowid; return jsonify(user={"username": name, "role": role}), 201
@app.post("/api/login")
def login():
    b = request.get_json(silent=True) or {}; u = db().execute("SELECT * FROM users WHERE username=?", (str(b.get("username", "")).strip(),)).fetchone()
    if not u or not check_password_hash(u["pw_hash"], str(b.get("password", ""))): return err("Wrong username or password.", 401)
    session.clear(); session["uid"] = u["id"]; return jsonify(user=user_json(u))
@app.post("/api/logout")
def logout(): session.clear(); return jsonify(ok=True)
@app.get("/api/me")
def whoami():
    u = me(); return jsonify(user=user_json(u) if u else None)

@app.post("/api/snapshots")
def save():
    u = me()
    if not u: return err("Sign in first.", 401)
    if u["role"] != "student": return err("Only student accounts save snapshots.", 403)
    try: inputs = clean(request.get_json(silent=True) or {})
    except ValueError as e: return err(str(e))
    p, lvl = score(inputs)
    db().execute("INSERT INTO snapshots(user_id,created,inputs,risk,level) VALUES(?,?,?,?,?)", (u["id"], datetime.now(timezone.utc).isoformat(timespec="seconds"), json.dumps(inputs), p, lvl)); db().commit()
    return jsonify(risk=p, level=lvl), 201
@app.get("/api/history")
def history():
    u = me()
    if not u: return err("Sign in first.", 401)
    rows = db().execute("SELECT created, risk, level FROM snapshots WHERE user_id=? ORDER BY id", (u["id"],)).fetchall()
    return jsonify(history=[dict(r) for r in rows])
@app.get("/api/cohort")
def cohort():
    u = me()
    if not u: return err("Sign in first.", 401)
    if u["role"] != "faculty": return err("Faculty only.", 403)
    rows = db().execute("""SELECT u.username, s.inputs FROM snapshots s JOIN users u ON u.id=s.user_id
                           WHERE s.id=(SELECT MAX(id) FROM snapshots WHERE user_id=u.id) ORDER BY u.username""").fetchall()
    out = []
    for r in rows:
        i = json.loads(r["inputs"]); out.append({"id": r["username"], **i, **{"quiz_"+k: i[k] for k in SUBJECTS}})
    return jsonify(students=out)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
