"""Automated tests for the backend (run: python test_app.py)"""
import os, tempfile
os.environ["EDUMENTOR_DB"] = os.path.join(tempfile.mkdtemp(), "t.db"); os.environ["FACULTY_CODE"] = "letmein"
import app as A
c = A.app.test_client(); good = dict(Maths=40,Programming=50,DBMS=45,Networks=42,AI_Basics=48,assignment_avg=52,attendance_pct=62,assignment_trend=-1.2,coding_hours_week=1,study_hours_week=5,missed_deadlines=4)
def check(name, cond): print(("PASS " if cond else "FAIL ")+name); assert cond, name
check("me anonymous", c.get("/api/me").json["user"] is None)
check("save needs login", c.post("/api/snapshots", json=good).status_code == 401)
check("short password rejected", c.post("/api/register", json={"username":"alice","password":"short"}).status_code == 400)
check("register student", c.post("/api/register", json={"username":"alice","password":"password123"}).status_code == 201)
check("duplicate username", c.post("/api/register", json={"username":"alice","password":"password123"}).status_code == 409)
r = c.post("/api/snapshots", json=good); check("save snapshot", r.status_code == 201 and r.json["level"] == "High")
check("invalid input rejected", c.post("/api/snapshots", json={**good,"attendance_pct":150}).status_code == 400)
check("missing field rejected", c.post("/api/snapshots", json={k:v for k,v in good.items() if k!="DBMS"}).status_code == 400)
better = {**good, **dict(Maths=75,Programming=80,DBMS=72,Networks=70,AI_Basics=78,assignment_avg=80,attendance_pct=90,assignment_trend=.8,coding_hours_week=6,study_hours_week=11,missed_deadlines=0)}
c.post("/api/snapshots", json=better); h = c.get("/api/history").json["history"]
check("history has 2 entries, risk falls", len(h) == 2 and h[1]["risk"] < h[0]["risk"])
check("student cannot read cohort", c.get("/api/cohort").status_code == 403)
c.post("/api/logout"); check("wrong password", c.post("/api/login", json={"username":"alice","password":"nope12345"}).status_code == 401)
check("faculty code enforced", c.post("/api/register", json={"username":"prof","password":"password123","role":"faculty","faculty_code":"bad"}).status_code == 403)
check("register faculty", c.post("/api/register", json={"username":"prof","password":"password123","role":"faculty","faculty_code":"letmein"}).status_code == 201)
co = c.get("/api/cohort").json["students"]; check("faculty sees latest snapshot per student", len(co) == 1 and co[0]["id"] == "alice" and co[0]["attendance_pct"] == 90)
check("faculty cannot save snapshots", c.post("/api/snapshots", json=good).status_code == 403)
check("password stored hashed", "password123" not in open(os.environ["EDUMENTOR_DB"], "rb").read().decode("latin1"))
print("ALL TESTS PASSED")
