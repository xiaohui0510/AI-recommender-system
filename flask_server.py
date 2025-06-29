from flask import Flask, request, jsonify
import sqlite3
from datetime import datetime
from datetime import date

app = Flask(__name__)
DB_FILE = "approvals.db"

# =============================
# DB SETUP & UTILITIES
# =============================

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS approvals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT UNIQUE,
                date TEXT DEFAULT (DATE('now')),  -- ✅ New column
                machine_ID TEXT,
                issue TEXT,
                action TEXT, 
                status TEXT DEFAULT 'Pending',
                appended INTEGER DEFAULT 0  -- ✅ New column
            )
        """)
        conn.commit()

def update_status(case_id, status):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            "UPDATE approvals SET status = ? WHERE case_id = ?",
            (status, case_id)
        )
        conn.commit()

def get_status(case_id):
    with sqlite3.connect(DB_FILE) as conn:
        result = conn.execute(
            "SELECT status FROM approvals WHERE case_id = ?",
            (case_id,)
        ).fetchone()
    return result[0] if result else "Pending"

# =============================
# ROUTES
# =============================

@app.route("/new_case", methods=["POST"])
def new_case():
    try:
        data = request.get_json(force=True)
        issue = data.get("issue", "").strip()
        action = data.get("action", "").strip()
        machine_id = data.get("machine_id", "").strip()  # ✅ Added
        current_date = datetime.now().strftime("%Y-%m-%d")  # ✅ Added

        if not issue or not action or not machine_id:
            return jsonify({"error": "Missing issue, action, or machine_id"}), 400

        with sqlite3.connect(DB_FILE) as conn:
            result = conn.execute("SELECT MAX(id) FROM approvals").fetchone()
            next_id = (result[0] or 0) + 1
            case_id = f"{next_id:03}"

            conn.execute(
                "INSERT INTO approvals (case_id, date, issue, action, machine_ID, status) VALUES (?, ?, ?, ?, ?, ?)",
                (case_id, current_date, issue, action, machine_id, "Pending")

            )
            conn.commit()

        return jsonify({"case_id": case_id})

    except Exception as e:
        print("❌ Error in /new_case:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/accept/<case_id>")
def accept(case_id):
    update_status(case_id, "Approved")
    return f"✅ Case {case_id} has been accepted."

@app.route("/decline/<case_id>")
def decline(case_id):
    update_status(case_id, "Declined")
    return f"❌ Case {case_id} has been declined."

@app.route("/status/<case_id>")
def status(case_id):
    return get_status(case_id)

@app.route("/all_cases")
def all_cases():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute("SELECT case_id, current_date, issue, action, machine_id, status FROM approvals ORDER BY id DESC LIMIT 5")
        cases = cursor.fetchall()
        result = [
            {
                "case_id": row[0],
                "date": row[1],
                "issue": row[2],
                "action": row[3],
                "machine_id": row[4],
                "status": row[5]
            }
            for row in cases
        ]
    return jsonify(result)

@app.route("/case/<case_id>")
def case_details(case_id):
    with sqlite3.connect(DB_FILE) as conn:
        row = conn.execute(
            "SELECT case_id, current_date, issue, action, machine_id, status FROM approvals WHERE case_id = ?",
            (case_id,)
        ).fetchone()
        if row:
            return jsonify({
                "case_id": row[0],
                "current_date":row[1],
                "issue": row[2],
                "action": row[3],
                "machine_id": row[4],
                "status": row[5]
            })
        else:
            return jsonify({"error": "Case not found"}), 404

# =============================
# INIT & RUN
# =============================

if __name__ == "__main__":
    init_db()
    print("✅ Flask server running on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
