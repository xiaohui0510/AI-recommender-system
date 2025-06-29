import sys
import pandas as pd
import torch
import pickle
import torch.nn.functional as F
import requests
import sqlite3
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QLabel, QPushButton,
    QLineEdit, QTextBrowser, QWidget, QMessageBox, QHBoxLayout,QComboBox
)
from PyQt5.QtCore import Qt, QTimer
from transformers import BertModel
from sqlalchemy import create_engine
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem


# === EMAIL CONFIG ===
MAILGUN_API_KEY = ''  # Replace with your real API key
MAILGUN_DOMAIN = ''    # Replace with your Mailgun domain
RECIPIENT_EMAIL = ''  # Replace with verified test email
SENDER_EMAIL = f'Verifier <mailgun@{MAILGUN_DOMAIN}>'
FLASK_SERVER_URL = ''

engine = create_engine('sqlite:///shift_reports.db')

def get_case_id_from_flask(issue, action, machine_id):
    response = requests.post(f"{FLASK_SERVER_URL}/new_case", json={"issue": issue, "action": action, "machine_id": machine_id})
    if response.status_code == 200:
        return response.json().get("case_id")
    else:
        raise Exception("Failed to get case ID from server.")

def load_model_and_data():
    # Updated to use the new focal_oversample model and separate label map
    model_path = "fnn_focal_oversample.pth"

    # Define FNN architecture matching the trained model
    class FNN(torch.nn.Module):
        def __init__(self, input_size, hidden_size, extra_hidden_size, num_classes):
            super(FNN, self).__init__()
            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.bn1 = torch.nn.BatchNorm1d(hidden_size)
            self.relu = torch.nn.ReLU()
            self.dropout = torch.nn.Dropout(0.5)
            self.fc2 = torch.nn.Linear(hidden_size, extra_hidden_size)
            self.bn2 = torch.nn.BatchNorm1d(extra_hidden_size)
            self.fc3 = torch.nn.Linear(extra_hidden_size, num_classes)

        def forward(self, x):
            x = self.fc1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc3(x)
            return x

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load label_to_resolution mapping from pickle
    with open("label_to_resolution.pkl", "rb") as f:
        label_to_resolution = pickle.load(f)
    num_classes = len(label_to_resolution)

    model = FNN(input_size=768, hidden_size=256, extra_hidden_size=256, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load tokenizer from the existing BERT pickle
    with open("bert_recommendation_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    tokenizer = model_data["tokenizer"]

    return model, tokenizer, label_to_resolution, device

def get_embedding(text, tokenizer, bert_model, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].to(device)
    return embedding

def recommend_resolution(new_issue, model, tokenizer, bert_model, label_to_resolution, device, top_k=5):
    """
    Generate top-k recommended resolutions for a new issue using BERT + FNN model.
    """
    model.eval()
    bert_model.eval()

    # Tokenize and embed issue using BERT
    inputs = tokenizer(new_issue, return_tensors='pt', truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        bert_output = bert_model(**inputs)
        embedding = bert_output.last_hidden_state.mean(dim=1)  # shape: [1, 768]
        outputs = model(embedding)
        probabilities = F.softmax(outputs, dim=1)
        top_probs, top_classes = torch.topk(probabilities, top_k)

    # Convert top-k predictions to (resolution, confidence) pairs
    recommendations = [
        (label_to_resolution[class_idx.item()], round(prob.item(), 4))
        for class_idx, prob in zip(top_classes[0], top_probs[0])
    ]
    return recommendations

class RecommenderUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Recommendation & Verification System")
        self.setGeometry(400, 100, 700, 900)
        self.model, self.tokenizer, self.label_to_resolution, self.device = load_model_and_data()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.bert_model.eval()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # === SECTION 1: Issue Input + Recommendation ===
        self.section1_label = QLabel("Section 1: AI Recommendation")
        layout.addWidget(self.section1_label)

        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Enter issue description (e.g. overload)")
        self.input_box.returnPressed.connect(self.get_recommendations)
        layout.addWidget(self.input_box)

        self.recommend_button = QPushButton("Get Recommendations")
        self.recommend_button.clicked.connect(self.get_recommendations)
        layout.addWidget(self.recommend_button)

        self.result_display = QTextBrowser()
        layout.addWidget(self.result_display)

        # === SECTION 2: Send Verification Email + Status Update ===
        self.section2_label = QLabel("Section 2: Verification Process")
        layout.addWidget(self.section2_label)

        self.verification_machine_input = QLineEdit()
        self.verification_machine_input.setPlaceholderText("Enter machine ID")
        layout.addWidget(self.verification_machine_input)

        self.verification_issue_input = QLineEdit()
        self.verification_issue_input.setPlaceholderText("Enter issue (e.g. overload)")
        layout.addWidget(self.verification_issue_input)

        self.verification_action_input = QLineEdit()
        self.verification_action_input.setPlaceholderText("Enter action (e.g. isolate power supply)")
        layout.addWidget(self.verification_action_input)

        self.send_verification_button = QPushButton("Send Verification Email")
        self.send_verification_button.clicked.connect(self.send_verification_email)
        layout.addWidget(self.send_verification_button)

        # === COMBINED SECTION: Search + Case Table ===
        self.section_combined_label = QLabel("Section 3: Search and View Cases")
        layout.addWidget(self.section_combined_label)

        # Dropdown for filter field
        self.search_field_dropdown = QComboBox()
        self.search_field_dropdown.addItems(["case_id", "date", "machine_id", "issue"])
        layout.addWidget(self.search_field_dropdown)

        # Input for search value
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter value to search")
        layout.addWidget(self.search_input)

        # Search button
        self.search_button = QPushButton("🔍 Search")
        self.search_button.clicked.connect(self.search_cases)
        layout.addWidget(self.search_button)

        # Table for cases
        self.case_table = QTableWidget()
        self.case_table.setColumnCount(6)
        self.case_table.setHorizontalHeaderLabels(["Case ID", "Date", "Machine ID", "Issue", "Action", "Status"])
        layout.addWidget(self.case_table)

        # Refresh button
        self.refresh_button = QPushButton("🔄 Refresh Table")
        self.refresh_button.clicked.connect(self.load_recent_cases)
        layout.addWidget(self.refresh_button)
        # Append button
        self.append_button = QPushButton("⬆️ Append Approved Cases")
        self.append_button.clicked.connect(self.append_approved_cases)
        layout.addWidget(self.append_button)

        # Load data initially
        self.load_recent_cases()

        # Final layout
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def get_recommendations(self):
        issue_description = self.input_box.text().strip()

        if not issue_description:
            self.result_display.setText("⚠️ Please enter a valid issue description.")
            return

        # 🔍 Call model
        recommendations = recommend_resolution(
            issue_description,
            self.model,
            self.tokenizer,
            self.bert_model,
            self.label_to_resolution,
            self.device
        )

        # 🖥️ Display result
        result_text = "🔍 Top AI Recommendations:\n"
        for idx, (resolution, confidence) in enumerate(recommendations, 1):
            result_text += f"{idx}. {resolution}\n"
        self.result_display.setText(result_text)

    def send_verification_email(self):
        issue = self.verification_issue_input.text().strip()
        action = self.verification_action_input.text().strip()
        machine_id = self.verification_machine_input.text().strip()
        if not issue or not action:
            QMessageBox.warning(self, "Input Missing", "Please fill in both Issue and Action.")
            return
        
        if not issue or not action or not machine_id:
            QMessageBox.warning(self, "Input Missing", "Please fill in all fields including machine ID.")
            return

        case_id = get_case_id_from_flask(issue, action, machine_id)

        accept_url = f"{FLASK_SERVER_URL}/accept/{case_id}"
        decline_url = f"{FLASK_SERVER_URL}/decline/{case_id}"

        subject = f"Verification Request: Case {case_id}"
        text_body = f"""Case ID: {case_id}
    Machine ID : {machine_id}
    Issue: {issue}
    Action: {action}

    Please review the following case:

    ✅ Accept: {accept_url}
    ❌ Decline: {decline_url}
    """

        url = f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages"
        auth = ("api", MAILGUN_API_KEY)
        data = {
            "from": SENDER_EMAIL,
            "to": RECIPIENT_EMAIL,
            "subject": subject,
            "text": text_body
        }

        try:
            response = requests.post(url, auth=auth, data=data)
            if response.status_code == 200:
                QMessageBox.information(self, "Email Sent", f"Case {case_id} verification has been sent to {RECIPIENT_EMAIL}.")
                self.start_status_polling(case_id)
            else:
                QMessageBox.critical(self, "Mailgun Error", f"Failed to send email.\nStatus: {response.status_code}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error: {str(e)}")

    def start_status_polling(self, case_id):
        self.timer = QTimer(self)
        self.timer.timeout.connect(lambda: self.check_case_status(case_id))
        self.timer.start(1000)  # every 1 second

    def check_case_status(self, case_id):
        try:
            response = requests.get(f"{FLASK_SERVER_URL}/status/{case_id}")
            if response.status_code == 200:
                status = response.text.strip()
                # Optionally print it or log it
                # print(f"Case {case_id} status: {status}")
                # if status.lower() in ["approved", "declined"]:
                #     self.timer.stop()
            else:
                print(f"Failed to check status. HTTP {response.status_code}")
        except Exception as e:
            print(f"Error checking case status: {str(e)}")

    def load_recent_cases(self):
        try:
            response = requests.get(f"{FLASK_SERVER_URL}/all_cases")
            if response.status_code == 200:
                cases = response.json()
                self.case_table.setRowCount(len(cases))
                for row_idx, case in enumerate(cases):
                    self.case_table.setItem(row_idx, 0, QTableWidgetItem(case["case_id"]))
                    self.case_table.setItem(row_idx, 1, QTableWidgetItem(case["date"]))
                    self.case_table.setItem(row_idx, 2, QTableWidgetItem(case["machine_id"]))
                    self.case_table.setItem(row_idx, 3, QTableWidgetItem(case["issue"]))
                    self.case_table.setItem(row_idx, 4, QTableWidgetItem(case["action"]))
                    self.case_table.setItem(row_idx, 5, QTableWidgetItem(case["status"]))
            else:
                QMessageBox.warning(self, "Error", f"Failed to load cases: HTTP {response.status_code}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not fetch case table:\n{str(e)}")

    def search_cases(self):
        field = self.search_field_dropdown.currentText()
        value = self.search_input.text().strip()

        if not value:
            QMessageBox.warning(self, "Missing Input", "Please enter a value to search.")
            return

        try:
            valid_fields = {"case_id", "date", "machine_ID", "issue", "action", "status"}
            if field not in valid_fields:
                QMessageBox.warning(self, "Invalid Field", "Invalid search field selected.")
                return

            conn = sqlite3.connect("approvals.db")
            cursor = conn.cursor()

            query = f"SELECT case_id, date, machine_ID, issue, action, status FROM approvals WHERE {field} LIKE ?"
            cursor.execute(query, ('%' + value + '%',))
            results = cursor.fetchall()

            self.case_table.setRowCount(len(results))
            for row_idx, row in enumerate(results):
                for col_idx, item in enumerate(row):
                    self.case_table.setItem(row_idx, col_idx, QTableWidgetItem(str(item)))

            conn.close()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Search failed:\n{str(e)}")

    def append_approved_cases(self):
        try:
            approval_conn = sqlite3.connect("approvals.db")
            shift_conn = sqlite3.connect("shift_reports.db")
            approval_cursor = approval_conn.cursor()
            shift_cursor = shift_conn.cursor()

            # ✅ Step 1: Get all approved cases that haven't been appended yet
            approval_cursor.execute("""
                SELECT case_id, machine_ID, issue, action, date FROM approvals
                WHERE status = 'Approved' AND appended = 0
            """)
            approved_cases = approval_cursor.fetchall()

            appended_count = 0

            # ✅ Step 2: Insert into shift_reports.db and update flag
            for case_id, machine_id, issue, action, date in approved_cases:
                shift_cursor.execute("""
                    INSERT INTO shift_reports (machine_id, issues, resolution, date)
                    VALUES (?, ?, ?, ?)
                """, (machine_id, issue, action, date))

                approval_cursor.execute("""
                    UPDATE approvals SET appended = 1 WHERE case_id = ?
                """, (case_id,))

                appended_count += 1

            # ✅ Step 3: Commit both DBs
            shift_conn.commit()
            approval_conn.commit()

            QMessageBox.information(self, "Append Complete", f"{appended_count} approved case(s) appended to shift_report.db.")

        except Exception as e:
            QMessageBox.critical(self, "Append Error", f"Failed to append:\n{str(e)}")

        finally:
            approval_conn.close()
            shift_conn.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RecommenderUI()
    window.show()
    sys.exit(app.exec_())
