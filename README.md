# AI-recommender-system
AI RECOMMENDATION & VERIFICATION SYSTEM - README
=================================================

This system is designed to help factory engineers log, verify, and retrieve machine maintenance resolutions using a BERT + FNN AI model and a PyQt5-based desktop interface.

CONTENTS:
---------
1. Requirements
2. Step-by-Step Setup Guide
   
   A. Register Mailgun for Email Verification
   
   B. Launch Flask Server (Backend)
   
   C. Run PyQt5 Desktop App (Frontend)
   
   D. Using the System
   
4. Troubleshooting

--------------------------------------------------
**1. REQUIREMENTS**
--------------------------------------------------
Python version: >= 3.8

Install required packages:
> pip install -r requirements.txt

--------------------------------------------------
**2. STEP-BY-STEP SETUP GUIDE**
--------------------------------------------------

A. REGISTER MAILGUN FOR EMAIL VERIFICATION
------------------------------------------
1. Visit https://www.mailgun.com/ and sign up for a free account.
2. Under "Sending Domains", add and verify a domain (e.g., sandbox123.mailgun.org).
3. Copy the following details:
   - MAILGUN_API_KEY
   - MAILGUN_DOMAIN (e.g., sandbox0f424d09...mailgun.org)
4. In `frontend&verification2.py`, replace:
   - `MAILGUN_API_KEY = '...'`
   - `MAILGUN_DOMAIN = '...'`
   - `RECIPIENT_EMAIL = 'your@email.com'`

B. LAUNCH FLASK SERVER (BACKEND)
--------------------------------
1. Open terminal in the project directory.
2. Run:
> python flask_server.py

3. The server should run at http://127.0.0.1:5000
4. If you want remote access (for Mailgun webhook):
   - Use `ngrok`: https://ngrok.com/
   - Run: `ngrok http 5000`
   - Copy the HTTPS URL and paste it into `FLASK_SERVER_URL` in `frontend&verification2.py`.

C. RUN PYQT5 DESKTOP APP (FRONTEND)
-----------------------------------
1. Run:
> python BERT&FNN2.py (for first time only)

2. Ensure the following model files are present in your folder:
   - `fnn_focal_oversample.pth`
   - `label_to_resolution.pkl`
   - `bert_recommendation_model.pkl`

3. Run:
> python frontend&verification2.py

4. The GUI window should launch.

D. USING THE SYSTEM
-------------------
1. Enter a machine issue in Section 1 and click "Get Recommendations".
2. In Section 2, fill in Machine ID, Issue, and Action.
3. Click "Send Verification Email". The PIC will receive an email to approve/decline.
4. Section 3 shows latest cases and allows searching.
5. Use "Append Approved Cases" to integrate approved cases into training data.

--------------------------------------------------
3. TROUBLESHOOTING
--------------------------------------------------

- If emails are not sent:
  - Ensure Mailgun domain and API key are correctly configured.
  - Ensure FLASK_SERVER_URL is accessible (use ngrok if remote).

- If model fails to load:
  - Make sure all .pkl and .pth files are in the same directory as the GUI script.

- If BERT download is slow:
  - Ensure internet connection or pre-download model via HuggingFace.

Enjoy the system!
--------------------------------------------------
Created by Wong Xiao Hui

