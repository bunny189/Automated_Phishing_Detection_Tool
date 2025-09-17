from flask import Flask, render_template, request
import os
from models_src.predict import ensemble_predict
from utils.email_sender import send_alert_email

app = Flask(__name__)

# Load phishing database (simple text file with known phishing URLs)
DATABASE_FILE = os.path.join("data", "phishing_db.txt")
with open(DATABASE_FILE, "r") as f:
    phishing_db = set(line.strip() for line in f.readlines())


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    user_email = None
    url = None

    if request.method == "POST":
        url = request.form.get("url")
        user_email = request.form.get("email")

        if url in phishing_db:
            result = "⚠️ This link is a known phishing site!"
            send_alert_email(user_email, url, is_phishing=True)
        else:
            prediction = ensemble_predict(url)
            if prediction == 1:
                result = "⚠️ This link is classified as a phishing site!"
                send_alert_email(user_email, url, is_phishing=True)
            else:
                result = "✅ This link appears safe."

    return render_template("index.html", result=result, url=url, email=user_email)


if __name__ == "__main__":
    app.run(debug=True)
