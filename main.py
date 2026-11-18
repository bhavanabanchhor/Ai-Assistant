
from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from groq import Groq

app = Flask(__name__)

# Load environment variables
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing from .env file")

# Initialize Groq client
client = Groq(api_key=api_key)

MODEL = "openai/gpt-oss-120b"


@app.route("/")
def hello_world():
    return render_template("index.html")


# ---------------- ASK ANYTHING ----------------
@app.route("/ask", methods=["POST"])
def ask():

    question = request.form.get("question", "").strip()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    try:

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful personal AI assistant. Give clear, friendly, accurate answers."
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            model=MODEL,
            temperature=0.5,
            max_tokens=1024
        )

        answer = chat_completion.choices[0].message.content

        return jsonify({
            "response": answer
        }), 200

    except Exception as e:
        print("Groq API Error:", repr(e))

        return jsonify({
            "error": str(e)
        }), 500

# ---------------- SUMMARIZE EMAIL ----------------
@app.route("/summarize", methods=["POST"])
def summarize():

    email_text = request.form.get("question", "").strip()

    if not email_text:
        return jsonify({
            "error": "Please paste an email to summarize."
        }), 400

    try:

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert email assistant. Summarize emails clearly and concisely."
                },
                {
                    "role": "user",
                    "content": f"Summarize this email in 2-3 sentences:\n\n{email_text}"
                }
            ],
            model=MODEL,
            temperature=0.3,
            max_tokens=512
        )

        summary = chat_completion.choices[0].message.content

        return jsonify({
            "summary": summary
        }), 200

    except Exception as e:

        print("Groq Summarization Error:", e)

        return jsonify({
            "error": "Email summarization is unavailable. Please try again."
        }), 503


if __name__ == "__main__":
    app.run(debug=True)
