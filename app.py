from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").lower()

    # Improved, natural-sounding responses
    if any(word in user_message for word in ["math", "numbers", "computers", "coding", "programming"]):
        reply = (
            "It sounds like you enjoy analytical and technical challenges. "
            "You might thrive in careers like Data Science, Software Engineering, or AI Research. "
            "These fields allow you to work with computers, problem-solving, and innovation."
        )
    elif any(word in user_message for word in ["art", "design", "drawing", "painting", "creativity"]):
        reply = (
            "Your creative side is shining through! "
            "Consider careers in Graphic Design, Animation, or UI/UX Design. "
            "These paths let you turn your artistic ideas into real-world projects."
        )
    elif any(word in user_message for word in ["movies", "film", "acting", "theater", "cinema", "writing", "storytelling"]):
        reply = (
            "You seem to have a passion for storytelling and entertainment. "
            "Careers in Film, Media, Content Creation, or Script Writing could be exciting for you. "
            "These paths allow you to bring stories to life and engage audiences."
        )
    elif any(word in user_message for word in ["music", "singing", "dance"]):
        reply = (
            "Your interest in performing arts is wonderful! "
            "You could explore careers in Music, Performing Arts, or Entertainment. "
            "These fields allow you to showcase your talent and creativity."
        )
    elif any(word in user_message for word in ["communication", "teaching", "people", "social", "languages"]):
        reply = (
            "It seems you enjoy interacting with people and sharing knowledge. "
            "Consider careers in Teaching, Counseling, or Public Relations. "
            "These paths let you help and influence others positively."
        )
    else:
        reply = (
            "You have diverse interests! "
            "Careers in Management, Entrepreneurship, or Marketing might suit you. "
            "These fields offer opportunities to lead projects, solve problems, and make an impact."
        )

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)
