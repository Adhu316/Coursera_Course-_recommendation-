from flask import Flask, request, render_template
from app.chatbot import SimpleRecommender
import os

app = Flask(__name__, template_folder="../templates")

recommender = SimpleRecommender("models/recommender.pkl")


@app.route("/", methods=["GET", "POST"])
def index():
    response = None
    if request.method == "POST":
        user_input = request.form["user_input"]
        response = recommender.ask(user_input)
    return render_template("index.html", response=response)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
