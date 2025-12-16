from flask import Flask, render_template, request
from predict import predict_video

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        link = request.form["link"]
        result = predict_video(link)
        return f"Harmful Content Category: {result}"
    return '''
        <form method="post">
            <input name="link" placeholder="YouTube link">
            <input type="submit">
        </form>
    '''

app.run(debug=True)
