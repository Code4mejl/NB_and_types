from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# load models
spam_model, spam_vec = joblib.load("models/spam_model.pkl")
fake_model, fake_vec = joblib.load("models/fake_model.pkl")
student_model = joblib.load("models/student_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


# SPAM PREDICTION
@app.route("/predict_spam", methods=["POST"])
def predict_spam():

    text = request.form["email"]

    vec = spam_vec.transform([text])

    pred = spam_model.predict(vec)[0]

    return jsonify({"result": pred})


# FAKE NEWS
@app.route("/predict_fake", methods=["POST"])
def predict_fake():

    text = request.form["news"]

    vec = fake_vec.transform([text])

    pred = fake_model.predict(vec)[0]

    return jsonify({"result": pred})


# STUDENT RESULT
@app.route("/predict_student", methods=["POST"])
def predict_student():

    math = float(request.form["math"])
    read = float(request.form["reading"])
    write = float(request.form["writing"])

    pred = student_model.predict([[math, read, write]])[0]

    if pred == 1:
        result = "PASS"
    else:
        result = "FAIL"

    return jsonify({"result": result})


if __name__ == "__main__":
    app.run(debug=True)