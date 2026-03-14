import pandas as pd
import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB


# -------------------------------
# SPAM DATASET (Multinomial NB)
# -------------------------------

# read dataset
spam = pd.read_csv("datasets/spam.csv", encoding="latin-1")

# select correct columns
spam = spam[["v1", "v2"]]

# rename columns
spam.columns = ["label", "text"]

X_spam = spam["text"]
y_spam = spam["label"]

vec_spam = CountVectorizer()

X_spam_vec = vec_spam.fit_transform(X_spam)

spam_model = MultinomialNB()

spam_model.fit(X_spam_vec, y_spam)

joblib.dump((spam_model, vec_spam), "models/spam_model.pkl")


# -------------------------------
# FAKE NEWS DATASET (Bernoulli NB)
# -------------------------------

fake = pd.read_csv("datasets/Fake.csv", encoding="latin-1")
true = pd.read_csv("datasets/True.csv", encoding="latin-1")

fake["label"] = "fake"
true["label"] = "real"

data = pd.concat([fake, true])

X_fake = data["text"]
y_fake = data["label"]

vec_fake = CountVectorizer(binary=True)

X_fake_vec = vec_fake.fit_transform(X_fake)

fake_model = BernoulliNB()

fake_model.fit(X_fake_vec, y_fake)

joblib.dump((fake_model, vec_fake), "models/fake_model.pkl")


# -------------------------------
# STUDENT DATASET (Gaussian NB)
# -------------------------------

students = pd.read_csv("datasets/students.csv")

X_student = students[["math score", "reading score", "writing score"]]

# convert score to pass/fail
avg_score = students[["math score","reading score","writing score"]].mean(axis=1)

y_student = (avg_score > 50).astype(int)

student_model = GaussianNB()

student_model.fit(X_student, y_student)

joblib.dump(student_model, "models/student_model.pkl")


print("All models trained successfully")