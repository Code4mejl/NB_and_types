# Naive Bayes Machine Learning Dashboard

A web-based machine learning dashboard that demonstrates three different variants of the Naive Bayes algorithm for real-world prediction tasks.

The system allows users to test spam detection, fake news classification, and student result prediction through an interactive dashboard.


# 🌐 Live Demo

🚀 **Live Application:**
PASTE_YOUR_RENDER_LINK_HERE

https://nb-and-types.onrender.com/



---

## Project Overview

This project demonstrates how the **Naive Bayes Classifier** can be applied to different types of datasets.

Three variants of Naive Bayes are implemented:

| Task                     | Model Used              |
| ------------------------ | ----------------------- |
| Spam Email Detection     | Multinomial Naive Bayes |
| Fake News Detection      | Bernoulli Naive Bayes   |
| Student Score Prediction | Gaussian Naive Bayes    |

The application is built using Python Flask and deployed with a simple HTML/CSS dashboard.

---

## Key Features

• Spam email detection
• Fake news classification
• Student performance prediction
• Interactive dashboard interface
• Machine learning models trained using real datasets
• Web deployment ready

---

## Machine Learning Models

### Multinomial Naive Bayes

Used for spam detection because it works well with word frequency features in text classification problems.

### Bernoulli Naive Bayes

Used for fake news detection since it focuses on binary word occurrence (word present or not).

### Gaussian Naive Bayes

Used for predicting student results because it works best with continuous numerical values such as exam scores.

---

## Datasets Used

1. SMS Spam Detection Dataset
2. Fake News Dataset (Fake.csv and True.csv)
3. Student Performance Dataset

---

## Project Architecture

Dataset → Data Preprocessing → Model Training → Model Serialization (.pkl) → Flask Backend → Web Dashboard

---

## Technologies Used

Python
Flask
Scikit-learn
Pandas
HTML
CSS

---

## Installation

Clone the repository

git clone https://github.com/yourusername/naive-bayes-dashboard.git

Navigate to the project folder

cd naive-bayes-dashboard

Install required packages

pip install -r requirements.txt

Train the models

python train_model.py

Run the application

python app.py

---

## Folder Structure

naive-bayes-dashboard
│
├── datasets
├── models
├── templates
├── static
├── train_model.py
├── app.py
└── requirements.txt

---

## Web Interface

The dashboard allows users to:

• Enter email text to detect spam
• Enter news text to classify fake or real news
• Enter student scores to predict pass or fail

---

## Future Improvements

• Add model accuracy visualization
• Add prediction confidence scores
• Improve UI with advanced dashboard components
• Integrate additional machine learning models

---

## Author

Machine Learning Project demonstrating multiple Naive Bayes variants in a single application.
