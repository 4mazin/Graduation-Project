# Graduation-Project

Course Recommendation System
A recommendation system built using TF-IDF and Sentence-BERT (SBERT) embeddings to suggest relevant courses based on user input or enrolled courses.
It includes a FastAPI backend and can be easily connected to a web frontend.

🚀 Features

✅ Content-based recommendations using TF-IDF
✅ Semantic similarity using SBERT embeddings
✅ Hybrid recommendation combining both models
✅ RESTful FastAPI backend with /search and /recommend endpoints
✅ CORS support for frontend integration
✅ Easily extendable with user interaction data

🧩 Tech Stack

Python 3.10+

FastAPI

scikit-learn

Sentence-Transformers

pandas / numpy

pickle / joblib

HTML, CSS, JavaScript (frontend)



The dataset was compiled from multiple sources on Kaggle, including:

all_courses.csv

udemy_courses.csv

online_courses.csv

After preprocessing and cleaning, over 5600+ course entries were used for training.



Model Training Overview

Text Cleaning: stopword removal, lemmatization, and punctuation cleaning

Feature Extraction:

TF-IDF Vectorization

Sentence-BERT embeddings

Normalization: applied to SBERT embeddings for cosine similarity

Model Saving:

tfidf_vectorizer.pkl

tfidf_matrix.pkl

sbert_embeddings.npy

🧠 How It Works
🔍 1. Search Function

Users can search for a topic, and the system finds the most relevant courses using TF-IDF similarity.

🎓 2. Recommendation Function

Given a list of enrolled courses, the system uses SBERT embeddings to recommend similar ones.

🧪 Run Locally
1️⃣ Clone the repository
git clone https://github.com/4mazin/GP_Recommendation_System.git
cd Graduation-Project

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Start the API
uvicorn api.main:app --reload

4️⃣ Test the endpoints

Open in browser:

http://127.0.0.1:8000/docs
