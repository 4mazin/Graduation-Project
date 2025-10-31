from fastapi import FastAPI, Query, Request
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
import os
app = FastAPI(title="Course Recommendation API")

#  Add this block here
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://127.0.0.1:5500"] if using Live Server
    allow_credentials=True,
    allow_methods=["*"],  # <-- includes OPTIONS (important!)
    allow_headers=["*"],
)

# ===== Connecting To Mongo =====
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/Course_Recommendation_System")
client = MongoClient(mongo_uri)
db = client["Course_Recommendation_System"]
collection = db["clean_courses"]

# ===== Load Models and Data =====
# courses = pd.read_pickle("./Models/courses_with_vectors.pkl")

with open("./Models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("./Models/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

sbert_embeddings = np.load("./Models/sbert_embeddings.npy")

# ===== Helper: Convert MongoDB cursor to list =====
def get_all_courses():
    return list(collection.find({}, {"_id": 0}))


# ===== Search Function =====
def search_courses_tfidf(query: str, top_n=10, min_sim=0.2):
    if not isinstance(query, str) or not query.strip():
        return {"error": "Invalid or empty query."}

    query_vector = vectorizer.transform([query])
    sim_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[::-1][:top_n]

    all_courses = get_all_courses()

    results = []
    for idx in top_indices:
        if sim_scores[idx] < min_sim:
            continue
        if idx >= len(all_courses):
            continue
        course = all_courses[idx]
        results.append({
            "course_name": course["course_name"],
            "description": course["description"],
            "url": course["url"],
            "similarity": float(sim_scores[idx])
        })


    # If no results above threshold
    if not results:
        return {"message": "Not Found"}

    return results



# ===== SBERT Recommendation Function =====
def recommend_by_course(enrolled_courses, top_n=10, min_sim=0.4):
    all_courses = get_all_courses()  # fetch from MongoDB

    if not isinstance(enrolled_courses, list):
        enrolled_courses = [enrolled_courses]

    lower_enrolled = [c.lower() for c in enrolled_courses]

    # Convert the MongoDB list to a DataFrame so we can use .str.lower()
    courses_df = pd.DataFrame(all_courses)

    matched = courses_df[courses_df['course_name'].str.lower().isin(lower_enrolled)]

    if matched.empty:
        return {"error": "No matching enrolled courses found."}

    indices = matched.index.tolist()
    query_emb = sbert_embeddings[indices].mean(axis=0).reshape(1, -1)

    sim_scores = cosine_similarity(query_emb, sbert_embeddings).flatten()
    sim_scores[indices] = -1  # Exclude enrolled courses

    top_indices = sim_scores.argsort()[::-1][:top_n]

    recommendations = []
    for idx in top_indices:
        if sim_scores[idx] < min_sim:
            continue  # skip if similarity is too low
        course = all_courses[idx]
        recommendations.append({
            "course_name": course["course_name"],
            "description": course["description"],
            "url": course["url"],
            "similarity": float(sim_scores[idx])
        })

    # If no valid recommendations above threshold
    if not recommendations:
        return {"message": "Not Found"}

    return recommendations


# ===== API Endpoints =====
@app.get("/")
def home():
    return {"message": "Welcome to the Course Search & Recommendation API!"}

@app.post("/search")
async def search(request: Request):
    data = await request.json()
    query = data.get("query", "")
    if not query:
        return {"error": "No query provided."}
    return {"results": search_courses_tfidf(query)}

@app.post("/recommend")
async def recommend(request: Request):
    data = await request.json()
    enrolled_courses = data.get("enrolled_courses", [])
    return {"recommendations": recommend_by_course(enrolled_courses)}
