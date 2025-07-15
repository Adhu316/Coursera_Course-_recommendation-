import pickle
from app.course_recommender import CourseRecommender

csv_path = "data/cleaned_coursera_4300.csv"

recommender = CourseRecommender(csv_path)

# Saving model as pickle file
with open("models/recommender.pkl", "wb") as f:
    pickle.dump(recommender, f)

print("Model saved to models/recommender.pkl")