import pandas as pd
import ast
import numpy as np
import re

# === Load the dataset ===
df = pd.read_csv("D:/PYTHON 3/ICTAK Python3/course-recommendation-project/data/raw/coursera_Social_Sciences_cleaned_final.csv")

# ------------------------
# Step 1: Initial Inspection
# ------------------------
print("Initial shape:", df.shape)
print("\nInitial missing values:\n", df.isnull().sum())

# ------------------------
# Step 2: Basic Cleaning
# ------------------------

# Convert 'skill_tags' from string to list
df['skill_tags'] = df['skill_tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# Normalize 'language', 'level', 'provider'
df['language'] = df['language'].str.lower().str.strip()
df['level'] = df['level'].replace({'Beginner': 'Beginner level'}).str.lower().str.strip()
df['provider'] = df['provider'].str.strip()

# Fill missing 'description' from 'course_detail'
df['description'] = df['description'].fillna(df['course_detail'])

# Fill missing 'instructor'
df['instructor'] = df['instructor'].fillna("Unknown")

# Convert 'certificate' to binary (1 = Yes, 0 = No)
df['certificate'] = df['certificate'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

# Clip 'price' and clean 'effort_hours'
df['price'] = df['price'].clip(upper=500)
df['effort_hours'] = df['effort_hours'].replace(0, np.nan)
df['effort_hours'] = df.groupby('level')['effort_hours'].transform(lambda x: x.fillna(x.median()))
df['effort_hours'] = df['effort_hours'].clip(upper=60)

# ------------------------
# Step 3: Duration Processing
# ------------------------

# Binary flag: has_duration
df['has_duration'] = df['schedule'].apply(lambda x: 1 if 'week' in str(x).lower() else 0)

# Estimate 'duration_weeks'
def estimate_duration(schedule):
    schedule = str(schedule).lower()
    match = re.search(r"(\d+)\s*(week|weeks)", schedule)
    if match:
        return float(match.group(1))
    elif 'short' in schedule or 'one' in schedule:
        return 1.0
    elif 'medium' in schedule:
        return 2.0
    elif 'long' in schedule:
        return 4.0
    elif 'month' in schedule:
        return 4.0
    else:
        return 2.0  # default

df['duration_weeks'] = df['schedule'].apply(estimate_duration)

# Create duration level
def duration_level(weeks):
    if weeks <= 1:
        return "short"
    elif weeks <= 3:
        return "medium"
    else:
        return "long"

df["duration_level"] = df["duration_weeks"].apply(duration_level)

# Binary flag for effort availability
df['has_effort'] = df['effort_hours'].notnull().astype(int)

# ------------------------
# Step 4: Feature Engineering
# ------------------------

# Remove unrated or missing ratings
df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
df = df[df['rating'] > 0].reset_index(drop=True)

# Clean num_reviews
df['num_reviews'] = pd.to_numeric(df['num_reviews'], errors='coerce').fillna(0)

# Create 'popular_course' flag
df['popular_course'] = (df['num_reviews'] >= 1000).astype(int)

# Create 'high_rating' (rating ≥ 4.5 and 50+ reviews)
df['high_rating'] = ((df['rating'] >= 4.5) & (df['num_reviews'] >= 50)).astype(int)

# Combine for NLP: full_text = title + description + course_detail
df['full_text'] = df['title'].astype(str) + ' ' + df['description'].astype(str) + ' ' + df['course_detail'].astype(str)

# Add category column
df['category'] = 'Social Sciences'

# Drop duplicates by title + provider
df = df.drop_duplicates(subset=['title', 'provider'])

# Drop rows missing core fields
df = df.dropna(subset=["description", "course_detail"])

# ------------------------
# Step 5: Final Inspection and Save
# ------------------------
print("Final shape:", df.shape)
print("\nFinal missing values:\n", df.isnull().sum())
print("\nSample rows:\n", df[['title', 'rating', 'num_reviews', 'level', 'duration_weeks', 'high_rating']].head())

# Save cleaned dataset
output_file = "D:/PYTHON 3/ICTAK Python3/course-recommendation-project/data/raw/coursera_Social_Sciences_cleaned_final_ready.csv"
df.to_csv(output_file, index=False)
print(f"\n Final cleaned dataset saved to: {output_file}")
