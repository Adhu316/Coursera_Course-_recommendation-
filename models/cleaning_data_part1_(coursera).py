import pandas as pd
import numpy as np
import re
from langdetect import detect, DetectorFactory
from ftfy import fix_text
import regex as re2
import emoji
import ast

# === Config ===
DetectorFactory.seed = 0

# === Step 1: Load Dataset ===
input_file = "D:/PYTHON 3/ICTAK Python3/course-recommendation-project/data/raw/coursera_Social_Sciences_cleaned.csv"
df = pd.read_csv(input_file)

# === Step 2: Fix Encoding, Remove Emojis, Normalize Punctuation ===
def clean_text_raw(text):
    text = fix_text(str(text))
    text = emoji.replace_emoji(text, replace='')
    text = re2.sub(r'[^\x00-\x7F]+', '', text)
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return text.strip()

columns_to_clean = ['title', 'description', 'course_detail', 'instructor', 'full_text']
for col in columns_to_clean:
    if col in df.columns:
        df[col] = df[col].astype(str).apply(clean_text_raw)

# === Step 3: Keep Only English Titles ===
def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

df = df[df['title'].apply(is_english)]

# === Step 4: Impute Missing Values ===
for col in ["schedule", "level", "instructor", "description", "skill_tags"]:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

# === Step 5: Clean and Convert 'price' ===
df["price"] = df["price"].replace("Free", "$0.00")
df["price"] = df["price"].astype(str).str.replace(r"[^0-9.]", "", regex=True)
df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)

# === Step 5.5: Clean and Convert 'enrolled' ===
if "enrolled" in df.columns:
    def convert_enrolled(x):
        try:
            x = str(x).lower().replace(',', '').replace('enrolled', '').strip()
            if 'k' in x:
                return float(x.replace('k', '')) * 1000
            return float(x)
        except:
            return np.nan

    df["enrolled"] = df["enrolled"].apply(convert_enrolled)
    df["enrolled"] = df["enrolled"].fillna(0)

    if df["enrolled"].nunique() <= 1:
        print("ℹ Dropping 'enrolled' column — all values are zero or the same.")
        df.drop(columns=['enrolled'], inplace=True)

# === Step 6: Extract 'duration_weeks' and 'effort_hours' from 'schedule' ===
def extract_duration_weeks(schedule_str):
    match = re.search(r"(\d+)\s*(week|weeks)", str(schedule_str).lower())
    return float(match.group(1)) if match else np.nan

def extract_effort_hours(schedule_str):
    numbers = list(map(int, re.findall(r'\d+', str(schedule_str))))
    return np.mean(numbers) if numbers else np.nan

df["duration_weeks"] = df["schedule"].apply(extract_duration_weeks)
df["effort_hours"] = df["schedule"].apply(extract_effort_hours)

# === Step 7: Convert skill_tags from string to list ===
def clean_skill_tags(tag_str):
    try:
        if isinstance(tag_str, str):
            return ast.literal_eval(tag_str)
    except:
        pass
    return []

df["skill_tags"] = df["skill_tags"].apply(clean_skill_tags)

# === Step 8: Clean text columns for NLP ===
def clean_text_nlp(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

for col in ["description", "course_detail", "full_text"]:
    if col in df.columns:
        df[col] = df[col].apply(clean_text_nlp)

# === Step 8.5: Ensure 'rating' and 'num_reviews' are numeric ===
df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
df["num_reviews"] = pd.to_numeric(df["num_reviews"], errors="coerce").fillna(0)

# === Step 8.6: Fill missing duration_weeks by median per level ===
df["duration_weeks"] = df.groupby("level")["duration_weeks"].transform(lambda x: x.fillna(x.median()))
df["has_duration"] = df["duration_weeks"].notnull().astype(int)

# === Step 9: Create Target Variable ===
df["high_rating"] = ((df["rating"] >= 4.5) & (df["num_reviews"] >= 50)).astype(int)

# === Step 10: Final Missing Value Report ===
print("🔍 Final missing values:\n", df.isnull().sum())

# === Step 11: Save Cleaned Dataset ===
output_file = "D:/PYTHON 3/ICTAK Python3/course-recommendation-project/data/raw/coursera_Social_Sciences_cleaned_final.csv"
df.to_csv(output_file, index=False)
print(f" Final cleaned dataset saved to: {output_file}")
