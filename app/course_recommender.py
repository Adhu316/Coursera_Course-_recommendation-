import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Union
from ast import literal_eval
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CourseRecommender:
    def __init__(self, data_path: str):
        self.data_loaded = False
        try:
            self.skill_synonyms = self._initialize_skill_synonyms()
            self.df = self.load_and_preprocess_data(data_path)
            self._initialize_tfidf()
            self.data_loaded = True
            logger.info("Recommender initialized successfully")
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def _initialize_skill_synonyms(self) -> Dict[str, List[str]]:
        return {
            "Data & Analytics": [
                "Data Analysis",
                "Analytics",
                "Business Intelligence",
                "Data Science",
                "Machine Learning",
                "Artificial Intelligence",
                "Big Data",
                "Data Management",
                "Statistical Analysis",
                "Data Visualization",
            ],
            "Programming & Development": [
                "Software Development",
                "Programming",
                "Web Development",
                "Cloud Computing",
                "Cybersecurity",
                "Databases",
                "API",
                "DevOps",
                "System Design",
                "Mobile Development",
            ],
            "Marketing & Sales": [
                "Digital Marketing",
                "Marketing Strategy",
                "Advertising",
                "Branding",
                "Sales",
                "Content Marketing",
                "Social Media Marketing",
                "Market Research",
                "Customer Relationship Management",
                "E-commerce",
            ],
            "Project & Operations Management": [
                "Project Management",
                "Operations Management",
                "Agile Methodology",
                "Supply Chain Management",
                "Process Improvement",
                "Lean Methodologies",
                "Risk Management",
                "Strategic Planning",
                "Resource Management",
                "Quality Management",
            ],
            "Business & Finance": [
                "Business Strategy",
                "Financial Management",
                "Accounting",
                "Economics",
                "Entrepreneurship",
                "Investment",
                "Corporate Finance",
                "Risk Management",
                "Financial Analysis",
                "Business Ethics",
            ],
            "Human Resources & Organizational Development": [
                "Human Resources",
                "Talent Management",
                "Recruitment",
                "Employee Engagement",
                "Organizational Development",
                "Leadership Development",
                "Performance Management",
                "Diversity and Inclusion",
                "Compensation Management",
                "Team Management",
            ],
            "Healthcare & Clinical": [
                "Public Health",
                "Nursing Practices",
                "Patient Care",
                "Medical Emergency",
                "Mental Health",
                "Pharmacology",
                "Epidemiology",
                "Health Informatics",
                "Clinical Leadership",
                "Preventative Care",
            ],
            "Design & Creative Arts": [
                "Graphic Design",
                "UI/UX Design",
                "User Experience Design",
                "Web Design",
                "Animation",
                "Photography",
                "Video Production",
                "Storytelling",
                "Creative Thinking",
                "Game Design",
            ],
            "Communication & Soft Skills": [
                "Communication",
                "Problem Solving",
                "Collaboration",
                "Leadership",
                "Critical Thinking",
                "Decision Making",
                "Negotiation",
                "Presentation Skills",
                "Adaptability",
                "Teamwork",
            ],
            "Research & Analysis": [
                "Research Methodologies",
                "Data Analysis",
                "Qualitative Research",
                "Quantitative Research",
                "Market Research",
                "Statistical Analysis",
                "Business Analysis",
                "System Analysis",
                "Competitive Analysis",
                "Policy Analysis",
            ],
            "Cybersecurity & IT Operations": [
                "Cybersecurity",
                "Network Security",
                "Cloud Computing",
                "Information Security",
                "Threat Detection",
                "Access Management",
                "Security Controls",
                "Vulnerability Management",
                "Incident Response",
                "IT Operations",
            ],
            "Product & UX/UI": [
                "Product Management",
                "Product Design",
                "User Experience Design",
                "User Interface Design",
                "User Research",
                "Prototyping",
                "User Story",
                "Product Strategy",
                "Usability Testing",
                "Service Design",
            ],
            "Education & Training": [
                "Training and Development",
                "Education Technology",
                "Professional Development",
                "Learning and Development",
                "Instructional Design",
                "Mentorship",
                "Coaching",
                "Curriculum Development",
                "Learning Management Systems",
                "Educational Planning",
            ],
            "Legal & Compliance": [
                "Regulatory Compliance",
                "Legal Research",
                "Contract Management",
                "Data Privacy",
                "Intellectual Property",
                "Corporate Governance",
                "Risk Management",
                "Ethical Standards",
                "Tax Compliance",
                "Labor Law",
            ],
        }

    def safe_literal_eval(self, x: str) -> List[str]:
        try:
            return literal_eval(x) if isinstance(x, str) else []
        except (ValueError, SyntaxError):
            return []

    def load_and_preprocess_data(self, path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(path)
            logger.info(f"Data loaded successfully with {len(df)} records")

            required_columns = [
                "title",
                "description",
                "skill_tags",
                "duration_weeks",
                "effort_hours",
                "level",
                "price",
                "rating",
                "provider",
                "url",
            ]
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            df = df.dropna(subset=["title", "description"])
            df["description"] = df["description"].fillna("")
            df["cleaned_description"] = df["description"].astype(str).str.lower()
            df["cleaned_title"] = df["title"].astype(str).str.lower()
            df["skill_tags"] = df["skill_tags"].apply(self.safe_literal_eval)

            df["expanded_skills"] = df["skill_tags"].apply(
                lambda tags: list(
                    set(
                        [tag for tag in tags if isinstance(tag, str)]
                        + [
                            new_term
                            for tag in tags
                            if isinstance(tag, str)
                            for new_term in self.skill_synonyms.get(tag, [])[:2]
                        ]
                    )
                )
            )

            df["combined_text"] = (
                df["cleaned_title"]
                + " "
                + df["expanded_skills"].apply(lambda x: " ".join(x) if x else "")
                + " "
                + df["cleaned_description"]
            ).str.strip()

            df["duration_weeks"] = pd.to_numeric(
                df["duration_weeks"], errors="coerce"
            ).fillna(2)
            df["effort_hours"] = pd.to_numeric(
                df["effort_hours"], errors="coerce"
            ).fillna(5)

            df["level"] = df["level"].astype(str).str.lower()
            level_map = {
                "beginner level": 1,
                "intermediate level": 2,
                "advanced level": 3,
            }
            df["difficulty_score"] = df["level"].map(level_map).fillna(1)

            df["price_numeric"] = (
                df["price"].replace(r"[\$,]", "", regex=True).astype(float).fillna(0)
            )
            df["is_free"] = df["price_numeric"].apply(lambda x: 1 if x == 0 else 0)
            max_price = df["price_numeric"].max()
            df["price_normalized"] = (
                df["price_numeric"] / max_price if max_price > 0 else 0
            )

            df["rating"] = (
                pd.to_numeric(df["rating"], errors="coerce").clip(1, 5).fillna(3)
            )

            df["category"] = (
                df["category"].astype(str).str.lower()
                if "category" in df
                else "unknown"
            )

            return df

        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise

    def _initialize_tfidf(self):
        """Initialize TF-IDF vectorizer and matrix"""
        try:
            self.tfidf = TfidfVectorizer(
                stop_words="english",
                max_features=5000,
                min_df=2,
                max_df=0.8,
                ngram_range=(1, 2),
            )
            self.tfidf_matrix = self.tfidf.fit_transform(self.df["combined_text"])
            logger.info("TF-IDF matrix created successfully")
        except Exception as e:
            logger.error(f"TF-IDF initialization failed: {str(e)}")
            raise

    def recommend(self, prompt: str, top_n: int = 5) -> Dict:
        """Generate course recommendations based on input prompt"""
        if not self.data_loaded:
            return {
                "error": "Recommendation system not properly initialized",
                "recommendations": [],
            }

        try:
            if not prompt or not isinstance(prompt, str):
                return {"error": "Invalid input prompt", "recommendations": []}

            processed_prompt = prompt.lower().strip()
            if len(processed_prompt) < 3:
                return {"error": "Input too short", "recommendations": []}

            prompt_vec = self.tfidf.transform([processed_prompt])
            cos_sim = cosine_similarity(prompt_vec, self.tfidf_matrix)
            top_indices = np.argsort(cos_sim[0])[-top_n*2:][::-1]  

            recommendations = []
            for idx in top_indices:
                try:
                    course = self.df.iloc[idx]
                    recommendations.append(
                        {
                            "title": course.get("title", "No title"),
                            "provider": course.get("provider", "Unknown provider"),
                            "url": course.get("url", "#"),
                            "description": (
                                (course.get("description", "")[:150] + "...")
                                if course.get("description")
                                else "No description"
                            ),
                            "level": course.get("level", "Unknown level").title(),
                            "difficulty_score": int(course.get("difficulty_score", 1)),
                            "duration_weeks": int(course.get("duration_weeks", 0)),
                            "effort_hours": int(course.get("effort_hours", 0)),
                            "rating": float(course.get("rating", 0)),
                            "price": (
                                f"${course.get('price_numeric', 0):.2f}"
                                if course.get("price_numeric", 0) > 0
                                else "Free"
                            ),
                            "similarity_score": float(cos_sim[0][idx]),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error processing course at index {idx}: {str(e)}")
                    continue

            recommendations.sort(key=lambda x: (x["difficulty_score"], -x["similarity_score"]))

            return {
                "recommendations": recommendations[:top_n],
                "status": "success",
                "count": len(recommendations[:top_n]),
                "query": prompt,
            }

        except Exception as e:
            logger.error(f"Recommendation failed: {str(e)}")
            return {"error": str(e), "recommendations": []}
