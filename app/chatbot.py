import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleRecommender:
    def __init__(self, model_path: str):
        self.recommender = None
        self.chat_history = []
        try:
            self.recommender = self.load_model(model_path)
            logger.info("Chatbot initialized successfully")
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def load_model(self, path):
        import pickle

        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
                if not hasattr(model, "recommend"):
                    raise ValueError("Loaded model doesn't have recommend method")
                return model
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

    def ask(self, user_input: str) -> str:
        start_time = time.time()

        if not self.recommender:
            return "System error: Recommendation engine not available. Please try again later."

        try:
            result = self.recommender.recommend(user_input)

            if not result or "recommendations" not in result:
                return "Sorry, I couldn't process your request. Please try again."

            if result.get("error"):
                return f"Error: {result['error']}"

            if not result["recommendations"]:
                return "No courses found matching your query. Please try different keywords."

            response = "Here are some course recommendations:<br><br>"
            for i, course in enumerate(result["recommendations"][:5], 1):
                link_html = (
                    f"<a href='{course['url']}' target='_blank'>{course['url']}</a>"
                )
                response += (
                    f"{i}. <b>{course['title']}</b> ({course['level']})<br>"
                    f"&nbsp;&nbsp;- Provider: {course['provider']}<br>"
                    f"&nbsp;&nbsp;- Duration: {course['duration_weeks']} weeks<br>"
                    f"&nbsp;&nbsp;- Rating: {course['rating']}<br>"
                    f"&nbsp;&nbsp;- Description: {course['description']}<br>"
                    f"&nbsp;&nbsp;- Link: {link_html}<br><br>"
                )

            response += (
                f"<br><i>Generated in {time.time() - start_time:.2f} seconds</i>"
            )
            return response

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return "Sorry, an unexpected error occurred. Please try again later."
