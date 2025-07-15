import re
import random
import time
import socket
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

SAVE_PATH = "D:/PYTHON 3/ICTAK Python3/course-recommendation-project/data/raw/coursera_Social_Sciences_cleaned.csv"
URL_CSV_PATH = "D:/PYTHON 3/ICTAK Python3/course-recommendation-project/data/raw/coursera_Social_Sciences_urls.csv"

user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)...",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...",
    "Mozilla/5.0 (X11; Linux x86_64)...",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0)...",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 13_6_1...)"
]

# Lock for thread-safe data appending
lock = threading.Lock()

def wait_random(min_sec=2, max_sec=4):
    time.sleep(random.uniform(min_sec, max_sec))

def setup_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(f"user-agent={random.choice(user_agents)}")
    prefs = {
        "profile.managed_default_content_settings.images": 2,
        "profile.managed_default_content_settings.stylesheets": 2,
        "profile.managed_default_content_settings.fonts": 2
    }
    options.add_experimental_option("prefs", prefs)
    return webdriver.Chrome(options=options)

# ------------- Your Original Function, Now Accepts a Driver Instance -------------
def extract_course_data(url):
    driver = setup_driver()
    try:
        driver.get(url)
        wait_random()

        def safe_xpath(xpath):
            try:
                return driver.find_element(By.XPATH, xpath).text.strip()
            except:
                return ""

        def extract_provider():
            try:
                return driver.find_element(By.XPATH, "//div[contains(@class,'css-1ujzbfc')]//img[@alt]").get_attribute("alt").strip()
            except:
                try:
                    return driver.find_element(By.XPATH, "//img[contains(@class, 'css-1f9gt0j') and @alt]").get_attribute("alt").strip()
                except:
                    return "Coursera"
    
        def extract_num_reviews():
            pattern = r"\(([\d,]+)\s+reviews?\)"
            try:
                review_text = driver.find_element(By.XPATH, "//p[contains(@class,'css-vac8rf')]").text.strip()
                match = re.search(pattern, review_text)
                if match:
                    return int(match.group(1).replace(",", ""))
            except:
                pass
            try:
                blocks = driver.find_elements(By.XPATH, "//div[contains(@class, 'css-dwgey1')]//p")
                for block in blocks:
                    match = re.search(pattern, block.text)
                    if match:
                        return int(match.group(1).replace(",", ""))
            except:
                pass
            return None

        def extract_enrolled():
            try:
                enrolled_elem = driver.find_element(By.XPATH, "//div[contains(@class,'css-1qi3xup')]//span/strong/span")
                return int(enrolled_elem.text.strip().replace(",", ""))
            except:
                try:
                    divs = driver.find_elements(By.XPATH, "//div[contains(@class,'css-1qi3xup')]")
                    for div in divs:
                        match = re.search(r"([\d,]+)\s+already enrolled", div.text)
                        if match:
                            return int(match.group(1).replace(",", ""))
                except:
                    pass
            return None

        def extract_instructor():
            try:
                instructors = driver.find_elements(By.XPATH, "//a[contains(@href, '/instructor/') and @class]")
                names = list(set(i.text.strip() for i in instructors if i.text.strip()))
                return ", ".join(names)
            except:
                return ""

        def extract_description():
            try:
                blocks = driver.find_elements(By.XPATH, "//div[contains(@class,'css-12wvpfc')]/p[contains(@class,'css-4s48ix')]")
                for block in blocks:
                    text = block.text.strip()
                    if text and not text.lower().startswith("instructors:"):
                        return text
            except:
                pass
            try:
                fallback = driver.find_elements(By.XPATH, "//div[@data-testid='cml-viewer']//p")
                return next((p.text.strip() for p in fallback if p.text.strip()), "")
            except:
                return ""

        def extract_schedule():
            try:
                texts = []
                blocks = driver.find_elements(By.XPATH, "//div[contains(@class,'css-dwgey1')]")
                for block in blocks:
                    for line in block.text.strip().split("\n"):
                        if any(word in line.lower() for word in ["week", "month", "hour"]):
                            texts.append(line.strip())
                return " ".join(texts)
            except:
                return ""

        def extract_rating():
            try:
                return driver.find_element(By.XPATH, "//div[@aria-label and contains(@aria-label, 'stars')]").text.strip()
            except:
                try:
                    return driver.find_element(By.XPATH, "//div[@role='group']").text.strip().split()[0]
                except:
                    return ""

        def extract_image_url():
            xpaths = [
                "//img[contains(@class, 'css-1f9gt0j')]",
                "//img[contains(@class, 'css-169m40h')]",
                "//div[contains(@class,'css-1ujzbfc')]//img[@alt]",
                "//img[contains(@src, 'coursera')]"
            ]
            for xpath in xpaths:
                try:
                    img = driver.find_element(By.XPATH, xpath)
                    src = img.get_attribute("src") or img.get_attribute("data-src")
                    if src:
                        return src.strip()
                except:
                    continue
            return ""

        def extract_certificate():
            try:
                divs = driver.find_elements(By.XPATH, "//div[contains(@class, 'css-1qfxccv')]")
                for div in divs:
                    if "certificate" in div.text.lower():
                        return "Yes"
                return "No"
            except:
                return "No"

        def extract_course_detail():
            try:
                blocks = driver.find_elements(By.XPATH, "//div[contains(@class,'cds-grid-item')]//div[@class='content-inner']//p")
                return "\n".join([p.text.strip() for p in blocks if p.text.strip()])
            except:
                return ""

        def extract_language():
            try:
                elem = driver.find_element(By.XPATH, "//div[contains(@class,'css-drc7pp') or contains(@class,'css-onm9p2')]/span")
                return elem.text.replace("Taught in", "").strip()
            except:
                return ""

        def extract_skill_tags():
            try:
                skills = driver.find_elements(By.XPATH, "//div[contains(@class,'css-1m3kxpf')]//ul//li//a")
                return list(set(s.text.strip() for s in skills if s.text.strip()))
            except:
                return []

        # Collect fields
        title = safe_xpath("//h1")
        provider = extract_provider()
        rating = extract_rating()
        num_reviews = extract_num_reviews()
        instructor = extract_instructor()
        level = safe_xpath("//div[contains(@class,'css-fk6qfz') and contains(text(),'level')]") or "Beginner"
        language = extract_language()
        schedule = extract_schedule()
        description = extract_description()
        image_url = extract_image_url()
        certificate = extract_certificate()
        course_detail = extract_course_detail()
        skill_tags = extract_skill_tags()
        enrolled = extract_enrolled()
        price = "Free"

        full_text = " | ".join([
            title, description, course_detail,
            "Skills: " + ", ".join(skill_tags),
            "Schedule: " + schedule,
            "Instructor: " + instructor
        ]).strip()

        try:
            high_rating = 1 if float(rating) >= 4.5 and num_reviews and num_reviews >= 50 else 0
        except:
            high_rating = 0

        return {
            "title": title, "provider": provider, "rating": rating, "num_reviews": num_reviews, "url": url,
            "schedule": schedule, "level": level, "language": language,
            "price": price, "description": description, "course_detail": course_detail,
            "certificate": certificate, "instructor": instructor, "full_text": full_text,
            "high_rating": high_rating, "skill_tags": skill_tags, "image_url": image_url,
            "enrolled": enrolled
        }
    except Exception as e:
        print(f"Failed at {url}: {e}")
        return None
    finally:
        driver.quit()

# ---------------------
# Main Parallel Loop
# ---------------------
course_links = pd.read_csv(URL_CSV_PATH)["url"].dropna().tolist()
results = []
BATCH_SIZE = 10

with ThreadPoolExecutor(max_workers=5) as executor:
    future_to_url = {executor.submit(extract_course_data, url): url for url in course_links}
    for i, future in enumerate(as_completed(future_to_url), 1):
        result = future.result()
        if result:
            with lock:
                results.append(result)
        if i % BATCH_SIZE == 0:
            pd.DataFrame(results).to_csv(SAVE_PATH, index=False)
            print(f"Saved {i} records")

# Final Save
df = pd.DataFrame(results)
df["description"] = df["description"].astype(str).str.encode("latin1", errors="ignore").str.decode("utf-8", errors="ignore")
df["full_text"] = df["full_text"].astype(str).str.encode("latin1", errors="ignore").str.decode("utf-8", errors="ignore")
df.to_csv(SAVE_PATH, index=False)
print("Scraping complete.")
