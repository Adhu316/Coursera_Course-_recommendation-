# Stage 1: Collect Data Science course URLs with checkpointing
import time, random
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import os

# Setup headless browser with random user-agent
options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument(f"user-agent=Mozilla/5.0")
driver = webdriver.Chrome(options=options)

# Save path for checkpoint
SAVE_PATH = "D:/PYTHON 3/ICTAK Python3/course-recommendation-project/data/raw/coursera_Information_echnology_urls.csv"

# Load checkpoint if exists
if os.path.exists(SAVE_PATH):
    seen_df = pd.read_csv(SAVE_PATH)
    seen = set(seen_df['url'].dropna().tolist())
    course_urls = list(seen)
    print(f" Resuming from checkpoint with {len(seen)} URLs.")
else:
    seen = set()
    course_urls = []

# Base URL with topic=Data Science
base_url = "https://www.coursera.org/courses?query=data%20science&topic=Information%20Technology&page="

# Scroll through paginated results
for page in range(1, 1000):  
    url = base_url + str(page)
    driver.get(url)
    time.sleep(random.uniform(2, 4))
    
    cards = driver.find_elements(By.XPATH, "//a[contains(@href, '/learn/') or contains(@href, '/professional-certificates/')]")
    new_links = 0
    for card in cards:
        href = card.get_attribute("href")
        if href:
            clean_url = href.split("?")[0]
            if clean_url not in seen:
                seen.add(clean_url)
                course_urls.append(clean_url)
                new_links += 1
    
    print(f" Page {page}: Found {new_links} new URLs")
    
    # Save progress after each page
    pd.DataFrame({"url": list(seen)}).to_csv(SAVE_PATH, index=False)

    if new_links == 0:
        print(" No new URLs — stopping.")
        break

driver.quit()
print(f" Final count: {len(course_urls)} Data Science course URLs saved to '{SAVE_PATH}'")
