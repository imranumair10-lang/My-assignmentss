from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
import os

url = "https://www.amazon.com/gp/browse.html?node=6563140011"
chrome_driver_path = r"C:\Users\HP\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe"

service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service)

driver.get(url)
wait = WebDriverWait(driver, 10)
cards = wait.until(EC.presence_of_all_elements_located((By.XPATH, "//a[.//img]")))

categories = []
for card in cards:
    category = {}
    category["url"] = card.get_attribute("href")

    try:
        img = card.find_element(By.TAG_NAME, "img")
        category["img"] = img.get_attribute("src")
        category["category_name"] = img.get_attribute("alt")
    except:
        category["img"] = ""
        category["category_name"] = ""

    parent_div = card.find_element(By.XPATH, "./ancestor::div[1]")
    category["div_id"] = parent_div.get_attribute("id")
    category["div_class"] = parent_div.get_attribute("class")


    if category["category_name"]:
        categories.append(category)

os.makedirs("Week6", exist_ok=True)

filename = "Week6/amazon_smart_home_categories.csv"
with open(filename, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(
        f,
        ["category_name", "url", "img", "div_id", "div_class", "price"]
    )
    w.writeheader()
    for c in categories:
        w.writerow(c)
driver.quit()

print(f"Saved {len(categories)} categories to {filename}")
