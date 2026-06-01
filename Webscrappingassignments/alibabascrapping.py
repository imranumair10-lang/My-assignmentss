from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
import os

url = "https://www.alibaba.com/trade/search?spm=a2700.product_home_newuser.header.132.2ce267afSeLPmg&SearchText=Auto+Accessories&indexArea=product_en&search_cource_scene=pc_home_product_category&has4Tab=true&tab=all"

driver = webdriver.Chrome()

driver.get(url)
wait = WebDriverWait(driver, 15)


cards = wait.until(
    EC.presence_of_all_elements_located((By.XPATH, "//a[.//img]"))
)

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

    try:
        parent_div = card.find_element(By.XPATH, "./ancestor::div[1]")
        category["div_id"] = parent_div.get_attribute("id")
        category["div_class"] = parent_div.get_attribute("class")
    except:
        category["div_id"] = ""
        category["div_class"] = ""

    if category["category_name"]:
        categories.append(category)

driver.quit()

os.makedirs("Week6", exist_ok=True)

filename = "Week6/alibabaproducts.csv"

with open(filename, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(
        f,
        fieldnames=["category_name", "url", "img", "div_id", "div_class", "price"]
    )
    w.writeheader()

    for c in categories:
        c["price"] = ""  
        w.writerow(c)

print(f"Saved {len(categories)} categories to {filename}")