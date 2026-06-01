from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
import os


url = "https://www.daraz.pk/catalog/?spm=a2a0e.tm80331704.cate_5.5.77cc5aa7fPImi7&q=Smart%20Phones&from=hp_categories&src=all_channel"
chrome_driver_path = r"C:\Users\HP\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe"

service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service)

driver.get(url)
wait = WebDriverWait(driver, 10)
products_elements = driver.find_elements(By.XPATH, "//div[@data-qa-locator='product-item']")
products = []  

for p in range(len(products_elements)):
    quote = {}
    innerImg = products_elements[p].find_element(By.TAG_NAME, "img")
    innera = products_elements[p].find_element(By.TAG_NAME, "a")
    quote["img"] = innerImg.get_attribute('src')
    quote["lines"] = innerImg.get_attribute('alt')
    quote["url"] = innera.get_attribute('href')
    products.append(quote)

filename = 'Week6/daraz_products.csv'
with open(filename, 'w', newline='') as f:
    w = csv.DictWriter(f, ['url','img','lines'])
    w.writeheader()
    for quote in products:
       w.writerow(quote)



