import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import re
from selenium.common.exceptions import NoSuchElementException
import urllib

def Typhoon_Image(year,occurence):
    s=Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=s)
    driver.get("http://agora.ex.nii.ac.jp/digital-typhoon/year/")
    link = driver.find_element_by_link_text(str(year))
    link.click()
    if occurence<10:
        link = driver.find_element_by_link_text(str(year)+ "0" + str(occurence))
    else:
        link = driver.find_element_by_link_text(str(year)+ str(occurence))
    link.click()
    div = driver.find_element_by_class_name("MINIIMG")
    href = div.find_element_by_css_selector('a').get_attribute('href')
    driver.get(href)
    for i in range(30):
        try:
            link = driver.find_element_by_link_text("< -24H")
            link.click()
        except NoSuchElementException:
            for i in range(10):
                try:
                    link = driver.find_element_by_link_text("< -6H")
                    link.click()
                except NoSuchElementException:
                    for i in range(10):
                        try:
                            link = driver.find_element_by_link_text("< -3H")
                            link.click()
                        except NoSuchElementException:
                            try:
                                for i in range(10):
                                    link = driver.find_element_by_link_text("< -1H")
                                    link.click()
                            except NoSuchElementException:
                                break
    div = driver.find_elements_by_class_name("SUMMARY")[0].text
    typhoon_name = div[div.index("Typhoon Name")+16 + len("Typhoon Name"): div.index("Basin")-1]
    typhoon_name = typhoon_name.replace('-','')
    typhoon_date = div[div.index("Observation Time")+1 + len("Observation Time"): div.index("Next Operations")-13]
    typhoon_hour = div[div.index("Observation Time")+12 + len("Observation Time"): div.index("Next Operations")-10]
    driver.find_elements_by_link_text('Magnify this')[3].click()
    typhoon_image_url = driver.current_url
    directory = typhoon_name + "_" + str(year)
    parent_dir = "C:/Users/Louis/Full Typhoon Images/"
    path = os.path.join(parent_dir, directory)
    os.mkdir(path)
    urllib.request.urlretrieve(typhoon_image_url, (path + "/"+ typhoon_name + "_"+ typhoon_date +"_"+typhoon_hour +".png"))
    driver.execute_script("window.history.go(-1)")
    while True:
        try:
            link = driver.find_element_by_link_text("+3H >")
            link.click()
            div = driver.find_elements_by_class_name("SUMMARY")[0].text
            typhoon_name = div[div.index("Typhoon Name")+16 + len("Typhoon Name"): div.index("Basin")-1]
            typhoon_name = typhoon_name.replace('-','')
            print(typhoon_name)
            typhoon_date = div[div.index("Observation Time")+1 + len("Observation Time"): div.index("Next Operations")-13]
            print(typhoon_date)
            typhoon_hour = div[div.index("Observation Time")+12 + len("Observation Time"): div.index("Next Operations")-10]
            print(typhoon_hour)
            driver.find_elements_by_link_text('Magnify this')[3].click()
            typhoon_image_url = driver.current_url 
            urllib.request.urlretrieve(typhoon_image_url, (path +"/"+ typhoon_name + "_"+ typhoon_date+ "_"+typhoon_hour +".png"))
            driver.execute_script("window.history.go(-1)")
        except IndexError:
            typhoon_year_current_website = div[div.index("Observation Time")+1 + len("Observation Time"): div.index("Next Operations")-19]
            print(typhoon_year_current_website)
            print(year)
            if int(typhoon_year_current_website) == year:
                print("equal")
                continue
            else:
                print("not equal")
                break
        except NoSuchElementException:
            print("Job Done")
            break


for i in range(2001, 2022,1):
    for j in range(1,40,1):
        try:
            print(i)
            print(j)
            Typhoon_Image(i,j)    
        except IndexError:
            continue
        except NoSuchElementException:
            break