from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get("https://www.slickcharts.com/sp500")
sp500_names = []
for i in range(1,506):
    name = driver.find_element_by_xpath(f'/html/body/div[2]/div[3]/div[1]/div/div/table/tbody/tr[{i}]/td[3]/a').text
    sp500_names.append(name)
print(sp500_names)