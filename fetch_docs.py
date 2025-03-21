import csv
import requests
from bs4 import BeautifulSoup

with open("./data/littlejsdocs.txt") as file:
    urls = file.readlines()
    # Remove trailing newline characters
    urls = [url.rstrip('\n') for url in urls if not url.startswith('#')]

def parse_webpage(url: str):
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("title").get_text()
    text = soup.find("div", class_="main-wrapper")
    text.find('footer').extract()
    return [
        title,
        text.get_text(),
        url
    ]

docs = []
for url in urls:
    docs.append(parse_webpage(url))

with open('./data/littledocs.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["title", "text", "url"]

    writer.writerow(field)
    for line in docs:
        writer.writerow([line[0], line[1], line[2]])

print('DONE')
