import requests
import os
from bs4 import BeautifulSoup

data_dir = "data"
os.makedirs("data", exist_ok=True)

url = "https://www.gutenberg.org/cache/epub/35/pg35.txt"\

response = requests.get(url)
if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()
else:
    print(f"Error: {response.status_code}")

with open('data/timemachine.txt', 'w', encoding='utf-8') as file:
    file.write(text)