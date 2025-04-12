import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import urllib.parse

# 타겟 URL
url = "https://namu.wiki/w/%EB%B6%84%EB%A5%98:%ED%95%9C%EA%B5%AD%20%EB%AF%BC%EC%A3%BC%ED%99%94%EC%9A%B4%EB%8F%99"

# 웹페이지 요청
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# 내부 링크 추출
links = set()
for a in soup.find_all("a", href=True):
    href = a["href"]
    full_url = urljoin(url, href)
    if full_url.startswith("https://namu.wiki/w/"):
        links.add(full_url)

# 링크를 정렬하고, 주석 달기
lines = sorted(links)
formatted = []
for line in lines:
    path = line.strip().split("/w/")[-1]
    decoded_title = urllib.parse.unquote(path)
    formatted.append(f'    "{line}",  # {decoded_title}')

print("namuwiki_links = [\n" + "\n".join(formatted) + "\n]")
