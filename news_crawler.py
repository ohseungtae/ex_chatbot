import requests
import urllib.parse
import random
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def jaccard_similarity(str1, str2):
    """Jaccard 유사도 계산 함수"""
    set1, set2 = set(str1.split()), set(str2.split())
    return len(set1 & set2) / len(set1 | set2)


def crawl_news(company, days):
    today = datetime.today()
    start_date = (today - timedelta(days=days)).strftime('%Y%m%d')
    end_date = today.strftime('%Y%m%d')
    encoded_query = urllib.parse.quote(company)

    url_template = f"https://search.naver.com/search.naver?where=news&query={encoded_query}&nso=so:r,p:from{start_date}to{end_date}&start={{}}"

    headers = {
        "User-Agent": random.choice([
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36"
        ])
    }

    news = []
    seen_urls = set()
    seen_titles = []
    seen_contents = []

    for page in range(1, 6):  # 1~5 페이지 크롤링
        url = url_template.format((page - 1) * 10 + 1)
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.select("ul.list_news > li")

        for article in articles:
            title_elem = article.select_one("a.news_tit")
            content_elem = article.select_one("div.news_dsc")

            if not title_elem:
                continue

            title = title_elem.text.strip()
            link = title_elem['href']
            content = content_elem.text.strip() if content_elem else ""

            # ✅ 1. URL 중복 검사
            if link in seen_urls:
                continue
            seen_urls.add(link)

            # ✅ 2. 제목 중복 검사 (TF-IDF 기반 유사도 체크)
            is_duplicate_title = False
            if seen_titles:
                vectorizer = TfidfVectorizer().fit_transform([title] + seen_titles)
                similarity_scores = cosine_similarity(vectorizer[0], vectorizer[1:]).flatten()
                if any(score > 0.1 for score in similarity_scores):  # 10% 이상 유사하면 중복으로 판단
                    is_duplicate_title = True

            if is_duplicate_title:
                continue
            seen_titles.append(title)

            # ✅ 3. 본문 유사도 검사 (Jaccard Similarity)
            is_duplicate_content = False
            for existing_content in seen_contents:
                if jaccard_similarity(content, existing_content) > 0.05:  # 5% 이상 유사하면 중복 처리
                    is_duplicate_content = True
                    break

            if is_duplicate_content:
                continue
            seen_contents.append(content)

            # ✅ 4. 본문이 너무 짧거나 없는 경우 제외
            if len(content) < 20:  # 20자 이하는 광고성, 불완전 기사일 가능성 높음
                continue

            news.append({"title": title, "link": link, "content": content})

    return news