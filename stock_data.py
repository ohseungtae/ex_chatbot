import pandas as pd
import yfinance as yf
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import time
import unicodedata

def get_recent_trading_day():
    """
    가장 최근 거래일을 구하는 함수

    Returns:
        str: 최근 거래일(YYYY-MM-DD 형식)
    """
    today = datetime.now()
    if today.hour < 9:  # 9시 이전이면 전날을 기준으로
        today -= timedelta(days=1)
    while today.weekday() in [5, 6]:  # 토요일(5), 일요일(6)이면 하루씩 감소
        today -= timedelta(days=1)
    return today.strftime('%Y-%m-%d')


def get_ticker(company, source="yahoo"):
    """
    기업명으로부터 증권 코드를 찾는 함수

    Args:
        company (str): 기업명
        source (str): 데이터 소스 ("yahoo" 또는 "fdr")

    Returns:
        str: 티커 코드
    """
    try:
        # 데이터 로드
        listing = fdr.StockListing('KRX')

        # 입력된 회사명 정규화
        normalized_company = company.strip().lower().replace(" ", "")

        # 부분 일치 및 전체 일치 검색
        exact_match = listing[listing['Name'].str.strip() == company.strip()]
        partial_match = listing[
            listing['Name']
            .str.strip()
            .str.lower()
            .str.replace(" ", "")
            .str.contains(normalized_company)
        ]

        # 매칭 로직
        if not exact_match.empty:
            ticker_row = exact_match.iloc[0]
        elif not partial_match.empty:
            ticker_row = partial_match.iloc[0]
        else:
            print("일치하는 기업을 찾을 수 없습니다.")
            return None

        # 티커 코드 생성
        krx_ticker = str(ticker_row["Code"]).zfill(6)

        if source == "yahoo":
            return krx_ticker + ".KS"  # 야후 파이낸스용 티커 변환
        return krx_ticker  # FinanceDataReader용 티커

    except Exception as e:
        print(f"티커 조회 중 오류 발생: {e}")
        return None
# 📌 네이버 Fchart API에서 분봉 데이터 가져오기 (최신 거래일 탐색 포함)
def get_naver_fchart_minute_data(stock_code, minute="1", days=1):
    """
    네이버 금융 Fchart API에서 분봉 데이터를 더 효율적으로 가져오기

    Args:
        stock_code (str): 종목 코드
        minute (str): 분 단위 (기본 1분)
        days (int): 조회 일수

    Returns:
        pd.DataFrame: 분봉 데이터
    """
    now = datetime.now()

    if now.hour < 9:
        now -= timedelta(days=1)

    # 📌 최신 거래일 찾기 (공휴일 대응)
    while True:
        target_date = now.strftime("%Y-%m-%d") if days == 1 else None
        url = f"https://fchart.stock.naver.com/sise.nhn?symbol={stock_code}&timeframe=minute&count={days * 78}&requestType=0"
        response = requests.get(url)

        if response.status_code != 200:
            return pd.DataFrame()  # 요청 실패 시 빈 데이터 반환

        soup = BeautifulSoup(response.text, "lxml")

        data_list = []
        for item in soup.find_all("item"):
            values = item["data"].split("|")
            if len(values) < 6:
                continue

            time_str, _, _, _, close, _ = values
            if close == "null":
                continue

            time_val = datetime.strptime(time_str, "%Y%m%d%H%M")
            close = float(close)

            if target_date:
                if time_val.strftime("%Y-%m-%d") == target_date:
                    data_list.append([time_val, close])
            else:
                data_list.append([time_val, close])

        df = pd.DataFrame(data_list, columns=["시간", "종가"])

        # 📌 ✅ 9시 ~ 15시 30분 데이터만 필터링
        df["시간"] = pd.to_datetime(df["시간"])
        df = df[(df["시간"].dt.time >= time(9, 0)) & (df["시간"].dt.time <= time(15, 30))]

        # ✅ 데이터가 없는 경우 → 하루 전으로 이동하여 다시 시도
        if df.empty:
            now -= timedelta(days=1)
            while now.weekday() in [5, 6]:  # 토요일(5) 또는 일요일(6)
                now -= timedelta(days=1)
        else:
            break  # 데이터를 찾았으면 반복 종료

    return df

# 📌 FinanceDataReader를 통해 일별 시세를 가져오는 함수
def get_daily_stock_data_fdr(ticker, period):
    """
    FinanceDataReader를 통해 일별 시세를 가져오는 함수
    Args:
        ticker (str): 티커 코드
        period (str): 기간 ("1month" 또는 "1year")
    Returns:
        DataFrame: 주식 데이터
    """
    try:
        end_date = get_recent_trading_day()
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(
            days=30 if period == "1month" else 365)).strftime('%Y-%m-%d')
        df = fdr.DataReader(ticker, start_date, end_date)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[df["Date"].dt.weekday < 5].reset_index(drop=True)  # ✅ 주말 데이터 제거
        return df
    except Exception as e:
        st.error(f"FinanceDataReader 데이터 불러오기 오류: {e}")
        return pd.DataFrame()


def standardize_company_name(company_name):
    """
    회사명을 표준화하는 함수

    Args:
        company_name (str): 입력된 회사명

    Returns:
        str: 표준화된 회사명
    """
    # 유니코드 정규화
    normalized_name = unicodedata.normalize('NFC', company_name)

    # 단어별로 나누어 첫 글자를 대문자로 변경
    words = normalized_name.split()
    standardized_words = [
        word.upper() if word.isalpha() and len(word) <= 2 else
        word.capitalize()
        for word in words
    ]

    return ' '.join(standardized_words).strip()
