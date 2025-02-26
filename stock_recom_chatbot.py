import streamlit as st
import requests
import urllib.parse
import FinanceDataReader as fdr
import tiktoken
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import re
import time


def main():
    st.set_page_config(page_title="기업 투자 분석 어시스턴트", page_icon=":chart_with_upwards_trend:")
    st.title("📊 기업 투자 분석 어시스턴트")

    # 세션 상태 초기화
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = False
    if "news_data" not in st.session_state:
        st.session_state.news_data = None
    if "company_name" not in st.session_state:
        st.session_state.company_name = None
    if "selected_period" not in st.session_state:
        st.session_state.selected_period = "1day"
    if "company_summary" not in st.session_state:
        st.session_state.company_summary = None

    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        company_name = st.text_input("분석할 기업명 (코스피 상장)")
        process = st.button("분석 시작")

    if process:
        if not openai_api_key or not company_name:
            st.info("OpenAI API 키와 기업명을 입력해주세요.")
            st.stop()

        # 새 분석 시작 시 이전 대화 내역 초기화
        st.session_state.chat_history = []

        with st.spinner(f"🔍 {company_name}에 대한 정보 수집 중..."):
            news_data = crawl_news(company_name)
            if not news_data:
                st.warning("해당 기업의 최근 뉴스를 찾을 수 없습니다.")
                st.stop()

            # 분석 결과를 session_state에 저장
            st.session_state.news_data = news_data
            st.session_state.company_name = company_name

            text_chunks = get_text_chunks(news_data)
            vectorstore = get_vectorstore(text_chunks)

            st.session_state.conversation = create_chat_chain(vectorstore, openai_api_key)

            # 기업 정보 요약 생성
            st.session_state.company_summary = generate_company_summary(company_name, news_data, openai_api_key)

            st.session_state.processComplete = True

    # 분석 결과가 있으면 항상 상단에 출력
    if st.session_state.processComplete and st.session_state.company_name:
        # 기업 정보 요약 표시
        if st.session_state.company_summary:
            st.markdown(st.session_state.company_summary, unsafe_allow_html=True)

        # 주가 차트 표시
        st.subheader(f"📈 {st.session_state.company_name} 주가 추이")
        selected_period = st.radio(
            "기간 선택",
            options=["1day", "week", "1month", "1year"],
            horizontal=True,
            index=["1day", "week", "1month", "1year"].index(st.session_state.selected_period)
        )
        if selected_period != st.session_state.selected_period:
            st.session_state.selected_period = selected_period

        with st.spinner(f"📊 {st.session_state.company_name} ({st.session_state.selected_period}) 데이터 불러오는 중..."):
            if selected_period in ["1day", "week"]:
                ticker = get_ticker(st.session_state.company_name, source="yahoo")
                if not ticker:
                    st.error("해당 기업의 야후 파이낸스 티커 코드를 찾을 수 없습니다.")
                    return

                interval = "1m" if selected_period == "1day" else "5m"
                df = get_intraday_data_yahoo(ticker, period="5d" if selected_period == "week" else "1d",
                                             interval=interval)
            else:
                ticker = get_ticker(st.session_state.company_name, source="fdr")
                if not ticker:
                    st.error("해당 기업의 FinanceDataReader 티커 코드를 찾을 수 없습니다.")
                    return

                df = get_daily_stock_data_fdr(ticker, selected_period)

            if df.empty:
                st.warning(
                    f"📉 {st.session_state.company_name} - 해당 기간({st.session_state.selected_period})의 거래 데이터가 없습니다.")
            else:
                plot_stock_plotly(df, st.session_state.company_name, st.session_state.selected_period)

        # 대화 인터페이스
        if not st.session_state.chat_history:
            st.markdown("""
            ### 💬 어떤 정보가 궁금하신가요?
            * 이 기업의 최근 실적은 어떤가요?
            * 현재 주가가 과대평가된 것 같나요?
            * 이 기업의 향후 성장 전망은 어떤가요?
            * 현재 시장 상황에서 투자 전략을 조언해주세요.
            """)
        else:
            st.markdown("### 💬 질문과 답변")

        # 대화 히스토리 표시
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                # HTML 형식으로 변환된 마크다운 콘텐츠 표시
                if message["role"] == "assistant":
                    st.markdown(message["content"], unsafe_allow_html=True)
                else:
                    st.markdown(message["content"])

                # 소스 문서 표시 (응답인 경우에만)
                if message["role"] == "assistant" and "source_documents" in message:
                    with st.expander("참고 뉴스 확인"):
                        for doc in message["source_documents"]:
                            st.markdown(f"- [{doc.metadata['source']}]({doc.metadata['source']})")

        # 채팅 입력: 사용자가 질문을 입력하면 대화가 이어짐
        if query := st.chat_input("질문을 입력해주세요."):
            # 사용자 메시지 추가
            st.session_state.chat_history.append({"role": "user", "content": query})

            # 응답 생성
            with st.chat_message("assistant"):
                with st.spinner("분석 중..."):
                    result = st.session_state.conversation({"question": query})
                    response = result['answer']

                    # 응답 강조 및 이모지 추가 처리
                    response = enhance_llm_response(response)

                    # 응답 표시 (HTML 허용)
                    st.markdown(response, unsafe_allow_html=True)

                    # 소스 문서 표시
                    with st.expander("참고 뉴스 확인"):
                        for doc in result['source_documents']:
                            st.markdown(f"- [{doc.metadata['source']}]({doc.metadata['source']})")

            # 응답을 대화 히스토리에 추가
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "source_documents": result.get('source_documents', [])
            })

            # 자동으로 페이지 새로고침 없이 대화 내용 업데이트
            st.rerun()


# LLM 응답 강화 함수 (이모지, 강조 등 추가)
def enhance_llm_response(text):
    # 섹션 제목에 이모지 추가
    text = re.sub(r'(## 최신 뉴스|## 뉴스 요약|## 최근 동향)', r'## 📰 \1', text)
    text = re.sub(r'(## 투자 전망|## 투자 분석|## 전망)', r'## 💹 \1', text)
    text = re.sub(r'(## 위험 요소|## 부정적 요인|## 리스크)', r'## ⚠️ \1', text)
    text = re.sub(r'(## 긍정적 요인|## 성장 기회|## 기회)', r'## ✅ \1', text)
    text = re.sub(r'(## 재무 분석|## 재무 상태|## 재무)', r'## 💰 \1', text)

    # 번호 매기기 강화 (1️⃣, 2️⃣, 3️⃣ 등)
    text = re.sub(r'(?m)^1\. ', r'1️⃣ ', text)
    text = re.sub(r'(?m)^2\. ', r'2️⃣ ', text)
    text = re.sub(r'(?m)^3\. ', r'3️⃣ ', text)
    text = re.sub(r'(?m)^4\. ', r'4️⃣ ', text)
    text = re.sub(r'(?m)^5\. ', r'5️⃣ ', text)

    # 중요 키워드 강조
    text = re.sub(r'(매출액|영업이익|순이익|실적|성장률|시장 점유율)', r'<b>\1</b>', text)
    text = re.sub(r'(급등|급락|상승|하락|성장|감소|인수|합병|계약|협약)', r'<b>\1</b>', text)

    # 투자 관련 키워드에 색상 강조
    text = re.sub(r'(매수|매도|추천|중립|보유)',
                  lambda
                      m: f'<span style="color:{"green" if m.group(1) in ["매수", "추천"] else "red" if m.group(1) == "매도" else "orange"}; font-weight:bold;">{m.group(1)}</span>',
                  text)

    return text


def generate_company_summary(company_name, news_data, openai_api_key):
    try:
        # 기업 정보 수집
        ticker_krx = get_ticker(company_name, source="fdr")
        if not ticker_krx:
            return f"## {company_name}에 대한 정보를 찾을 수 없습니다."

        ticker_yahoo = ticker_krx + ".KS"

        # 주가 정보 수집 (향상된 방식)
        stock_info = get_enhanced_stock_info(ticker_yahoo, ticker_krx)

        # 뉴스 요약 생성
        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-4', temperature=0)

        # 모든 뉴스 통합 후 전체 요약 요청
        all_news_text = "\n\n".join(
            [f"제목: {news['title']}\n내용: {news['content']}\n출처: {news['link']}" for news in news_data[:10]])

        prompt = f"""
        {company_name}에 관한 다음 뉴스들을 통합 분석하여 투자자에게 유용한 정보를 제공해주세요:

        {all_news_text}

        다음 형식으로 응답해주세요:
        1. 최신 동향: 통합적으로 정리된 최근 기업 핵심 동향 정보 3-5가지 (각 동향은 번호로 구분하고, 관련된 뉴스 출처 링크를 괄호 안에 포함)
        2. 투자에 영향을 미칠 수 있는 긍정적 요인과 부정적 요인
        3. 전반적인 투자 전망 및 조언
        """

        news_analysis = llm.predict(prompt)

        # HTML 강화된 요약 생성
        summary = f"""
        # 📊 {company_name} ({ticker_krx}) 투자 분석

        <div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <h2>🏢 기업 정보 요약</h2>
        <table style="width: 100%;">
          <tr>
            <td><b>현재 주가:</b></td>
            <td>{stock_info['current_price']} {stock_info['price_change_str']}</td>
          </tr>
          <tr>
            <td><b>52주 최고/최저:</b></td>
            <td>{stock_info['year_high']} / {stock_info['year_low']}</td>
          </tr>
          <tr>
            <td><b>시가총액:</b></td>
            <td>{stock_info['market_cap_str']}</td>
          </tr>
          <tr>
            <td><b>PER (주가수익비율):</b></td>
            <td>{stock_info['per']}</td>
          </tr>
          <tr>
            <td><b>PBR (주가순자산비율):</b></td>
            <td>{stock_info['pbr']}</td>
          </tr>
          <tr>
            <td><b>배당수익률:</b></td>
            <td>{stock_info['dividend_yield']}</td>
          </tr>
        </table>
        </div>

        <div style="background-color: #f5f5f5; padding: 15px; border-radius: 10px;">
        <h2>📰 최신 뉴스 및 분석</h2>
        {news_analysis}
        </div>
        """

        return summary
    except Exception as e:
        return f"## {company_name} 정보 분석 중 오류가 발생했습니다: {str(e)}"


# 향상된 주식 정보 수집 함수 (여러 소스에서 정보 통합)
def get_enhanced_stock_info(ticker_yahoo, ticker_krx):
    stock_info = {}

    # 두 방식으로 정보 수집 시도
    try:
        # 1. yfinance 사용
        yf_info = yf.Ticker(ticker_yahoo).info

        # 2. FinanceDataReader 사용 (한국 주식 정보)
        fdr_info = get_fdr_stock_info(ticker_krx)

        # 통합하여 저장 (yfinance와 FinanceDataReader 결과 병합)
        current_price = yf_info.get('currentPrice') or fdr_info.get('current_price')
        if current_price and current_price != '정보 없음':
            current_price = f"{int(current_price):,}원"
        else:
            current_price = '정보 없음'

        previous_close = yf_info.get('previousClose') or fdr_info.get('previous_close')

        # 가격 변동 계산
        if current_price != '정보 없음' and previous_close and previous_close != '정보 없음':
            try:
                if isinstance(current_price, str):
                    current_price_val = int(current_price.replace(',', '').replace('원', ''))
                else:
                    current_price_val = current_price

                price_change = ((current_price_val - previous_close) / previous_close) * 100
                color = "green" if price_change >= 0 else "red"
                price_change_str = f"<span style='color:{color};'>({price_change:+.2f}%)</span>"
            except:
                price_change_str = ""
        else:
            price_change_str = ""

        # 52주 최고/최저 설정
        year_high = yf_info.get('fiftyTwoWeekHigh') or fdr_info.get('year_high')
        if year_high and year_high != '정보 없음':
            year_high = f"{int(year_high):,}원"
        else:
            year_high = '정보 없음'

        year_low = yf_info.get('fiftyTwoWeekLow') or fdr_info.get('year_low')
        if year_low and year_low != '정보 없음':
            year_low = f"{int(year_low):,}원"
        else:
            year_low = '정보 없음'

        # 시가총액 계산
        market_cap = yf_info.get('marketCap') or fdr_info.get('market_cap')
        if market_cap and market_cap != '정보 없음':
            market_cap = market_cap / 1000000000000  # 조 단위로 변환
            market_cap_str = f"{market_cap:.2f}조 원"
        else:
            market_cap_str = "정보 없음"

        # PER 및 PBR 설정
        per = yf_info.get('trailingPE') or fdr_info.get('per')
        if per and per != '정보 없음':
            per = f"{per:.2f}"
        else:
            per = '정보 없음'

        pbr = yf_info.get('priceToBook') or fdr_info.get('pbr')
        if pbr and pbr != '정보 없음':
            pbr = f"{pbr:.2f}"
        else:
            pbr = '정보 없음'

        # 배당수익률 추가
        dividend_yield = yf_info.get('dividendYield') or fdr_info.get('dividend_yield')
        if dividend_yield and dividend_yield != '정보 없음':
            if dividend_yield < 1:  # 소수점으로 표시된 경우
                dividend_yield = f"{dividend_yield * 100:.2f}%"
            else:
                dividend_yield = f"{dividend_yield:.2f}%"
        else:
            dividend_yield = '정보 없음'

    except Exception as e:
        # 오류 발생 시 기본값으로 설정
        current_price = '정보 없음'
        price_change_str = ""
        year_high = '정보 없음'
        year_low = '정보 없음'
        market_cap_str = '정보 없음'
        per = '정보 없음'
        pbr = '정보 없음'
        dividend_yield = '정보 없음'

    # 결과 딕셔너리에 저장
    stock_info['current_price'] = current_price
    stock_info['price_change_str'] = price_change_str
    stock_info['year_high'] = year_high
    stock_info['year_low'] = year_low
    stock_info['market_cap_str'] = market_cap_str
    stock_info['per'] = per
    stock_info['pbr'] = pbr
    stock_info['dividend_yield'] = dividend_yield

    return stock_info


# FinanceDataReader를 통한 추가 주식 정보 수집 함수
def get_fdr_stock_info(ticker_krx):
    try:
        # 오늘 날짜 기준 데이터 가져오기
        today = datetime.now().strftime('%Y-%m-%d')
        last_year = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        # 지난 1년간의 주가 데이터 수집
        df = fdr.DataReader(ticker_krx, last_year, today)

        if df.empty:
            return {
                'current_price': '정보 없음',
                'previous_close': '정보 없음',
                'year_high': '정보 없음',
                'year_low': '정보 없음',
                'per': '정보 없음',
                'pbr': '정보 없음',
                'dividend_yield': '정보 없음',
                'market_cap': '정보 없음'
            }

        # 52주 최고/최저가 계산
        year_high = df['High'].max()
        year_low = df['Low'].min()

        # 현재가 (마지막 종가)
        current_price = df['Close'].iloc[-1]
        previous_close = df['Close'].iloc[-2] if len(df) > 1 else current_price

        # KRX 통합정보 가져오기 시도
        try:
            krx_df = fdr.StockListing('KRX')
            stock_row = krx_df[krx_df['Code'] == ticker_krx]

            if not stock_row.empty:
                per = stock_row['PER'].iloc[0]
                pbr = stock_row['PBR'].iloc[0]
                market_cap = stock_row['Market Cap'].iloc[0]
            else:
                per = '정보 없음'
                pbr = '정보 없음'
                market_cap = '정보 없음'
        except:
            per = '정보 없음'
            pbr = '정보 없음'
            market_cap = '정보 없음'

        # 배당수익률은 일반적으로 KRX 정보에서 제공하지 않음
        dividend_yield = '정보 없음'

        return {
            'current_price': current_price,
            'previous_close': previous_close,
            'year_high': year_high,
            'year_low': year_low,
            'per': per,
            'pbr': pbr,
            'dividend_yield': dividend_yield,
            'market_cap': market_cap
        }
    except Exception as e:
        return {
            'current_price': '정보 없음',
            'previous_close': '정보 없음',
            'year_high': '정보 없음',
            'year_low': '정보 없음',
            'per': '정보 없음',
            'pbr': '정보 없음',
            'dividend_yield': '정보 없음',
            'market_cap': '정보 없음'
        }


# 네이버 뉴스 크롤링 함수
def crawl_news(company):
    today = datetime.today()
    start_date = (today - timedelta(days=5)).strftime('%Y%m%d')
    end_date = today.strftime('%Y%m%d')
    encoded_query = urllib.parse.quote(company)
    url = f"https://search.naver.com/search.naver?where=news&query={encoded_query}&nso=so:r,p:from{start_date}to{end_date}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    articles = soup.select("ul.list_news > li")

    data = []
    for article in articles[:10]:
        title = article.select_one("a.news_tit").text
        link = article.select_one("a.news_tit")['href']
        content = article.select_one("div.news_dsc").text if article.select_one("div.news_dsc") else ""
        data.append({"title": title, "link": link, "content": content})

    return data


# 토크나이저 함수
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


# 청크 사이즈에 맞게 텍스트 자르는 함수
def get_text_chunks(news_data):
    texts = [f"{item['title']}\n{item['content']}" for item in news_data]
    metadatas = [{"source": item["link"]} for item in news_data]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    return text_splitter.create_documents(texts, metadatas=metadatas)


# 벡터사이즈 불러와서 텍스트 저장하는 함수
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return FAISS.from_documents(text_chunks, embeddings)


# 불러와서 chain 만드는 함수
def create_chat_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-4', temperature=0)

    # 시스템 메시지 설정하여 투자 조언 특화 챗봇으로 만들기
    system_message = """
    당신은 투자 분석 전문가입니다. 주어진 기업 정보와 뉴스를 분석하여 투자자에게 유용한 조언을 제공합니다.

    다음 사항을 준수하여 응답해주세요:
    - 항상 뉴스 데이터에 기반한 객관적인 정보를 제공하세요.
    - 투자의 위험성을 항상 고려하고 균형 잡힌 시각을 유지하세요.
    - 확실하지 않은 내용은 추측이라고 명시하세요.
    - 법적 책임이 있는 확정적인 투자 권유는 하지 마세요.
    - 응답에 다음과 같은 이모지를 적절히 사용하여 가독성을 높이세요: 📊 💹 📰 ⚠️ ✅ 💰
    - 중요한 수치나 정보는 굵게 표시하여 강조하세요.
    - 주요 포인트는 번호를 매겨 구분하고 (1️⃣, 2️⃣, 3️⃣), 머리글에는 ## 표시를 사용하세요.
    """

    return ConversationalRetrievalChain.from_llm(
        llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h, return_source_documents=True,
        verbose=True)


# 최근 거래일 찾기 함수
def get_recent_trading_day():
    today = datetime.now()
    if today.hour < 9:  # 9시 이전이면 전날을 기준으로
        today -= timedelta(days=1)

    while today.weekday() in [5, 6]:  # 토요일(5), 일요일(6)이면 하루씩 감소
        today -= timedelta(days=1)

    return today.strftime('%Y-%m-%d')


#  티커 조회 함수 (야후 & FinanceDataReader)
def get_ticker(company, source="yahoo"):
    try:
        listing = fdr.StockListing('KRX')
        ticker_row = listing[listing["Name"].str.strip() == company.strip()]
        if not ticker_row.empty:
            krx_ticker = str(ticker_row.iloc[0]["Code"]).zfill(6)
            if source == "yahoo":
                return krx_ticker + ".KS"  # ✅ 야후 파이낸스용 티커 변환
            return krx_ticker  # ✅ FinanceDataReader용 티커
        return None

    except Exception as e:
        st.error(f"티커 조회 중 오류 발생: {e}")
        return None


# 야후 파이낸스에서 분봉 데이터 가져오기 (1day, week)
def get_intraday_data_yahoo(ticker, period="1d", interval="1m"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        df = df.rename(columns={"Datetime": "Date", "Close": "Close"})

        # ✅ 주말 데이터 제거 (혹시 남아있는 경우 대비)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[df["Date"].dt.weekday < 5].reset_index(drop=True)

        return df
    except Exception as e:
        st.error(f"야후 파이낸스 데이터 불러오기 오류: {e}")
        return pd.DataFrame()

# FinanceDataReader를 통한 일별 시세 (1month, 1year)
def get_daily_stock_data_fdr(ticker, period):
    try:
        end_date = get_recent_trading_day()
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(
            days=30 if period == "1month" else 365)).strftime('%Y-%m-%d')
        df = fdr.DataReader(ticker, start_date, end_date)

        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        df = df.rename(columns={"Date": "Date", "Close": "Close"})

        # ✅ 주말 데이터 완전 제거
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[df["Date"].dt.weekday < 5].reset_index(drop=True)

        return df
    except Exception as e:
        st.error(f"FinanceDataReader 데이터 불러오기 오류: {e}")
        return pd.DataFrame()

# Plotly를 이용한 주가 시각화 함수 (x축 포맷 최적화)
def plot_stock_plotly(df, company, period):
    if df is None or df.empty:
        st.warning(f"📉 {company} - 해당 기간({period})의 거래 데이터가 없습니다.")
        return

    fig = go.Figure()

    # ✅ x축 날짜 형식 설정
    if period == "1day":
        df["FormattedDate"] = df["Date"].dt.strftime("%H:%M")  # ✅ 1day → HH:MM 형식
    elif period == "week":
        df["FormattedDate"] = df["Date"].dt.strftime("%m-%d %H:%M")  # ✅ week → MM-DD HH:MM 형식
    else:
        df["FormattedDate"] = df["Date"].dt.strftime("%m-%d")  # ✅ 1month, 1year → MM-DD 형식

    if period in ["1day", "week"]:
        fig.add_trace(go.Scatter(
            x=df["FormattedDate"],
            y=df["Close"],
            mode="lines+markers",
            line=dict(color="royalblue", width=2),
            marker=dict(size=5),
            name="체결가"
        ))
    else:
        fig.add_trace(go.Candlestick(
            x=df["FormattedDate"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="캔들 차트"
        ))

    fig.update_layout(
        title=f"{company} 주가 ({period})",
        xaxis_title="시간" if period == "1day" else "날짜",
        yaxis_title="주가 (KRW)",
        template="plotly_white",
        xaxis=dict(showgrid=True, type="category", tickangle=-45),
        hovermode="x unified"
    )

    st.plotly_chart(fig)


if __name__ == '__main__':
    main()