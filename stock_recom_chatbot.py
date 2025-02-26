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
            st.markdown(st.session_state.company_summary)

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

                    # 응답 표시
                    st.markdown(response)

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


def generate_company_summary(company_name, news_data, openai_api_key):
    try:
        # 기업 정보 수집
        ticker_krx = get_ticker(company_name, source="fdr")
        if not ticker_krx:
            return f"## {company_name}에 대한 정보를 찾을 수 없습니다."

        ticker_yahoo = ticker_krx + ".KS"

        # 주가 정보 수집
        try:
            stock_info = yf.Ticker(ticker_yahoo).info
            current_price = stock_info.get('currentPrice', '정보 없음')
            previous_close = stock_info.get('previousClose', current_price)

            if current_price != '정보 없음' and previous_close != '정보 없음':
                price_change = ((current_price - previous_close) / previous_close) * 100
                price_change_str = f"({price_change:.2f}%)"
            else:
                price_change_str = ""

            year_high = stock_info.get('fiftyTwoWeekHigh', '정보 없음')
            year_low = stock_info.get('fiftyTwoWeekLow', '정보 없음')
            market_cap = stock_info.get('marketCap', '정보 없음')
            if market_cap != '정보 없음':
                market_cap = market_cap / 1000000000000  # 조 단위로 변환
                market_cap_str = f"{market_cap:.2f}조 원"
            else:
                market_cap_str = "정보 없음"

            per = stock_info.get('trailingPE', '정보 없음')
            pbr = stock_info.get('priceToBook', '정보 없음')
        except:
            current_price = '정보 없음'
            price_change_str = ""
            year_high = '정보 없음'
            year_low = '정보 없음'
            market_cap_str = '정보 없음'
            per = '정보 없음'
            pbr = '정보 없음'

        # 뉴스 요약 생성
        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-4', temperature=0)

        news_text = "\n\n".join([f"제목: {news['title']}\n내용: {news['content']}" for news in news_data[:5]])

        prompt = f"""
        {company_name}에 관한 다음 뉴스들을 분석하여 투자자에게 유용한 정보를 제공해주세요:

        {news_text}

        다음 형식으로 응답해주세요:
        1. 최신 뉴스 주요 내용 3가지 요약 (각 뉴스별로 제목, 출처, 요약, 관련 키워드 포함)
        2. 투자에 영향을 미칠 수 있는 긍정적 요인과 부정적 요인
        3. 전반적인 투자 전망 및 조언
        """

        news_analysis = llm.predict(prompt)

        # 최종 요약 생성
        summary = f"""
        # 📊 {company_name} ({ticker_krx}) 투자 분석

        ## 기업 정보 요약
        * **현재 주가:** {current_price:,}원 {price_change_str}
        * **52주 최고/최저:** {year_high:,}원 / {year_low:,}원
        * **시가총액:** {market_cap_str}
        * **PER (주가수익비율):** {per}
        * **PBR (주가순자산비율):** {pbr}

        ## 📰 최신 뉴스 및 분석

        {news_analysis}
        """

        return summary
    except Exception as e:
        return f"## {company_name} 정보 분석 중 오류가 발생했습니다: {str(e)}"


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
    - 항상 뉴스 데이터에 기반한 객관적인 정보를 제공하세요.
    - 투자의 위험성을 항상 고려하고 균형 잡힌 시각을 유지하세요.
    - 확실하지 않은 내용은 추측이라고 명시하세요.
    - 법적 책임이 있는 확정적인 투자 권유는 하지 마세요.
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