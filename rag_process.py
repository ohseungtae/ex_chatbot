import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


def tiktoken_len(text):
    """
    텍스트의 토큰 길이를 계산하는 함수

    Args:
        text (str): 토큰화할 텍스트

    Returns:
        int: 토큰 길이
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


def get_text_chunks(news_data):
    """
    뉴스 데이터를 청크로 나누는 함수

    Args:
        news_data (list): 뉴스 데이터 목록

    Returns:
        list: 처리된 텍스트 청크
    """
    texts = [f"{item['title']}\n{item['content']}" for item in news_data]
    metadatas = [{"source": item["link"]} for item in news_data]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    return text_splitter.create_documents(texts, metadatas=metadatas)


def get_vectorstore(text_chunks):
    """
    텍스트 청크에서 벡터 저장소를 생성하는 함수

    Args:
        text_chunks (list): 텍스트 청크 목록

    Returns:
        FAISS: 생성된 벡터 저장소
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return FAISS.from_documents(text_chunks, embeddings)


def create_chat_chain(vectorstore, openai_api_key):
    """
    대화 체인을 생성하는 함수

    Args:
        vectorstore (FAISS): 벡터 저장소
        openai_api_key (str): OpenAI API 키

    Returns:
        ConversationalRetrievalChain: 생성된 대화 체인
    """
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-4', temperature=0)
    return ConversationalRetrievalChain.from_llm(
        llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h, return_source_documents=True)