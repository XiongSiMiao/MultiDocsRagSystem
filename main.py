from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import logging
import os
import re
import json
import pdfplumber
import numpy as np
import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from APIServices import ToolService
from IntentRecognizer import IntentRecognizer
from ToolAgent import ToolAgent
import mysql.connector

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Comprehensive Q&A and Tool Service",
    description="API service for intent recognition, database queries, API tool calls, and knowledge-based Q&A",
    version="1.0.0",
    docs_url=None,
    redoc_url=None
)

# 定义请求模型
class QueryRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    segments: str = Field(..., description="比赛阶段，如'初赛'")
    paper: str = Field(..., description="试卷编号，如'B'")
    id: int = Field(..., description="题目ID")
    category: str = Field(None, description="题目类别，如'选择题'或'问答题'")
    content: str = Field(None, description="题目内容/选项")

# 定义响应模型
class QueryResponse(BaseModel):
    segments: str
    paper: str
    id: int
    answer: str

# 全局变量 - 启动时一次性加载
MODEL_NAME = './bge-small-zh-v1.5'
CROSS_ENCODER_NAME = './mmarco-mMiniLMv2-L12-H384-v1'
VECTOR_DB_DIR = './vector_db'
KNOWLEDGE_DIR = "知识文档"
DEEPSEEK_API_KEY = "sk-afabddf5ba0c4c96abb3567aa3324605"
EMBEDDINGS = None
VECTORSTORE = None
CROSS_ENCODER = None
TOOL_AGENT = None

def robust_pdf_loader(file_path: str) -> list[Document]:
    """使用 pdfplumber 加载 PDF，处理复杂布局，返回 Document 对象"""
    documents = []
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text(layout=True)
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            documents.append(Document(page_content=text, metadata={"source": file_path}))
        logging.info(f"pdfplumber 成功加载: {file_path}")
    except Exception as e:
        logging.warning(f"pdfplumber 加载失败: {file_path}, 错误: {e}, 尝试 PyPDFLoader")
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            logging.info(f"PyPDFLoader 成功加载: {file_path}")
        except Exception as e2:
            logging.error(f"PyPDFLoader 加载失败: {file_path}, 错误: {e2}")
    return documents

def robust_docx_loader(file_path: str) -> list[Document]:
    """尝试多种 DOCX 加载器，返回 Document 对象"""
    loaders = [UnstructuredWordDocumentLoader]
    for Loader in loaders:
        try:
            loader = Loader(file_path)
            documents = loader.load()
            if documents:
                logging.info(f"{Loader.__name__} 成功加载: {file_path}")
                return documents
        except Exception as e:
            logging.warning(f"{Loader.__name__} 加载失败: {file_path}, 错误: {e}")
    logging.error(f"所有 DOCX 加载器均失败: {file_path}")
    return []

def build_vector_db(knowledge_dir: str, output_dir: str = "./vector_db", model_name: str = "./bge-small-zh-v1.5"):
    """从文档目录生成 FAISS 向量库，保存为离线文件"""
    documents = []
    for file in os.listdir(knowledge_dir):
        file_path = os.path.join(knowledge_dir, file)
        ext = os.path.splitext(file)[1].lower()
        try:
            if ext == ".pdf":
                docs = robust_pdf_loader(file_path)
            elif ext in [".doc", ".docx"]:
                docs = robust_docx_loader(file_path)
            elif ext == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
            else:
                logging.info(f"跳过不支持格式: {file_path}")
                continue
            documents.extend(docs)
            logging.info(f"成功加载: {file_path}")
        except Exception as e:
            logging.error(f"加载 {file_path} 失败: {e}")

    if not documents:
        raise ValueError("没有成功加载任何文档！")
    logging.info(f"共加载 {len(documents)} 个文档")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["。", "！", "？", "\n\n"],
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    logging.info(f"生成 {len(chunks)} 个 chunk")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"}
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(output_dir, exist_ok=True)
    vectorstore.save_local(output_dir)
    logging.info(f"向量库已保存到 {output_dir}/index.faiss 和 index.pkl")
    return vectorstore, embeddings

def load_and_query(query: str, top_k: int = 5):
    """优化查询：使用全局模型，无进度条"""
    global VECTORSTORE, CROSS_ENCODER
    retriever = VECTORSTORE.as_retriever(search_kwargs={"k": 20})
    docs = retriever.invoke(query)
    pairs = [[query, doc.page_content] for doc in docs]
    scores = CROSS_ENCODER.predict(pairs, show_progress_bar=False, batch_size=16)
    top_indices = np.argsort(scores)[::-1][:top_k]
    top_docs = [docs[i] for i in top_indices]
    return top_docs

def call_deepseek_api(query: str, chunks: list, category: str = None, content: str = None) -> str:
    """调用 DeepSeek API"""
    context = "\n\n".join([f"Chunk {i+1}: {chunk.page_content}" for i, chunk in enumerate(chunks)])
    if category == "选择题":
        prompt = f"""
        你是一个专业的金融领域助手。基于以下上下文，回答选择题。请直接返回正确选项的字母（如 A, B, C, D）。
        问题: {query}
        选项: {content}
        上下文: {context}
        答案："""
    elif category == "问答题":
        prompt = f"""
        你是一个专业的金融领域助手。基于以下上下文，回答用户的问题。确保只给出对应的答案，让答案最准确、简洁。如果没有找到内容，直接返回：根据已有知识无法回答
        问题: {query}
        上下文: {context}
        答案："""
    else:
        prompt = f"""
        你是一个专业的金融领域助手。基于以下上下文，回答用户的问题。
        问题: {query}
        上下文: {context}
        答案："""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant in the financial domain."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300,
        "temperature": 0.1
    }
    try:
        response = requests.post("https://api.deepseek.com/v1/chat/completions", 
                               json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        answer = response.json()['choices'][0]['message']['content'].strip()
        logger.info(f"DeepSeek 回答: {answer}")
        return answer
    except Exception as e:
        logger.error(f"DeepSeek API失败: {str(e)}")
        return "无法生成答案"

def process_query(question: str, category: str = None, content: str = None) -> str:
    """处理用户问题，结合意图识别、数据库查询、API工具调用和知识问答"""
    global TOOL_AGENT
    intent_result = TOOL_AGENT.intent_recognizer.recognize_intent(question)

    if intent_result:
        if '数据库' in intent_result:
            sql_query = intent_result['数据库']
            result = TOOL_AGENT.execute_sql_query(sql_query)
            return result
        elif '工具依赖调用' in intent_result:
            result = TOOL_AGENT.execute_tool_dependency_call(intent_result['工具依赖调用'])
            return result
        else:
            intent_result, new_question = TOOL_AGENT.execute_and_generate_new_question(question)
            return new_question

    top_docs = load_and_query(question)
    answer = call_deepseek_api(question, top_docs, category, content)
    logger.info(f"知识问答回答: {answer}")
    if category == "选择题":
        match = re.match(r'^[A-D]$', answer.strip())
        if match:
            return match.group(0)
        for char in answer.strip():
            if char in 'ABCD':
                return char
    return answer

@app.post("/api/query", response_model=QueryResponse, response_model_exclude_none=True)
async def handle_query_request(request: QueryRequest):
    try:
        response_data = {
            "segments": request.segments,
            "paper": request.paper,
            "id": request.id,
            "answer": process_query(request.question, request.category, request.content)
        }
        logger.info(f"请求处理成功: ID {request.question}")
        return JSONResponse(
            content=response_data,
            status_code=200,
            headers={"Content-Type": "application/json; charset=utf-8"}
        )
    except Exception as e:
        logger.error(f"请求处理错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

def run_server():
    """启动服务器 - 一次性加载所有模型"""
    global EMBEDDINGS, VECTORSTORE, CROSS_ENCODER, TOOL_AGENT
    logger.info("初始化模型和向量库...")

    EMBEDDINGS = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"}
    )
    if not os.path.exists(os.path.join(VECTOR_DB_DIR, "index.faiss")):
        logger.info("生成新的向量库...")
        VECTORSTORE, EMBEDDINGS = build_vector_db(KNOWLEDGE_DIR)
    else:
        logger.info("加载现有向量库...")
        VECTORSTORE = FAISS.load_local(
            VECTOR_DB_DIR,
            EMBEDDINGS,
            allow_dangerous_deserialization=True
        )

    logger.info("加载 Rerank 模型...")
    CROSS_ENCODER = CrossEncoder(CROSS_ENCODER_NAME)

    logger.info("初始化 ToolAgent...")
    TOOL_AGENT = ToolAgent(
        base_url="http://api.example.com:30000",
        app_id="your_app_id",
        app_key="your_app_key"
    )

    logger.info("所有模型加载完成！")
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=10000,
        log_level="info",
        access_log=False,
        timeout_keep_alive=30,
        limit_concurrency=1000,
        limit_max_requests=10000,
    )
    server = uvicorn.Server(config)
    logger.info("Starting server on http://0.0.0.0:10000")
    server.run()

if __name__ == "__main__":
    run_server()