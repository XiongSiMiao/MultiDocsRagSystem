from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn
import logging
from typing import Optional, List, Any
import json
import os
import re
import pdfplumber
import numpy as np
import requests
from langchain_huggingface import HuggingFaceEmbeddings  # 新导入，解决弃用警告
from langchain.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Exam API Service",
    description="API service for exam questions and answers (Optimized LangChain RAG)",
    version="1.0.0",
    docs_url=None,
    redoc_url=None
)

# 定义请求模型
class ExamRequest(BaseModel):
    segments: str = Field(..., description="比赛阶段，如'初赛'")
    paper: str = Field(..., description="试卷编号，如'B'")
    id: int = Field(..., description="题目ID")
    question: Optional[str] = Field(None, description="问题文本")
    category: Optional[str] = Field(None, description="题目类别，如'选择题'或'问答题'")
    content: Optional[str] = Field(None, description="题目内容/选项")

    @validator('*', pre=True)
    def remove_whitespace(cls, v):
        if isinstance(v, str):
            return v.strip()
        return v

# 定义响应模型
class ExamResponse(BaseModel):
    segments: str
    paper: str
    id: int
    answer: Any

# 性能监控中间件
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    import time
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"Processed request in {process_time:.4f}s")
    return response

# 全局变量 - 启动时一次性加载
MODEL_NAME = './bge-small-zh-v1.5'
CROSS_ENCODER_NAME = './mmarco-mMiniLMv2-L12-H384-v1'
VECTOR_DB_DIR = './vector_db'
KNOWLEDGE_DIR = "知识文档"
DEEPSEEK_API_KEY = "sk-afabddf5ba0c4c96abb3567aa3324605"
EMBEDDINGS = None
VECTORSTORE = None
CROSS_ENCODER = None

def robust_pdf_loader(file_path: str) -> List[Document]:
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

def robust_docx_loader(file_path: str) -> List[Document]:
    """尝试多种 DOCX 加载器，返回 Document 对象"""
    loaders = [Docx2txtLoader, UnstructuredWordDocumentLoader]
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
    """
    从文档目录生成 FAISS 向量库，保存为离线文件。
    :param knowledge_dir: 文档文件夹（PDF/DOC/TXT）
    :param output_dir: 保存路径（FAISS 索引和元数据）
    :param model_name: 嵌入模型（中文优化）
    """
    # 加载文档
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

    # 分块（语义分割）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["。", "！", "？", "\n\n"],
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    logging.info(f"生成 {len(chunks)} 个 chunk")

    # 生成嵌入
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"}  # 改 "cuda" 若有 GPU
        )
        logging.info(f"成功加载嵌入模型: {model_name}")
    except Exception as e:
        logging.warning(f"加载本地模型 {model_name} 失败: {e}, 尝试在线模型")
        model_name = "./bge-small-zh-v1.5"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"}
        )
        logging.info(f"成功加载在线模型: {model_name}")

    # 建 FAISS 索引（cosine 相似度）
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    vectorstore.save_local(output_dir)
    logging.info(f"向量库已保存到 {output_dir}/index.faiss 和 index.pkl")
    return vectorstore, embeddings

def load_and_query(query: str, top_k: int = 5):
    """优化查询：使用全局模型，无进度条"""
    global VECTORSTORE, CROSS_ENCODER
    
    # 粗召回
    retriever = VECTORSTORE.as_retriever(search_kwargs={"k": 20})
    docs = retriever.invoke(query)
    
    # Rerank - 禁用进度条
    pairs = [[query, doc.page_content] for doc in docs]
    scores = CROSS_ENCODER.predict(pairs, show_progress_bar=False, batch_size=16)
    
    # 按分数排序，取 top-K
    top_indices = np.argsort(scores)[::-1][:top_k]
    top_docs = [docs[i] for i in top_indices]
    return top_docs

def call_deepseek_api(query: str, chunks: List[Any], category: Optional[str] = None, content: Optional[str] = None) -> str:
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
        "max_tokens": 300,  # 减少token数
        "temperature": 0.1  # 降低随机性
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

def get_answer(request_data: dict) -> Any:
    """生成答案"""
    query = request_data.get('question')
    category = request_data.get('category')
    content = request_data.get('content')
    
    if not query:
        return ""
    
    logger.info(f"处理查询: {query[:50]}...")
    top_docs = load_and_query(query)
    answer = call_deepseek_api(query, top_docs, category, content)
    context = "\n\n".join([f"Chunk {i+1}: {doc.page_content}" for i, doc in enumerate(top_docs)])
    logger.info(f"召回的文档：{context[:200]}...")
    
    # 处理选择题答案
    if category == "选择题":
        match = re.match(r'^[A-D]$', answer.strip())
        if match:
            return match.group(0)
        # 如果不是单个字母，尝试提取
        for char in answer.strip():
            if char in 'ABCD':
                return char
        return answer
    
    return answer

@app.post("/api/exam", response_model=ExamResponse, response_model_exclude_none=True)
async def handle_exam_request(request: Request):
    try:
        raw_data = await request.json()
        required_fields = ['segments', 'paper', 'id']
        for field in required_fields:
            if field not in raw_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        response_data = {
            "segments": raw_data['segments'],
            "paper": raw_data['paper'],
            "id": raw_data['id'],
            "answer": get_answer(raw_data)
        }
        
        return JSONResponse(
            content=response_data,
            status_code=200,
            headers={"Content-Type": "application/json; charset=utf-8"}
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        logger.error(f"请求处理错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

def run_server():
    """启动服务器 - 一次性加载所有模型"""
    global EMBEDDINGS, VECTORSTORE, CROSS_ENCODER
    
    logger.info("初始化模型和向量库...")
    
    # 1. 加载嵌入模型
    EMBEDDINGS = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"}
    )
    
    # 2. 加载或创建向量库
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
    
    # 3. 加载 CrossEncoder（一次性）
    logger.info("加载 Rerank 模型...")
    CROSS_ENCODER = CrossEncoder(CROSS_ENCODER_NAME)
    
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