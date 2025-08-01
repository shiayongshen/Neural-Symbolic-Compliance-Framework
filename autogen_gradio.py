import os
import gradio as gr
from typing import List, Dict
from gradio_look import extract_softs_from_code,extract_softs_description_and_varnames
from autogen import *
from search_deep_laws import LegalSearchEngine
from chromadb import Client, Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from FlagEmbedding import FlagReranker
from autogen.coding import DockerCommandLineCodeExecutor, LocalCommandLineCodeExecutor
from fpdf import FPDF
import os
from datetime import datetime

AUTOGEN_USE_DOCKER = False  # Set to True if you want to use Docker for code execution
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")
# 1. Configuration: LLM + Reranker
llm_config = { "config_list": [{ "model": "gpt-4o-mini", "api_key": OPENAI_API_KEY }] }
llm_config_o3 = { "config_list": [{ "model": "gpt-4o", "api_key": OPENAI_API_KEY }] }
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True, devices=["cuda:0"])

try:
    legal_search_engine = LegalSearchEngine()
    legal_search_available = True
except Exception as e:
    print(f"法條搜索引擎初始化失敗: {e}")
    legal_search_available = False

def save_conversation_to_pdf(user_query: str, conversation_output: str) -> str:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # 使用可顯示中文的字型（需先下載 .ttf 檔並放在同目錄）
    font_path = "NotoSansTC-VariableFont_wght.ttf"  # 或 ArialUnicodeMS.ttf
    pdf.add_font('Noto', '', font_path, uni=True)
    pdf.set_font("Noto", size=12)

    pdf.multi_cell(0, 10, f"使用者提問：{user_query}\n\n")
    pdf.multi_cell(0, 10, conversation_output)

    os.makedirs("pdf_outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"legal_chat_{timestamp}.pdf"
    file_path = os.path.join("pdf_outputs", filename)
    pdf.output(file_path)

    return file_path

def legal_article_search(query, top_k, rerank_top_n, hybrid_alpha):
    """法條搜索功能"""
    if not legal_search_available:
        return "法條搜索引擎未正確初始化，請檢查資料庫配置。"
    
    try:
        # 更新混合搜索權重
        legal_search_engine.hybrid_alpha = hybrid_alpha
        
        # 執行搜索
        results = legal_search_engine.search(
            query=query, 
            top_k=top_k, 
            rerank_top_n=rerank_top_n
        )
        
        if not results:
            return "未找到相關法條。"
        
        # 格式化結果
        formatted_output = f"# 法條查詢結果\n\n"
        formatted_output += f"**查詢：** {query}\n\n"
        formatted_output += f"**搜索參數：** Top-K={top_k}, 重排序數量={rerank_top_n}, 混合權重={hybrid_alpha}\n\n"
        
        for i, result in enumerate(results):
            metadata = result.get("metadata", {})
            law_name = metadata.get("法律名稱", "未知法律")
            article = metadata.get("條", "未知條款")
            score = result.get("score", 0)
            sources = result.get("sources", [])
            
            formatted_output += f"## 結果 {i+1}\n"
            formatted_output += f"**相關度：** {score:.3f} \n\n"
            formatted_output += f"**法律：** {law_name} \n\n"
            formatted_output += f"**條文：** {article} \n\n"
            if sources:
                formatted_output += f"**搜索來源：** {', '.join(sources)} \n\n"
            formatted_output += f"**內容：**\n```\n{result['content']}\n```\n\n"
            formatted_output += "---\n\n"
        
        return formatted_output
        
    except Exception as e:
        return f"搜索時發生錯誤：{str(e)}"


def get_related_laws_analysis(query, top_k, rerank_top_n):
    """獲取相關法條分析"""
    if not legal_search_available:
        return "法條搜索引擎未正確初始化。"
    
    try:
        direct_relevant, indirectly_relevant = legal_search_engine.get_related_laws(
            query=query,
            top_k=top_k,
            rerank_top_n=rerank_top_n
        )
        
        output = f"# 法條相關性分析\n\n"
        output += f"**查詢：** {query}\n\n"
        
        if direct_relevant:
            output += f"## 🎯 直接相關法條 ({len(direct_relevant)} 條)\n\n"
            for i, result in enumerate(direct_relevant):
                metadata = result.get("metadata", {})
                law_name = metadata.get("法律名稱", "未知法律")
                article = metadata.get("條", "未知條款")
                score = result.get("score", 0)
                
                output += f"### {i+1}. {law_name}  {article} \n"
                output += f"**相關度：** {score:.3f} \n"
                output += f"**內容：** {result['content'][:200]}...\n\n"
        
        if indirectly_relevant:
            output += f"## 🔗 間接相關法條 ({len(indirectly_relevant)} 條)\n\n"
            for i, result in enumerate(indirectly_relevant):
                metadata = result.get("metadata", {})
                law_name = metadata.get("法律名稱", "未知法律")
                article = metadata.get("條", "未知條款")
                score = result.get("score", 0)
                
                output += f"### {i+1}. {law_name} 第 {article} 條\n"
                output += f"**相關度：** {score:.3f}\n"
                output += f"**內容：** {result['content'][:150]}...\n\n"
        
        if not direct_relevant and not indirectly_relevant:
            output += "未找到相關法條。\n"
        
        return output
        
    except Exception as e:
        return f"分析時發生錯誤：{str(e)}"


def get_chroma_collection():
    client = Client(Settings(
        persist_directory="chroma_db",
        is_persistent=True
    ))
    embedding_function = OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name='text-embedding-ada-002'
    )
    return client.get_collection("legal_casesv1", embedding_function=embedding_function)

def search_and_rerank(query: str, top_k=5):
    global extracted_codes
    extracted_codes = [] 
    collection = get_chroma_collection()
    search_results = collection.query(
        query_texts=[query],
        n_results=top_k * 2
    )
    
    documents = search_results['documents'][0]
    metadatas = search_results['metadatas'][0]
    ids = search_results['ids'][0]
    
    if not documents:
        return {'ranked_documents': [], 'ranked_metadatas': [], 'ids': []}
    
    ranking_scores = []
    for doc in documents:
        score = reranker.compute_score([query, doc])
        ranking_scores.append(score)
    
    indexed_scores = list(enumerate(ranking_scores))
    sorted_indexed_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
    ranked_indices = [idx for idx, _ in sorted_indexed_scores[:1]]
    
    ranked_documents = [documents[i] for i in ranked_indices]
    ranked_metadatas = [metadatas[i] for i in ranked_indices]
    ranked_ids = [ids[i] for i in ranked_indices]
    for i, metadata in enumerate(ranked_metadatas):
        if metadata and 'z3code' in metadata:
            code = metadata['z3code']
            if code and code.strip():
                # 為每個程式碼片段創建檔案
                filename = f"case_{ranked_ids[i]}_code.py"
                filepath = os.path.join("code_execution", filename)
                
                # 確保目錄存在
                os.makedirs("code_execution", exist_ok=True)
                
                # 寫入程式碼檔案
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(code)
                
                extracted_codes.append({
                    'case_id': ranked_ids[i],
                    'filename': filename,
                    'filepath': filepath,
                    'code': code
                })
    
    return {
        'ranked_documents': ranked_documents, 
        'ranked_metadatas': ranked_metadatas, 
        'ids': ranked_ids,
        'extracted_codes': extracted_codes
    }

def get_extracted_codes():
    """獲取已提取的程式碼列表"""
    global extracted_codes
    return extracted_codes

# 修改所有的 is_termination_msg
def is_termination_msg(x):
    """檢查是否為終止訊息"""
    if "content" not in x or x["content"] is None:
        return False  # 空訊息不終止
    
    content = x["content"].strip().lower()
    
    # 檢查是否包含 terminate (不區分大小寫)
    if "terminate" in content:
        return True
    
    return False

def read_soft_config(filename: str) -> str:
    filepath = os.path.join("code_execution", filename)
    with open(filepath, "r", encoding="utf-8") as f:
        code_string = f.read()
    return extract_softs_from_code(code_string)


def get_softs_labels_and_vars(filename: str):
    filepath = os.path.join("code_execution", filename)
    with open(filepath, "r", encoding="utf-8") as f:
        code_string = f.read()
    return extract_softs_description_and_varnames(code_string)

initializer = UserProxyAgent(
    name="Init",
    human_input_mode="NEVER",
    code_execution_config=False,
)

legal_analyst = AssistantAgent(
    is_termination_msg=is_termination_msg,
    name="legal_assistant",
    llm_config=llm_config_o3,
    system_message="""
    你是一位資深的金融法規與科技應用綜合分析師，負責統整所有專門代理人的分析結果。

    **你將接收到三個專門分析師的報告：**
    1. **案例分析師**：案例背景與違規行為分析
    2. **程式碼分析師**：Z3求解器結果與改善建議分析  
    3. **法律分析師**：法條違規點與合規要求分析

    **你的任務是整合這些專業觀點，撰寫完整的綜合分析報告：**

    ## 問題背景
    - 整合案例分析師的背景描述

    ## 案例摘要  
    - 綜合案例的核心違規行為和影響

    ## 程式改善機制分析
    - 整合程式碼分析師的改善建議
    - 並說明透過什麼樣的改善可以讓其變成合規

    ## 法條合規分析
    - 整合法律分析師的法規分析
    - 明確指出合規改善方向

    ## 結論與建議
    - 提供整合性的決策建議
    - 強調技術改善與法規合規的結合

    請確保：
    - 避免重複各分析師已詳述的內容
    - 著重於跨領域的整合與洞察
    - 提供可執行的綜合建議

    完成後請回覆 "TERMINATE"。
    """,
)

search_agent = AssistantAgent(
    name="search_agent",
    llm_config=llm_config,
    system_message="""
    你是一個專門負責搜索法律案例資料庫的代理。
    當收到法律問題時，你需要：
    1. 使用 search_and_rerank 函數搜索相關案例
    2. 將搜索結果整理並傳遞給下一個代理
    3. 如果搜索到的案例包含程式碼，請標註出來
    
    搜索完成後，請說 "搜索完成，案例已找到" 並將結果傳遞給程式執行代理。
    """,
    is_termination_msg=is_termination_msg,
)

find_code_agent = AssistantAgent(
    name="find_code_agent",
    llm_config=llm_config,
    system_message="""
    當搜索代理找到包含程式碼的案例時，你需要參考下面的格式，注意，工作目錄請不需要更動，你需要更改的只有要執行哪一個python檔案，請你注意縮排並確保格式正確：
    ```python
import os
import subprocess
                                                                                                           
def main():
    result = subprocess.run(
        ['python', 'case_case_0_code.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8'  
    )
    return result.stdout

if __name__ == "__main__":
    output = main()
    print(output)
    ```
    這種格式的程式碼，run的是前面我們搜索到的python檔案，你不需要另外書寫z3-solver的程式碼，你也不需要另外解釋。
    """,
    is_termination_msg=is_termination_msg,
)
work_dir = os.path.abspath("code_execution")
executor = LocalCommandLineCodeExecutor(work_dir=work_dir)
code_executor = UserProxyAgent(
    name="code_executor",
    human_input_mode="NEVER",
    is_termination_msg=is_termination_msg,
    code_execution_config={
        "executor": executor,
        "last_n_messages": 1,
    }
)

debug_agent = AssistantAgent(
    name="debug_agent",
    llm_config=llm_config,
    system_message="""
    你是一個專門負責程式除錯的代理。
    當程式執行出現錯誤時，你需要：
    1. 提供修正後的程式碼
    2. 將修正結果回傳給程式執行代理重新執行
    
    除錯完成後，請說明問題所在和解決方案。
    """,
    is_termination_msg=is_termination_msg,
)

case_analyst = AssistantAgent(
    name="case_analyst",
    llm_config=llm_config,
    system_message="""
    你是一位專精於金融法規與合規案例分析的顧問。

    **你的專門任務：分析金融裁罰案例的背景與違規行為**

    請根據搜索代理提供的金融裁罰案例資料，重點分析：
    - 違規行為的具體類型與發生背景
    - 被裁罰機構的具體作為或疏失
    - 金管會或主管機關的裁罰理由和依據
    - 此案例反映的制度性風險或內控缺陷

    請將內容條理清晰地整理，重點明確，為後續的程式碼分析和法律分析提供基礎。
    
    **注意：你只負責案例背景分析，不需要分析程式碼或法條內容。**
    """,
    is_termination_msg=is_termination_msg,
)

code_analyst = AssistantAgent(
    name="code_analyst",
    llm_config=llm_config,
    system_message="""
    你是一位精通金融科技應用、模型推論與財務風控的分析師。

    **你的專門任務：分析Z3求解器程式碼的執行結果與改善建議**

    你會接收到程式執行代理的執行結果，該程式使用Z3求解器分析金融案例的可能改善方案。

    請重點分析：
    1. **變數對比分析**：對比「預設值」與「求解建議值」，指出哪些變數有明顯改善方向
    2. **金融意涵解讀**：解釋這些變數的實際意義（如資本適足率、風險資本、合規行為等）
    3. **布林變數解讀**：說明 `True`/`False` 的改變代表哪些可執行的行動方向
    4. **改善可行性**：評估這些數值調整在實務上的可行性

    請以專業但易懂的方式回覆，著重於「輸出數據的實務解讀」。

    **注意：你只負責程式碼執行結果分析，不需要分析案例背景或法條內容。**
    """,
    is_termination_msg=is_termination_msg,
)

law_analyst = AssistantAgent(
    name="law_analyst",
    llm_config=llm_config,
    system_message="""
    你是一位專業的金融法規分析師。

    **你的專門任務：從法規角度解析案例中的違法點與合規要求**

    請根據案例內容和相關法條，專門分析：
    - **違法認定**：案例中具體違反了哪些法條條文（條號與內容）
    - **法規義務**：該法條對金融機構或從業人員規定了什麼義務
    - **違規構成**：案例中的作為或不作為如何構成法規違反
    - **合規標準**：根據法條要求，機構應如何調整以符合規範

    請以法律邏輯清晰的方式進行解釋，引用具體條文並逐句對應違規行為。

    **注意：你只負責法律法規分析，不需要分析程式碼執行結果或重複案例背景。**
    """,
    is_termination_msg=is_termination_msg,
)
case_summarizer=   AssistantAgent(
    name="case_summarizer",
    llm_config=llm_config,
    system_message="""
你是一位法律資訊整理員，負責將前一位代理人提供的大量案例資料進行摘要。

請從中萃取出：
- 案例名稱或裁罰機構（若有）
- 裁罰原因或爭點類型（如：未落實洗錢防制）
- 涉及的金融行為或缺失行為
- 是否提及適用法條（可列出條號或條文名稱）

請將摘要控制在 200~300 字內，重點清楚，避免引述過長段落，也不需評論或分析。
    """,
)

user_proxy = UserProxyAgent(
    is_termination_msg=lambda x: "content" in x
    and x["content"] is not None
    and x["content"].rstrip().endswith("TERMINATE"),
    name="user_proxy",
    code_execution_config=False,
    human_input_mode="NEVER",
)
softs_parser = AssistantAgent(
    name="softs_parser",
    llm_config=llm_config,
    system_message="""
    你會接收到一個 Python 檔案名稱（如 case_123_code.py），請呼叫 `read_soft_config` 與 `get_softs_labels_and_vars` 函數獲得 soft constraints 區塊的內容與變數資訊。
    請以條列清單顯示有哪些變數可調整，並提示使用者可以輸入上下界、變為硬約束或略過。
    """
)


constraint_editor = AssistantAgent(
    name="constraint_editor",
    llm_config=llm_config,
    system_message="""
    你會根據使用者輸入的自然語言（如「將A設為硬約束」「B上下限 50~100」）來結構化成 json 指令，再傳遞給執行 Agent。
    """
)

register_function(
    read_soft_config,
    caller=softs_parser,
    executor=user_proxy,
    name="read_soft_config",
    description="讀取指定檔案中的 soft constraint 區塊程式碼"
)

register_function(
    get_softs_labels_and_vars,
    caller=softs_parser,
    executor=user_proxy,
    name="get_softs_labels_and_vars",
    description="回傳指定檔案中 soft constraint 的中文標籤與變數名"
)



register_function(
    search_and_rerank,
    caller=search_agent,
    executor=user_proxy,
    name="search_and_rerank",
    description="Re‐rank a list of candidate documents based on FlagReranker scores."
)


# register_function(
#     get_extracted_codes,
#     caller=find_code_agent,
#     executor=user_proxy,
#     name="get_extracted_codes",
#     description="獲取從法律案例中提取的程式碼列表"
# )
def extract_summary(messages, include_roles=["legal_assistant", "code_executor", "user_proxy"]):
    """
    從上層 messages 中擷取出重要角色的對話歷史，用於 nested group 的 context 準備。
    """
    summary_messages = []
    for msg in messages:
        role = msg.get("name")
        content = msg.get("content", "").strip()
        if role in include_roles and content:
            summary_messages.append({
                "name": role,
                "content": content
            })
    return summary_messages

# def build_soft_group():
#     soft_gc = GroupChat(
#         agents=[softs_parser, constraint_editor, softs_executor],
#         messages=[],
#         max_round=10
#     )
#     return GroupChatManager(
#         name="group_soft_editor",
#         groupchat=soft_gc,
#         llm_config=llm_config,
#         is_termination_msg=is_termination_msg
#     )
analysts = [case_analyst, code_analyst, law_analyst]

def state_transition(last_speaker, groupchat):
    messages = groupchat.messages
    named_messages = [msg for msg in messages if 'name' in msg]
    
    # 檢查前一個發言者
    if len(named_messages) >= 2:
        previous_agent_name = named_messages[-2]['name']
        if previous_agent_name == "search_agent":
            return case_summarizer 
    
    if last_speaker is initializer:
        return search_agent
    elif last_speaker is code_executor:
        if "exitcode: 1" in messages[-1]["content"]:
            return debug_agent
        else:
            # 程式執行成功後，先進行案例分析
            return case_analyst
    elif last_speaker is case_summarizer:
        return find_code_agent
    elif last_speaker is case_analyst:
        # 案例分析完成後，進行程式碼分析
        return code_analyst
    elif last_speaker is code_analyst:
        # 程式碼分析完成後，進行法律分析
        return law_analyst
    elif last_speaker is law_analyst:
        # 所有分析完成後，進行最終結論
        return legal_analyst
    elif last_speaker is find_code_agent:
        return code_executor
    elif last_speaker is debug_agent:
        return code_executor
    else:
        return "auto"

chat_history = []
def create_group_chat():
    return GroupChat(
        agents=[initializer,user_proxy,find_code_agent, search_agent, code_executor, debug_agent, legal_analyst,case_analyst, code_analyst, law_analyst,case_summarizer],
        messages=[],
        max_round=20,
        speaker_selection_method=state_transition,
    )

def legal_query_interface(user_query, history):
    """Gradio interface function for legal queries with group chat"""
    try:
        # 建立新的群組聊天
        group_chat = create_group_chat()
        group_chat_manager = GroupChatManager(
            groupchat=group_chat,
            llm_config=llm_config,
            is_termination_msg=is_termination_msg,
        )
                # 初始化對話
        chat_result = initializer.initiate_chat(
            group_chat_manager,
            message=user_query,
        )
        
        # 獲取完整的對話歷史
        chat_messages = chat_result.chat_history if hasattr(chat_result, 'chat_history') else []
        
        # 格式化對話歷史
        conversation_output = ""
        for i, message in enumerate(chat_messages):
            role = message.get("name", "unknown")
            content = message.get("content", "")
            tool_calls = message.get("tool_calls", [])
            
            # 根據角色標示
            if role == "user_proxy":
                continue
                #conversation_output += f"**👤 工具執行Agent:**\n{content}\n\n"
            elif role == "case_summarizer":
                conversation_output += f"**📄 案例摘要Agent：**\n{content}\n\n"
            elif role == "case_analyst":
                conversation_output += f"**📊 案例分析Agent：**\n{content}\n\n"
            elif role == "code_analyst":
                conversation_output += f"**🖥️ 程式碼分析Agent：**\n{content}\n\n"
            elif role == "law_analyst":
                conversation_output += f"**📜 法律分析Agent：**\n{content}\n\n"
            elif role == "search_agent":
                conversation_output += f"**🔍 搜索Agent：**\n{content}\n\n"
            elif role == "code_executor":
                conversation_output += f"**💻 程式執行Agent：**\n{content}\n\n"
            elif role == "debug_agent":
                conversation_output += f"**🐛 除錯Agent：**\n{content}\n\n"
            elif role == "legal_assistant":
                conversation_output += f"**⚖️ 結論分析Agent：**\n{content}\n\n"
            elif role == "find_code_agent":
                conversation_output += f"**🔍 程式碼生成Agent:**\n{content}\n\n"
            
            # 如果有工具呼叫，列出詳細資訊
            if tool_calls:
                for tool_call in tool_calls:
                    function_name = tool_call.get("function", {}).get("name", "未知函數")
                    arguments = tool_call.get("function", {}).get("arguments", {})
                    conversation_output += f"🔧 **{role} 建議使用工具：** `{function_name}`，參數：{arguments}\n\n"
            
            conversation_output += "---\n\n"
        
        # 更新歷史記錄
        history.append([user_query, conversation_output])
        #pdf_path = save_conversation_to_pdf(user_query, conversation_output)
        return history, ""  # 第三個就是傳給 gr.File()

        
    except Exception as e:
        error_message = f"發生錯誤：{str(e)}"
        history.append([user_query, error_message])
        return history, "", None

def format_search_results(query):
    """Format search results for display"""
    try:
        results = search_and_rerank(query, top_k=5)
        
        if not results['ranked_documents']:
            return "未找到相關案例。"
        
        formatted_output = f"## 查詢：{query}\n\n"
        
        for i, (doc, metadata, doc_id) in enumerate(zip(
            results['ranked_documents'], 
            results['ranked_metadatas'], 
            results['ids']
        )):
            formatted_output += f"### 案例 {i+1}\n"
            formatted_output += f"**ID：** {doc_id}\n"
            formatted_output += f"**內容：** {doc[:500]}...\n"
            if metadata:
                formatted_output += f"**元資料：** {metadata}\n"
            formatted_output += "\n---\n\n"
        
        return formatted_output
        
    except Exception as e:
        return f"搜尋時發生錯誤：{str(e)}"

# Gradio Interface
with gr.Blocks(title="Financial Compliance Agent", theme=gr.themes.Soft()) as demo:
    with gr.Row():
        gr.Markdown("# 🏛️ Financial Compliance Agent")

    
    with gr.Tab("🤖 多 AGENT 金融判例分析"):
        with gr.Row():
            query_input = gr.Textbox(
                label="請輸入您的法律問題",
                placeholder="例如：請幫我找出與『資本適足率不足』相關的最新判決案例",
                lines=2,
                scale=4
            )
            submit_btn = gr.Button("提交查詢", variant="primary", scale=1)
        
        clear_btn = gr.Button("清除對話", variant="secondary")
        chatbot = gr.Chatbot(
            label="多代理對話歷史",
            height=600,
            show_label=True
        )
        #download_pdf_btn = gr.File(label="⬇️ 下載對話 PDF")
        
        
        
        submit_btn.click(
        fn=legal_query_interface,
        inputs=[query_input, chatbot],
        outputs=[chatbot, query_input]
        )

        query_input.submit(
            fn=legal_query_interface,
            inputs=[query_input, chatbot],
            outputs=[chatbot, query_input]
        )
        clear_btn.click(
            fn=lambda: ([], ""),
            outputs=[chatbot, query_input]
        )
    
    with gr.Tab("🔍 案例搜尋"):
        with gr.Row():
            with gr.Column():
                search_input = gr.Textbox(
                    label="搜尋關鍵字",
                    placeholder="輸入法條名稱或關鍵字",
                    lines=2
                )
                search_btn = gr.Button("搜尋案例", variant="secondary")
            
            with gr.Column():
                search_results = gr.Markdown(
                    label="搜尋結果",
                    value="請輸入關鍵字開始搜尋"
                )
        
        search_btn.click(
            fn=format_search_results,
            inputs=search_input,
            outputs=search_results
        )
    
    with gr.Tab("📜 法條查詢"):
        gr.Markdown("### 🔍 法條智能搜索引擎")
        gr.Markdown("使用混合搜索技術（BM25 + 向量搜索 + 重排序）查找相關法條")
        
        with gr.Row():
            with gr.Column(scale=1):
                # 搜索參數設置
                gr.Markdown("#### ⚙️ 搜索參數")
                
                article_query = gr.Textbox(
                    label="法條查詢",
                    placeholder="例如：公司法 資本適足率",
                    lines=2
                )
                
                with gr.Row():
                    top_k_slider = gr.Slider(
                        minimum=5,
                        maximum=50,
                        value=15,
                        step=5,
                        label="初始搜索數量 (Top-K)",
                        info="從資料庫中檢索的初始結果數量"
                    )
                    
                    rerank_top_n_slider = gr.Slider(
                        minimum=3,
                        maximum=15,
                        value=5,
                        step=1,
                        label="最終結果數量",
                        info="重排序後返回的最終結果數量"
                    )
                
                hybrid_alpha_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.8,
                    step=0.1,
                    label="混合搜索權重 (Alpha)",
                    info="向量搜索權重，1.0=純向量搜索，0.0=純BM25搜索"
                )
                
                with gr.Row():
                    article_search_btn = gr.Button("🔍 搜索法條", variant="primary")
                    #analysis_btn = gr.Button("📊 相關性分析", variant="primary")
                
                # 搜索統計信息
                gr.Markdown("#### 📊 搜索說明")
                gr.Markdown("""
                - **混合搜索權重**: 調整向量搜索與BM25搜索的比重
                - **向量搜索**: 適合語義相似性搜索
                - **BM25搜索**: 適合關鍵詞精確匹配
                - **重排序**: 使用BERT模型進行精確相關性排序
                """)
            
            with gr.Column(scale=2):
                # 搜索結果顯示
                article_results = gr.Markdown(
                    label="法條搜索結果",
                    value="請輸入查詢內容並點擊搜索按鈕",
                    height=600
                )
        article_search_btn.click(
            fn=legal_article_search,
            inputs=[article_query, top_k_slider, rerank_top_n_slider, hybrid_alpha_slider],
            outputs=article_results
        )
        
        # analysis_btn.click(
        #     fn=get_related_laws_analysis,
        #     inputs=[article_query, top_k_slider, rerank_top_n_slider],
        #     outputs=article_results
        # )
        
        # Enter 鍵觸發搜索
        article_query.submit(
            fn=legal_article_search,
            inputs=[article_query, top_k_slider, rerank_top_n_slider, hybrid_alpha_slider],
            outputs=article_results
        )
        

    with gr.Tab("📖 使用說明"):
        gr.Markdown("""
        ## 📖 多代理系統說明
        
        ### 🤖 代理角色
        
        1. **🔍 搜索代理 (Search Agent)**
           - 負責搜索法律案例資料庫
           - 使用 RAG + Reranker 技術找出最相關的案例
        
        2. **💻 程式執行代理 (Code Executor Agent)**
           - 負責執行案例中包含的程式碼
           - 安全執行並記錄結果
        
        3. **🐛 除錯代理 (Debug Agent)**
           - 當程式執行出錯時進行除錯
           - 提供修正建議和解決方案
        
        4. **⚖️ 法律分析師 (Legal Analyst Agent)**
           - 整合所有資訊進行最終法律分析
           - 提供專業的法律建議和解釋
        
        ### 🔄 工作流程
        
        1. 使用者提出法律問題
        2. 搜索代理搜索相關案例
        3. 程式執行代理執行案例中的程式碼
        4. 如有錯誤，除錯代理進行修正
        5. 法律分析師提供最終分析報告
        
        ### ⚠️ 注意事項
        - 本系統僅供參考，不構成法律建議
        - 如需專業法律諮詢，請聯繫律師
        - 程式執行在安全環境中進行
        """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        debug=True
    )