import os
import json
import pandas as pd
import openai
import logging
from typing import List, Dict, Any
from search_related_laws import LegalSearchEngine
from config import OPENAI_API_KEY

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
search_engine = LegalSearchEngine(
            persist_directory="./chroma_db",
            collection_name="law_articles_csv_version1"
        )
# 設定OpenAI API金鑰
openai.api_key = OPENAI_API_KEY

def read_csv_data(csv_path: str) -> pd.DataFrame:
    """
    讀取CSV檔案
    """
    try:
        df = pd.read_csv(csv_path)
        df = df[90:]
        logger.info(f"成功讀取CSV檔案，共 {len(df)} 筆資料")
        
        return df
    except Exception as e:
        logger.error(f"讀取CSV檔案時出錯: {e}")
        raise

def generate_search_queries(law_text: str) -> List[str]:
    """
    使用LLM分析法條，生成需要額外搜索的問題
    
    Args:
        law_text: 法條文本
        
    Returns:
        生成的搜索問題列表
    """
    # 準備提示詞
    prompt = f"""
請分析以下法條，識別可能不夠清楚或需要額外說明的部分。
請生成3-5個針對這些不明確部分的搜索問題，以便找到補充說明或詳細定義，請注意，請你不要過度生成太多問題！

法條:
{law_text}

請特別關注以下方面：
1. 法條中提到的專業術語的定義或計算方式
2. 法條中提到的閾值或標準的具體數值
3. 請你針對法條中提到的機構是什麼來詢問問題，像是如果是保險法就會是保險業、銀行法就是銀行業等

只需列出問題清單，每個問題應簡潔明確，使用完整問句。
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500
        )
        
        result = response.choices[0].message.content
        
        # 提取問題列表
        queries = []
        for line in result.strip().split("\n"):
            line = line.strip()
            # 跳過空行或僅包含數字、符號的行
            if not line or line.isdigit() or line in [".", "-", "*"]:
                continue
            # 清理行首的數字、符號等
            import re
            line = re.sub(r'^[\d\.\-\*\(\)]+\s*', '', line)
            if line:
                queries.append(line)
        
        logger.info(f"生成了 {len(queries)} 個搜索查詢")
        return queries
    
    except Exception as e:
        logger.error(f"生成搜索查詢時出錯: {e}")
        # 提供一些默認查詢
        return ["資本適足率的定義和計算方法是什麼？", 
                "保險業資本等級的具體分類標準是什麼？", 
                "保險業財務改善計畫應包含哪些內容？"]

def search_and_evaluate(queries: List[str], original_law_text: str) -> List[Dict[str, Any]]:
    """
    使用生成的查詢搜索相關法條，並評估它們的有用性
    
    Args:
        queries: 搜索查詢列表
        original_law_text: 原始法條文本
        
    Returns:
        有用的法條列表
    """
    # 初始化搜索引擎
    
    # 儲存所有找到的相關法條
    all_found_laws = []
    
    # 對每個查詢進行搜索
    for query in queries:
        try:
            direct_relevant, indirectly_relevant = search_engine.get_related_laws(
                query, top_k=10, rerank_top_n=5
            )
            
            # 合併結果
            all_results = direct_relevant + indirectly_relevant
            
            # 添加不重複的法條到結果列表
            for result_item in all_results: # Renamed 'result' to 'result_item' to avoid conflict
                if not any(law["id"] == result_item["id"] for law in all_found_laws):
                    all_found_laws.append(result_item)
            
        except Exception as e:
            logger.error(f"執行查詢 '{query}' 時出錯: {e}")
    
    # 如果找到的法條太少，全都保留
    if len(all_found_laws) <= 5:
        return all_found_laws
    
    # 使用LLM評估法條的有用性
    found_laws_text = "\n\n".join([
        f"ID: {law['id']}\n"
        f"法律: {law['metadata'].get('法律名稱', '未知法律')} 第 {law['metadata'].get('條', '未知條款')} 條\n"
        f"內容: {law['content']}"
        for law in all_found_laws
    ])

    prompt = f"""
請評估以下搜索結果是否對理解變數有幫助。

原始法條:
{original_law_text}


搜索結果:
{found_laws_text}

請識別對理解原始法條有幫助的法條，特別是提供了:
1. 專業術語的明確定義
2. 具體的計算方法或公式

請以JSON格式回答，列出有用法條的ID：
{{
    "useful_law_ids": ["id1", "id2", ...]
}}
請注意，在生成的時候請你不用用``` json 開頭 跟 ``` 結尾
    """
    
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        
        result = response.choices[0].message.content
        logger
        # 解析JSON
        try:
            import json
            result_json = json.loads(result)
            useful_law_ids = result_json.get("useful_law_ids", [])
            
            # 篩選有用的法條
            useful_laws = [law for law in all_found_laws if law["id"] in useful_law_ids]
            
            logger.info(f"從 {len(all_found_laws)} 個結果中，識別出 {len(useful_laws)} 個有用的法條")
            return useful_laws
            
        except json.JSONDecodeError:
            logger.error(f"解析LLM返回的JSON失敗，使用所有搜索結果")
            return all_found_laws
        
    except Exception as e:
        logger.error(f"評估法條有用性時出錯: {e}")
        return all_found_laws

def process_csv_laws(batch_size: int = 30): # 新增 batch_size 參數，預設為50
    """
    處理CSV文件中的法條，找尋相關補充法條，並分批儲存結果。
    """
    # 讀取CSV檔案
    csv_path = "full_laws2014.csv"
    df = read_csv_data(csv_path)
    
    # 確認欄位
    columns = df.columns.tolist()
    logger.info(f"CSV欄位: {columns}")
    
    # 假設第二列為法條欄位
    law_column = columns[1] if len(columns) > 1 else columns[0]
    logger.info(f"使用 '{law_column}' 欄位作為法條資料")
    
    # 儲存所有結果
    all_processed_results = [] # 重新命名以避免混淆
    output_path = "./law_supplement_results_2014_90.json"
    
    # 處理每一行資料
    for i, row in df.iterrows():
        i = i-90
        logger.info(f"處理第 {i+1}/{len(df)} 筆資料")
        
        # 獲取法條文本
        law_text = row[law_column]
        if not isinstance(law_text, str) or not law_text.strip():
            logger.warning(f"跳過第 {i+1} 筆資料，法條文本為空")
            continue
        
        # 生成搜索查詢
        queries = generate_search_queries(law_text)
        
        # 執行搜索並評估結果
        useful_laws = search_and_evaluate(queries, law_text)
        
        # 保存結果
        current_row_result = { # 重新命名以避免混淆
            "original_law_text": law_text,
            "generated_queries": queries,
            "supplementary_laws": useful_laws
        }
        
        # 如果CSV有其他列，也保存這些資訊
        for col in columns:
            if col != law_column:
                current_row_result[col] = row[col]
        
        all_processed_results.append(current_row_result)
        
        # 每處理完一批次或到達檔案末尾時保存
        if (i + 1) % batch_size == 0 or (i + 1) == len(df):
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(all_processed_results, f, ensure_ascii=False, indent=2)
                logger.info(f"已處理 {len(all_processed_results)}/{len(df)} 筆資料，結果已更新至 {output_path}")
            except Exception as e:
                logger.error(f"保存批次結果時出錯: {e}")
    
    logger.info(f"所有資料處理完成。最終結果已保存至 {output_path}")

if __name__ == "__main__":
    process_csv_laws()