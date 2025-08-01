import re
import json
import os
import ast
import gradio as gr
import pandas as pd
import numpy as np
import openai
from chromadb import Client, Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from FlagEmbedding import FlagReranker
from z3 import *
from config import OPENAI_API_KEY
from cone_of_influence import directed_inference_core
from search_related_laws import LegalSearchEngine
import logging

# 設定環境變數
openai.api_key = OPENAI_API_KEY
# 初始化 FlagReranker 重排序器
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

# 連接到 Chroma 資料庫
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

# 搜索並重新排序結果
def search_and_rerank(query, top_k=5):
    collection = get_chroma_collection()
    search_results = collection.query(
        query_texts=[query],
        n_results=top_k * 2
    )
    
    documents = search_results['documents'][0]
    metadatas = search_results['metadatas'][0]
    ids = search_results['ids'][0]
    
    if not documents:
        return [], [], []
    
    ranking_scores = []
    for doc in documents:
        score = reranker.compute_score([query, doc])
        ranking_scores.append(score)
    
    indexed_scores = list(enumerate(ranking_scores))
    sorted_indexed_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
    ranked_indices = [idx for idx, _ in sorted_indexed_scores[:top_k]]
    
    ranked_documents = [documents[i] for i in ranked_indices]
    ranked_metadatas = [metadatas[i] for i in ranked_indices]
    ranked_ids = [ids[i] for i in ranked_indices]
    return ranked_documents, ranked_metadatas, ranked_ids
def execute_code(code_string, vars_to_extract=None):
    """
    在當前 Python 進程中執行程式碼，並可選擇提取特定變數的模型解。
    返回 (執行輸出字串, 模型解字典)。
    """
    import io
    import sys
    from contextlib import redirect_stdout, redirect_stderr

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    
    # 確保 z3 的 sat, unsat, unknown 可用
    # local_namespace = {'__builtins__': globals()['__builtins__']} # 更安全的 exec
    local_namespace = {}


    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            exec("from z3 import *\n" + code_string, local_namespace)
        
        stdout_output = stdout_buffer.getvalue()
        stderr_output = stderr_buffer.getvalue()
        
        execution_summary = ""
        if stderr_output:
            execution_summary = f"執行成功但有警告/錯誤:\n{stderr_output}\n{stdout_output}"
        else:
            execution_summary = f"執行成功:\n{stdout_output}"

        model_values = {}
        if vars_to_extract:
            opt = local_namespace.get('opt')
            # 確保 z3 的 sat, unsat 等常數在 local_namespace 中可用
            # 如果 exec "from z3 import *" 成功，它們應該已經存在
            z3_sat = local_namespace.get('sat')
            z3_unsat = local_namespace.get('unsat')
            z3_unknown = local_namespace.get('unknown')

            if opt and hasattr(opt, 'check') and hasattr(opt, 'model') and z3_sat is not None:
                try:
                    check_result = opt.check()
                    if check_result == z3_sat:
                        m = opt.model()
                        for var_name in vars_to_extract:
                            if var_name in local_namespace:
                                z3_var = local_namespace[var_name]
                                if isinstance(z3_var, ExprRef): # 確保是 Z3 表達式
                                    try:
                                        val = m.eval(z3_var, model_completion=True)
                                        if is_true(val): model_values[var_name] = "True"
                                        elif is_false(val): model_values[var_name] = "False"
                                        elif isinstance(val, (IntNumRef, RatNumRef, FPNumRef, AlgebraicNumRef)):
                                            try: model_values[var_name] = val.as_decimal(10).rstrip('?')
                                            except: model_values[var_name] = str(val)
                                        else: model_values[var_name] = str(val)
                                    except Exception as e_eval:
                                        model_values[var_name] = f"評估變數 {var_name} 時出錯: {e_eval}"
                                else:
                                    model_values[var_name] = f"變數 {var_name} 不是有效的 Z3 表達式"
                            else:
                                model_values[var_name] = f"變數 {var_name} 未在程式碼命名空間中找到"
                    elif check_result == z3_unsat:
                        for var_name in vars_to_extract: model_values[var_name] = "UNSAT (不可滿足)"
                    else: # unknown
                        for var_name in vars_to_extract: model_values[var_name] = "Unknown (未知狀態)"
                except Exception as e_check:
                    logging.error(f"模型檢查或提取時發生錯誤: {e_check}", exc_info=True)
                    for var_name in vars_to_extract: model_values[var_name] = f"模型檢查/提取錯誤: {e_check}"
            else:
                for var_name in vars_to_extract: model_values[var_name] = "無法訪問 Z3 Optimize 物件或 SAT 狀態"
        
        return execution_summary, model_values
    except Exception as e:
        return f"執行出錯：{str(e)}\n{stderr_buffer.getvalue()}", {}

   
def gpt4_reasoning(execution_result_str, law_article): # 確保此函式期望的是字串
    prompt = f"""
下面為裁罰案例透過 Z3 Solver 執行的結果，若sat代表在修改soft的constraint後會是可以變成合法的，翻轉的部分為企業可以改善的部分，
請針對這些部分配合法規做一些說明，例如可以做哪些措施來讓這個變數可以翻轉。
程式碼輸出結果：
{execution_result_str}

相關法條:
{law_article}
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"GPT-4o reasoning 發生錯誤: {str(e)}"
def search_cases(query, top_k=3):
    """
    搜索案例並返回多個輸出。
    """
    if not query.strip():
        return "請輸入查詢內容", "", "", "", "", "", [] 
    
    documents, metadatas, ids = search_and_rerank(query, top_k)
    if not documents:
        return "沒有找到相關結果", "", "", "", "", "", []
    
    results_text = ""
    for i, (doc, meta, id) in enumerate(zip(documents, metadatas, ids)):
        results_text += f"### 案例 {i+1} (ID: {id})\n\n"
        results_text += f"**案例內容**: {doc[:500]}...\n\n"
        results_text += f"**相關法條**: {meta.get('law_article', '無')}\n\n"
        results_text += "---\n\n"
    
    top_code = metadatas[0].get("z3code", "無相關程式碼")
    law_article = metadatas[0].get("law_article", "無") # 將 law_article 的獲取提前
    
    if not top_code or top_code == "無相關程式碼":
        return results_text, top_code, "沒有可執行的程式碼", "", top_code, law_article, documents
    
    execution_summary_str, _ = execute_code(top_code) # 只取執行摘要字串
    # 如果 gpt4_reasoning 也只需要字串，則傳遞 execution_summary_str
    reasoning_result = gpt4_reasoning(execution_summary_str, law_article) 
    return results_text, top_code, execution_summary_str, reasoning_result, top_code, law_article, documents

def extract_softs_from_code(code_string):
    """
    從 top_code 中提取 softs 區塊，擷取 # === START SOFTS CONFIG === 與 # === END SOFTS CONFIG === 之間的內容
    """
    pattern = re.compile(r"# === START SOFTS CONFIG ===(.*?)# === END SOFTS CONFIG ===", re.DOTALL)
    match = pattern.search(code_string)
    if match:
        soft_block = match.group(1).strip()
        return soft_block
    else:
        return "未找到 softs 配置區塊"

def update_softs_and_execute(original_top_code, new_soft_config, vars_to_extract=None):
    """
    將修改後的 softs 區塊置換回 top_code，然後重新執行程式碼。
    返回 (更新後的程式碼, 執行輸出字串, 模型解字典)。
    """
    pattern = re.compile(r'(# === START SOFTS CONFIG ===)(.*?)(# === END SOFTS CONFIG ===)', re.DOTALL)
    if pattern.search(original_top_code):
        updated_top_code = pattern.sub(r'\1\n' + new_soft_config + r'\n\3', original_top_code)
    else:
        # 如果沒有標記，則附加到末尾 (或選擇其他策略)
        updated_top_code = original_top_code + "\n# === START SOFTS CONFIG ===\n" + new_soft_config + "\n# === END SOFTS CONFIG ===\n"
    
    execution_result_str, model_values = execute_code(updated_top_code, vars_to_extract)
    return updated_top_code, execution_result_str, model_values


# --- 以下為使用表單介面修改 softs 配置的輔助函式 ---
def parse_softs_config_dynamic(softs_text):
    """
    只用正則把 label, var name, expected_str 都以字串方式抓出來
    """
    pattern = re.compile(r'\(\s*"([^"]+)"\s*,\s*([\w_]+)\s*,\s*(.*?)\s*\)')
    matches = pattern.findall(softs_text)
    softs_list = []
    for label, var_name, expected_str in matches:
        softs_list.append({
            "label": label,
            "var": var_name,         # 暫時先當成字串
            "expected": expected_str # 也先當成字串
        })
    return softs_list

def build_softs_dataframe(softs_text, code_string):
    """
    將 softs 配置解析為 DataFrame，正確評估改善前後的值
    """
    # 1. 解析純字串版 softs 配置
    parsed_raw = parse_softs_config_dynamic(softs_text)
    
    # 2. 動態執行程式碼獲取 Z3 環境
    ns = {}
    exec("from z3 import *\n" + code_string, ns)
    
    # 3. 取得 solver 與求解結果
    opt = ns.get('opt', None)
    model = None
    if opt and opt.check().r == 1:
        model = opt.model()
    
    # 4. 構建 DataFrame 行
    rows = []
    for item in parsed_raw:
        label = item["label"]
        var_name = item["var"]
        exp_str = item["expected"]
        
        # 5. 將變數名對應到 Z3 物件
        var_expr = var_name
        if var_name in ns:
            var_expr = ns[var_name]
            
        # 6. 處理期望值（可能是變數名、字面值或表達式）
        exp_expr = exp_str
        if isinstance(exp_str, str) and exp_str in ns:
            exp_expr = ns[exp_str]
        else:
            try:
                exp_expr = ast.literal_eval(exp_str)
            except:
                pass
                
        # 7. 評估期望值
        expected_val = exp_expr
        if model and isinstance(exp_expr, ExprRef):
            v = model.eval(exp_expr, model_completion=True)
            if is_true(v):
                expected_val = True
            elif is_false(v):
                expected_val = False
            else:
                try:
                    expected_val = float(v.as_decimal(10).rstrip('?'))
                except:
                    expected_val = str(v)
                    
        # 8. 評估實際值
        actual_val = None
        if model and isinstance(var_expr, ExprRef):
            v2 = model.eval(var_expr, model_completion=True)
            if is_true(v2):
                actual_val = True
            elif is_false(v2):
                actual_val = False
            else:
                try:
                    actual_val = float(v2.as_decimal(10).rstrip('?'))
                except:
                    actual_val = str(v2)
                    
        # 9. 判斷數值型與設定上下界
        is_numeric = isinstance(expected_val, (int, float)) and not isinstance(expected_val, bool)
        lb = round(max(0, float(expected_val)*0.8), 2) if is_numeric else None
        ub = round(float(expected_val)*1.2, 2) if is_numeric else None
        
        # 10. 構建資料行
        row = {
            "label": label,
            "actual": expected_val,
            "expected": actual_val,
            "hard": False,
            "use_bounds": is_numeric,
            "lower_bound": lb,
            "upper_bound": ub
        }
        rows.append(row)
    
    # 11. 返回 DataFrame 與純字串配置的 JSON
    df = pd.DataFrame(rows)
    return df, json.dumps(parsed_raw, ensure_ascii=False)

def extract_softs_description_and_varnames(code_string):
    """從程式碼中提取 softs 陣列的中文描述和變數名對應關係"""
    if not code_string or code_string == "無相關程式碼":
        return ["請先搜索案例"], {}
    
    try:
        # 使用正則表達式找出 softs 數組定義部分
        pattern = re.compile(r'softs\s*=\s*\[(.*?)\]', re.DOTALL)
        match = pattern.search(code_string)
        if not match:
            return ["代碼中未找到 softs 數組"], {}
            
        softs_block = match.group(1)
        
        # 解析每行，提取中文描述和變數名
        line_pattern = re.compile(r'\(\s*"([^"]+)"\s*,\s*([\w_]+)\s*,')
        matches = line_pattern.findall(softs_block)
        
        # 建立選單選項和映射表
        chinese_descriptions = []
        var_map = {}
        
        for desc, var_name in matches:
            chinese_descriptions.append(desc)  # 直接添加中文描述
            var_map[desc] = var_name
                
        return chinese_descriptions, var_map
    except Exception as e:
        print(f"提取中文描述和變數名時發生錯誤: {str(e)}")
        return ["提取過程發生錯誤"], {}

def update_softs_from_df(original_top_code, parsed_softs_json, df_data, law_article, original_execution_result, case_documents):
    """
    根據更新後的 DataFrame 與原始解析配置重構 softs 區塊，
    使用者勾選 (hard True) 的加入硬約束 (opt.add)，
    其他的加入軟約束 (opt.add_soft)。
    數值型硬約束可選擇使用上下界值或嚴格等於。
    最後將更新後的 softs 區塊置換回 top_code，再重新執行程式碼。
    """
    parsed_softs = json.loads(parsed_softs_json)
    updated_values_list = df_data if isinstance(df_data, list) else df_data.to_dict(orient="records")
    df_data_map = {row["label"]: row for row in updated_values_list}

    hard_lines = []
    soft_lines = []
    modified_items = []
    bound_lines = []
    
    vars_made_hard_for_solution_extraction = [] # 收集設為硬約束的 Z3 變數名稱

    for item in parsed_softs:
        label = item["label"]
        var = item["var"] # Z3 變數的字串名稱
        df_row_data = df_data_map.get(label)

        if not df_row_data:
            continue

        make_hard = df_row_data.get("hard", False)
        
        value_to_use_for_constraint = None
        if make_hard:
            value_to_use_for_constraint = df_row_data["expected"]
            vars_made_hard_for_solution_extraction.append(var) # 添加 Z3 變數名稱
        else:
            value_to_use_for_constraint = df_row_data["actual"]
        
        new_val_str = ""
        if isinstance(value_to_use_for_constraint, bool) or str(value_to_use_for_constraint).lower() in ["true", "false"]:
            new_val_str = "True" if str(value_to_use_for_constraint).lower() == "true" or value_to_use_for_constraint is True else "False"
        elif isinstance(value_to_use_for_constraint, (int, float)) or \
             (isinstance(value_to_use_for_constraint, str) and str(value_to_use_for_constraint).replace('.', '', 1).replace('-', '', 1).isdigit()):
            try:
                num_val = float(value_to_use_for_constraint)
                if num_val == int(num_val): new_val_str = str(int(num_val))
                else: new_val_str = f"{num_val:.2f}"
            except: new_val_str = str(value_to_use_for_constraint)
        else: 
            new_val_str = f'"{value_to_use_for_constraint}"'
        
        line = f'    ("{label}", {var}, {new_val_str})'

        if make_hard:
            use_bounds = df_row_data.get("use_bounds", False)
            lower_bound = df_row_data.get("lower_bound")
            upper_bound = df_row_data.get("upper_bound")
            val_for_numeric_check = df_row_data["expected"]
            is_numeric_val = isinstance(val_for_numeric_check, (int, float)) or \
                             (isinstance(val_for_numeric_check, str) and str(val_for_numeric_check).replace('.', '', 1).replace('-', '', 1).isdigit())
            
            if is_numeric_val and use_bounds and lower_bound is not None and upper_bound is not None:
                bound_lines.append(f'    # 為 {label} 增加上下界')
                bound_lines.append(f'    opt.add({var} >= {float(lower_bound):.2f})')
                bound_lines.append(f'    opt.add({var} <= {float(upper_bound):.2f})')
                constraint_desc = f"- {label} (目標值: {new_val_str}): 變為硬約束，使用上下界：{float(lower_bound):.2f} ≤ {var} ≤ {float(upper_bound):.2f}"
            else:
                hard_lines.append(line)
                constraint_desc = f"- {label} (目標值: {new_val_str}): 變為硬約束，使用嚴格等於 ({var} == {new_val_str})"
            modified_items.append(constraint_desc)
        else:
            soft_lines.append(line)
    
    new_softs_block = "hard_constraints = [\n" + ",\n".join(hard_lines) + "\n]\n\n"
    new_softs_block += "softs = [\n" + ",\n".join(soft_lines) + "\n]\n\n"
    
    if hard_lines:
        new_softs_block += "# 加入嚴格等於的硬約束\n"
        new_softs_block += "for (label, expr, expected_value) in hard_constraints:\n"
        new_softs_block += "    opt.add(expr == expected_value)\n\n"
    
    if bound_lines:
        new_softs_block += "# 加入數值型硬約束的上下界\n"
        clean_bound_lines = [line.lstrip() for line in bound_lines]
        formatted_bound_lines = [line for line in clean_bound_lines]
        new_softs_block += "\n".join(formatted_bound_lines) + "\n\n"
    
    new_softs_block += "# 加入軟約束\n"
    new_softs_block += "for (label, expr, expected_value) in softs:\n"
    new_softs_block += "    opt.add_soft(expr == expected_value, weight=1, id=label)\n"
    
    updated_code, new_execution_result_str, solved_hard_constraint_model_values = \
        update_softs_and_execute(original_top_code, new_softs_block, list(set(vars_made_hard_for_solution_extraction)))
    
    modified_description = "\n".join(modified_items) if modified_items else "無項目被修改為硬約束"
    
    current_case_content = "無案例內容提供"
    if case_documents and isinstance(case_documents, list) and len(case_documents) > 0:
        current_case_content = case_documents[0] # 假設第一個文檔是主要案例內容

    reasoning_result = gpt4_reasoning_updated(
        new_execution_result_str, 
        law_article, 
        modified_description, 
        original_top_code,
        original_execution_result,
        solved_hard_constraint_model_values,
        current_case_content # 新增案例內容參數
    )
    
    return updated_code, new_execution_result_str, reasoning_result

def analyze_cone_of_influence(top_code, variable_of_interest):
    """
    使用 directed_inference_core 分析變數的影響圈，並顯示其當前解。
    """
    if not top_code or not variable_of_interest:
        return "請先搜索案例並輸入需要分析的變數名稱"
    
    try:
        # 準備執行環境
        ns = {}
        exec("from z3 import *\n" + top_code, ns)
        
        # 獲取 Optimize 物件
        opt = ns.get('opt')
        if not isinstance(opt, Optimize):
            return "在程式碼中找不到 Z3 Optimize 物件 'opt'"

        # 獲取目標變數的 Z3 物件
        z3_var_of_interest = ns.get(variable_of_interest)
        if not isinstance(z3_var_of_interest, ExprRef):
            return f"在程式碼中找不到變數 '{variable_of_interest}' 或其不是有效的 Z3 表達式"

        current_value_str = "未知或無法求解"
        # 檢查模型是否已求解且可滿足
        if opt.check().r == 1: # Z3_L_TRUE means sat
            model = opt.model()
            evaluated_value = model.eval(z3_var_of_interest, model_completion=True)
            
            # 格式化求解值
            if is_true(evaluated_value):
                current_value_str = "True"
            elif is_false(evaluated_value):
                current_value_str = "False"
            elif isinstance(evaluated_value, (IntNumRef, RatNumRef, FPNumRef, AlgebraicNumRef)):
                try:
                    # 嘗試轉換為小數，如果失敗則轉為字串
                    decimal_val = evaluated_value.as_decimal(10).rstrip('?')
                    current_value_str = decimal_val
                except:
                    current_value_str = str(evaluated_value)
            else:
                current_value_str = str(evaluated_value)
        elif opt.check().r == -1: # Z3_L_FALSE means unsat
            current_value_str = "模型不可滿足 (unsat)"
        else: # Z3_L_UNDEF means unknown
            current_value_str = "模型求解狀態未知"

        # 執行影響圈分析
        core_cons, core_var_domains = directed_inference_core(opt, variable_of_interest)
        
        # 格式化結果輸出
        formatted_result = f"## 變數影響圈分析結果\n\n"
        formatted_result += f"目標變數: **{variable_of_interest}**\n"
        formatted_result += f"原始模型解: **{current_value_str}**\n\n"
        
        # 格式化核心約束
        formatted_result += "### 核心約束條件:\n"
        if core_cons:
            for i, constraint in enumerate(core_cons):
                formatted_result += f"{i+1}. `{constraint}`\n"
        else:
            formatted_result += "無相關約束條件\n"
        
        # 格式化核心變數域
        formatted_result += "\n### 核心變數域:\n"
        if core_var_domains:
            for var, domain in core_var_domains.items():
                formatted_result += f"- **{var}**: {domain}\n"
        else:
            formatted_result += "無相關變數域\n"
            
        return formatted_result
    except Exception as e:
        # 考慮加入更詳細的錯誤日誌
        logging.error(f"分析影響圈時發生錯誤: {e}", exc_info=True)
        return f"分析過程中發生錯誤: {str(e)}"
def gpt4_reasoning_updated(execution_result, law_article, modified_constraints, original_code, original_execution_result, solved_hard_constraint_values=None, case_content=None):
    """
    專門針對修改後的約束（從軟變硬）進行法律推理
    
    參數：
    execution_result: 修改後 Z3 程式的執行結果 (字串)
    law_article: 相關法條
    modified_constraints: 被修改的約束描述（從軟變硬的項目）
    original_code: 原始未修改的 Z3 程式碼
    original_execution_result: 原始 Z3 程式的執行結果
    solved_hard_constraint_values: 一個字典，包含被設為硬約束的變數及其在模型中的解 (如果 SAT)
    case_content: 原始案例的內容描述
    """
    
    solved_values_str = "未提供或模型不可滿足。"
    if solved_hard_constraint_values:
        items = []
        for var, val in solved_hard_constraint_values.items():
            items.append(f"  - 變數 '{var}': {val}")
        if items:
            solved_values_str = "\n".join(items)
        else:
            solved_values_str = "無特定硬約束變數的解被提取，或模型不可滿足。"

    case_content_str = case_content if case_content else "未提供案例內容。"

    prompt = f"""
下面是根據特定金融案例的 Z3 Solver 執行結果。使用者已將部分軟約束(soft constraint)轉換為硬約束(hard constraint)，
請針對這些修改進行分析，說明這些特定變數從軟約束轉為硬約束後對結果的影響。

原始案例內容：
```
{case_content_str}
```

原始 Z3 程式碼執行結果：
```
{original_execution_result}
```

被修改為硬約束的項目描述 (括號內為使用者設定的目標或範圍)：
{modified_constraints}

針對這些被修改為硬約束的項目，其在模型求解後的實際值如下 (如果求解成功且變數可評估):
{solved_values_str}

修改後 Z3 程式碼的完整執行結果：
{execution_result}

相關法條：
{law_article}

請基於以上所有資訊進行分析：
1.  解釋將這些特定約束從軟變硬後，對整體案例合規性的影響。
2.  請你針對這些約束結合法條多做一些細部的說明，像是如何提升或減少該變數能夠達成合規
3.  請你注重於法律與數據之間的關聯性做分析，並提供具體的建議或潛在的風險點。
4.  請你針對有翻轉的變數提出討論，並說明為何會得到該解與哪個法條有關聯
5.  你需要提供一些合規性的建議與變數更改值後的一些洞察，這些建議應該是針對如何改善合規性的具體措施。
6.  請你針對轉換成硬約束的變數詳細的說明，像是求解的結果以及可以改善的方向。
請務必多描述數據的部分，並將模型解的實際值與設定目標進行比較分析，這是我們這個工具的特色。
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"GPT-4o reasoning 發生錯誤: {str(e)}"

# --- 建立 Gradio 介面 ---
with gr.Blocks(title="金融案例Agent") as demo:
    gr.Markdown("# 金融案例Agent")
    gr.Markdown("輸入查詢，系統將搜尋最相關的裁罰案例、提取程式碼中的 softs 設定，並進行法律推理。")
    
    with gr.Row():
        query_input = gr.Textbox(label="請輸入查詢內容", lines=3)
        top_k_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="顯示結果數量")
    search_button = gr.Button("搜索")
    documents_state = gr.State([])
    # hidden state 用來儲存 top_code (後續用於更新 softs)
    top_code_state = gr.State()
    law_article_state = gr.State()
    with gr.Row():
        with gr.Column():
            results_output = gr.Markdown(label="搜索結果")
        with gr.Column():
            code_output = gr.Code(language="python", label="程式碼 (top_code)")
            execution_output = gr.TextArea(label="程式碼執行結果", lines=10)
            reasoning_output = gr.Markdown(label="GPT-4o 法律推理結果")
    variable_dropdown = gr.State([])  # 先用 State 暫存
    var_mapping_state = gr.State({})
    # 搜索按鈕回傳 5 個輸出，其中第二個（top_code）同時更新 hidden state
    search_button.click(
        fn=search_cases,
        inputs=[query_input, top_k_slider],
        outputs=[results_output, code_output, execution_output, reasoning_output, top_code_state, law_article_state,documents_state]  # 增加 law_article_state
    ).then(
        fn=extract_softs_description_and_varnames,
        inputs=[top_code_state],
        outputs=[variable_dropdown, var_mapping_state]
    )
    
    with gr.Tab("Extract and Modify Softs Constraint"):
        #gr.Markdown("從搜索結果中提取 softs 配置，並透過表單方式修改各個條件的 value（僅顯示 label 與 value）。")
        # 顯示從 top_code 中提取的 softs 配置（純文字版，可供參考）
        softs_config_text = gr.Textbox(label="Soft Constraint", lines=10)
        # 當 hidden state (top_code_state) 更新時，自動提取 softs 配置
        top_code_state.change(fn=extract_softs_from_code, inputs=top_code_state, outputs=softs_config_text)
        
        #gr.Markdown("【表單方式調整】根據上方 softs 配置生成調整表單：")
        # 利用 DataFrame 呈現 softs 配置 (僅包含 label 與 value)
        softs_df = gr.Dataframe(
            label="Softs Constraint", 
            headers=["label", "expected", "actual", "hard", "use_bounds", "lower_bound", "upper_bound"], 
            datatype=["str", "str", "str", "bool", "bool", "number", "number"],
            interactive=True
        )
        parsed_softs_state = gr.State()
        generate_df_button = gr.Button("Submit")
        generate_df_button.click(
            fn=build_softs_dataframe,
            inputs=[softs_config_text, top_code_state],
            outputs=[softs_df, parsed_softs_state]
        )
        gr.Markdown("Update Constraint")
        updated_code_output = gr.Code(language="python", label="更新後的 top_code")
        new_execution_output = gr.TextArea(label="更新後程式碼執行結果", lines=10)
        update_df_button = gr.Button("Update and Execute")
        new_reasoning_output = gr.Markdown(label="更新後 GPT-4o 法律推理結果")
        update_df_button.click(
            fn=update_softs_from_df,
            inputs=[top_code_state, parsed_softs_state, softs_df, law_article_state, execution_output, documents_state], # 新增 documents_state
            outputs=[updated_code_output, new_execution_output, new_reasoning_output]
        )
    with gr.Tab("Cone of Influence 影響圈分析"):
        gr.Markdown("## 變數影響圈分析")
        gr.Markdown("請選擇想要分析的變數，系統將分析該變數的影響圈關係")
        
        # 使用保存中文到變數映射的 State 組件
        var_mapping_state = gr.State({})
        
        # 使用下拉選單，顯示中文描述
        variable_dropdown = gr.Dropdown(
            label="選擇變數", 
            choices=[],
            interactive=True,
            allow_custom_value=True,
            max_choices=1  # 限制最多選擇一個
        )
        
        analyze_button = gr.Button("進行影響圈分析")
        influence_result = gr.Markdown(label="影響圈分析結果")
        
        # 處理下拉選單更新
# 處理下拉選單更新
        def update_variable_dropdown(top_code):
            chinese_descriptions, var_map = extract_softs_description_and_varnames(top_code)
            if not isinstance(chinese_descriptions, list):
                chinese_descriptions = [chinese_descriptions]
            # 這裡用 gr.update 來回傳新的 choices (以及重置 value)
            return gr.update(choices=chinese_descriptions, value=None), var_map
                
        # 當 top_code 更新時更新下拉選單
        top_code_state.change(
            fn=update_variable_dropdown,
            inputs=top_code_state,
            outputs=[variable_dropdown, var_mapping_state]
        )
        
        # 處理影響圈分析
        def process_influence_analysis(top_code, selected_option, var_map):
            # 從選擇中獲取實際變數名
            var_name = var_map.get(selected_option, selected_option)
            return analyze_cone_of_influence(top_code, var_name)
        
        try:
            search_engine = LegalSearchEngine(
                persist_directory="./chroma_db",
                collection_name="law_articles_csv"
            )
        except Exception as e:
            print(f"初始化搜索引擎時出錯: {e}")
        
        def query_rewriting(variable_name, original_case=None, top_code=None):
            """
            生成適合向量數據庫搜索的法律查詢語句
            
            參數:
            variable_name: 變數名稱（可能是中文或英文）
            original_case: 原始案例內容（可選）
            top_code: 相關程式碼（可選）
            """
            # 嘗試從程式碼中提取變數的中文描述（如果提供了英文變數名）
            chinese_desc = ""
            if top_code and variable_name and not any(ord(c) > 127 for c in variable_name):
                try:
                    # 使用正則表達式找出與英文變數名對應的中文描述
                    pattern = re.compile(rf'["\']([^"\']+)["\']\s*,\s*{variable_name}\s*,')
                    match = pattern.search(top_code)
                    if match:
                        chinese_desc = match.group(1)
                except:
                    pass
            
            # 提取案例的關鍵資訊
            case_summary = original_case
            
            prompt = f"""
        我希望你扮演一位專業的法律檢索專家。

        給定法律變數資訊：
        - 變數名稱：{variable_name}
        {f"- 變數中文描述：{chinese_desc}" if chinese_desc else ""}

        {f"相關案例摘要：{case_summary}" if case_summary else ""}

        請基於以上資訊生成一個查詢問題用來查詢向量資料庫。

        要求：
        1. 包含該法律變數的同義詞、近義表達或常見用語；
        2. 補充相關法律名稱、章節或條款編號；
        3. 語句結構簡潔，適合向量檢索；
        4. 僅返回最終的檢索語句，不要任何其他說明或步驟。
        5. 如果變數名是英文但有中文描述，請主要使用中文描述。
            """
    
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4.1-nano",
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"GPT-4o reasoning 發生錯誤: {str(e)}"

        def search_laws_for_variable(top_code, selected_option, var_map,documents_value):
    # 從選擇中獲取實際變數名
            var_name = var_map.get(selected_option, selected_option)
            
            # 執行影響圈分析
            influence_result = analyze_cone_of_influence(top_code, var_name)
            
            # 生成查詢並搜索法條
            try:
                search_engine = LegalSearchEngine(
                    persist_directory="./chroma_db",
                    collection_name="law_articles_csv"
                )
                
                # 使用變數名產生查詢
                query = query_rewriting(
                    variable_name=var_name,
                    original_case=documents_value[0] if documents_value and len(documents_value) > 0 else None,
                    top_code=top_code
                )
                                
                # 搜索相關法條
                direct_relevant, indirectly_relevant = search_engine.get_related_laws(
                    query, top_k=10, rerank_top_n=5
                )
                
                # 格式化法條搜索結果
                law_result = "## 相關法條搜索結果\n\n"
                law_result += f"### 搜索詞: {query}\n\n"
                
                law_result += "### 直接相關法條:\n"
                if direct_relevant:
                    for i, law in enumerate(direct_relevant):
                        law_result += f"{i+1}. **{law['metadata'].get('法律名稱', '未知法律')}  {law['metadata'].get('條', '未知條款')} **\n"
                        law_result += f"   {law.get('content', '無內容')}\n\n"
                else:
                    law_result += "未找到直接相關法條\n\n"
                    
                law_result += "### 間接相關法條:\n"
                if indirectly_relevant:
                    for i, law in enumerate(indirectly_relevant):
                        law_result += f"{i+1}. **{law['metadata'].get('法律名稱', '未知法律')}  {law['metadata'].get('條', '未知條款')} **\n"
                        law_result += f"   {law.get('content', '無內容')}\n\n"
                else:
                    law_result += "未找到間接相關法條\n\n"
                    
                # 合併影響圈分析和法條搜索結果
                combined_result = influence_result + "\n\n" + law_result
                return combined_result
            except Exception as e:
                return influence_result + f"\n\n## 法條搜索失敗\n\n錯誤: {str(e)}"
        
        analyze_button.click(
            fn=search_laws_for_variable,
            inputs=[top_code_state, variable_dropdown, var_mapping_state, documents_state],
            outputs=influence_result
        )
if __name__ == "__main__":
    demo.launch()
