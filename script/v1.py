import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
from tqdm import tqdm
from openai import OpenAI
import json
import traceback
import concurrent.futures
from threading import Lock

# 全局变量
MODEL_NAME = "deepseek-chat"
OUTPUT_PATH = f'../output/result_val_v1.json'
DATA_PATH = f'../data/val.json'

# 创建OpenAI客户端
client = OpenAI(api_key="sk-3042e2fdffd1477cb302254b61870aca", base_url="https://api.deepseek.com")

# 创建线程锁用于保护共享资源
print_lock = Lock()
results_lock = Lock()

def validate_sentence(sentence):
    """
    验证刑罚组合的合法性
    返回: (bool, str) - (是否合法, 错误信息)
    """
    # 提取各种刑罚的值
    imprisonment = sentence.get("predicted_sentence3", {}).get("value", 0)  # 有期徒刑
    life_imprisonment = sentence.get("predicted_sentence5", {}).get("value", False)  # 无期徒刑
    detention = sentence.get("predicted_sentence2", {}).get("value", 0)  # 拘役
    surveillance = sentence.get("predicted_sentence1", {}).get("value", 0)  # 管制
    probation = sentence.get("predicted_sentence4", {}).get("value", 0)  # 缓刑
    death_penalty = sentence.get("predicted_sentence6", {}).get("value", False)  # 死刑

    # 1. 检查多种主刑共存
    main_punishments = 0
    if imprisonment > 0: main_punishments += 1
    if life_imprisonment: main_punishments += 1
    if detention > 0: main_punishments += 1
    if surveillance > 0: main_punishments += 1
    if death_penalty: main_punishments += 1
    
    if main_punishments > 1:
        return False, "错误：不能同时存在多个主刑"

    # 2. 检查缓刑的合法性
    if probation > 0:
        # 2.1 检查是否有主刑
        if not (imprisonment > 0 or detention > 0):
            return False, "错误：有缓刑但无可缓刑的主刑"
        
        # 2.2 检查是否与不适用缓刑的主刑共存
        if surveillance > 0 or life_imprisonment or death_penalty:
            return False, "错误：缓刑不能与管制、无期徒刑或死刑共存"

    return True, ""

def predict_case(case):
    # 构建提示词
    prompt = f"""你是一个法律助手。请根据以下案件信息预测判决结果和量刑标签。

案件详情:
{case['case_detail']}

涉案人员:
{case['person']}

请以JSON格式返回判决结果，并严格遵守以下刑罚规则：

1. 对于每个涉案人员，以下主刑中只能选择一个：
   - 管制（月数）
   - 拘役（月数）
   - 有期徒刑（月数）
   - 无期徒刑
   - 死刑

2. 缓刑规则：
   - 缓刑必须依附于拘役或有期徒刑
   - 管制、无期徒刑、死刑不能适用缓刑

格式示例:
{{
    "case_judgment": "本院认为，...",
    "case_judgment_label": [
        {{
            "person_name": "涉案人员姓名",
            "predicted_sentence1": {{"value": 0, "desc": "管制（月数）"}},
            "predicted_sentence2": {{"value": 0, "desc": "拘役（月数）"}},
            "predicted_sentence3": {{"value": 12, "desc": "有期徒刑（月数）"}},
            "predicted_sentence4": {{"value": 24, "desc": "缓刑（月数）"}},
            "predicted_sentence5": {{"value": false, "desc": "无期徒刑"}},
            "predicted_sentence6": {{"value": false, "desc": "死刑"}}
        }}
    ]
}}

请确保返回的是合法的JSON格式，并且每个涉案人员的刑罚组合符合上述规则。"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个专业的法律助手,请根据案件信息预测判决结果"},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        
        prediction = response.choices[0].message.content
        if '```json' in prediction:
            prediction = prediction.split('```json')[1].split('```')[0]

        return prediction
        
    except Exception as e:
        return None

def process_single_case(case):
    """处理单个案件的函数"""
    try:
        prediction = predict_case(case)
        if prediction:
            try:
                prediction_json = json.loads(prediction)
                
                # 验证每个涉案人员的刑罚组合
                valid_sentences = []
                for sentence in prediction_json['case_judgment_label']:
                    is_valid, error_msg = validate_sentence(sentence)
                    if is_valid:
                        valid_sentences.append(sentence)
                    else:
                        # 移除无效的刑罚组合
                        for key in ['predicted_sentence1', 'predicted_sentence2', 'predicted_sentence3', 
                                  'predicted_sentence4', 'predicted_sentence5', 'predicted_sentence6']:
                            if key in sentence:
                                sentence[key]['value'] = 0 if isinstance(sentence[key]['value'], (int, float)) else False
                        valid_sentences.append(sentence)
                
                case['case_judgment'] = prediction_json['case_judgment']
                case['case_judgment_label'] = valid_sentences
            except Exception as e:
                case['case_judgment'] = f"Error: {str(e)}"
                case['case_judgment_label'] = []
        else:
            case['case_judgment'] = "Error: No prediction made"
            case['case_judgment_label'] = []
        return case
    except Exception as e:
        error_case = {
            "case_id": case['case_id'],
            "case_detail": case['case_detail'],
            "person": case['person'],
            "case_judgment": f"Error: {str(e)}",
            "case_judgment_label": []
        }
        return error_case

try:
    # 读取测试数据
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    results = []
    max_workers = 100
    
    # 使用线程池并发处理案件
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_case = {executor.submit(process_single_case, case): case for case in test_cases}
        
        # 使用tqdm显示进度
        with tqdm(total=len(test_cases), desc="处理案件") as pbar:
            for future in concurrent.futures.as_completed(future_to_case):
                case = future_to_case[future]
                try:
                    processed_case = future.result()
                    if processed_case:
                        with results_lock:
                            results.append(processed_case)
                except Exception as e:
                    error_case = {
                        "case_id": case['case_id'],
                        "case_detail": case['case_detail'],
                        "person": case['person'],
                        "case_judgment": f"Error: {str(e)}",
                        "case_judgment_label": []
                    }
                    with results_lock:
                        results.append(error_case)
                pbar.update(1)

    # 按case_id排序并保存结果
    sorted_results = sorted(results, key=lambda x: x['case_id'])
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(sorted_results, f, ensure_ascii=False, indent=2)

except Exception as e:
    error_case = {
        "case_id": -1,
        "case_detail": "",
        "person": [],
        "case_judgment": f"Error in main process: {str(e)}",
        "case_judgment_label": []
    }
    results = [error_case]
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

