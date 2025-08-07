# -*- coding: utf-8 -*-
import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
from tqdm import tqdm
from openai import OpenAI
import json
import traceback
import concurrent.futures
from threading import Lock
import random

# 全局变量
MODEL_NAME = "deepseek-chat"
TRAIN_DATA_PATH = '../data/train.json'
N_SHOT = 8

# DATA_PATH = f'../data/val.json'
# OUTPUT_PATH = f'../output/result_val_v3_n{N_SHOT}.json'

DATA_PATH = f'../data/IntentionalInjury_cases_test.json'
OUTPUT_PATH = f'../output/result_v31_n{N_SHOT}.json'

# 加载训练数据
def load_train_examples():
    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    return train_data

# 随机选择n个训练示例
def get_random_examples(n=N_SHOT):
    try:
        train_data = load_train_examples()
        return random.sample(train_data, n)
    except Exception as e:
        print(f"Warning: Failed to load training examples: {e}")
        return []

# 格式化单个示例
def format_example(example):
    return f"""案例：
{example['case_detail']}

涉案人员：
{example['person']}

判决结果：
{example['case_judgment']}

量刑标签：
{json.dumps(example['case_judgment_label'], ensure_ascii=False, indent=2)}
"""

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

def predict_case(case, n_shot=N_SHOT):
    # 获取随机示例
    examples = get_random_examples(n_shot)
    examples_text = "\n\n".join([format_example(ex) for ex in examples])
    
    # 构建提示词
    prompt = f"""你是一个法律助手。请根据以下案件信息预测判决结果和量刑标签。

以下是一些类似案例的判决示例：

{examples_text}

以下是一些情节很轻且没什么严重伤亡的案例，类似这种的可以直接判拘役而不必徒刑:

 **省**市**区人民法院\n刑事判决书\n(2016)*0105刑初6134*号\n公诉机关**市**区人民检察院。\n被告人童*甲，男，1982 年 10 月 10 日出生，汉族，**省**市人，文化程度初中，户籍地址四川省**市**区**镇**村九组 16 号。因本案于 2016 年 7 月 12 日被羁押，同日被\n刑事拘留，同年 8 月 17 日被逮捕。现被羁押于**市**区看守\n所。\n辩护人邓*甲、伍*甲，广东海智律师事务所律师。\n**市**区人民检察院以*海检诉刑诉[2016]1483 号起\n诉书指控被告人童*甲于 2016 年 5 月 19 日零时许，伙同其余同\n案人（均另案处理）至本市**区西畔里沙溪东新树网吧楼下，\n持钢筋对被害人李某某、邓某某进行殴打（经鉴定，邓某某的损\n伤构成轻伤一级）。得手后，被告人童*甲等人逃离现场。2016\n年 7 月 12 日，被告人童*甲被民警抓获。于 2016 年 11 月 10 日\n向本院以被告人童*甲犯故意伤害罪提起公诉，并附随案移送的\n证据予以证实。本院依法适用速裁程序，实行独任审判，公开开\n庭审理了本案。**市**区人民检察院指派检察员许*甲出庭\n－1－\n支持公诉，被告人童*甲及其辩护人邓*甲到庭参加了诉讼，在\n开庭审理过程中亦无异议，现已审理终结。

 **省**市**区人民法院\n刑事判决书\n（2016）* 010* 刑初 3109* 号\n公诉机关**市**区人民检察院。\n被告人刘*甲，男，1973 年 12 月 5 日出生，汉族，**省\n**市人，文化程度小学，户籍地址**市**区横枝岗一街二\n巷 6 号。因本案于 2016 年 3 月 7 日被羁押，同月 8 日被刑事拘\n留，同月 22 日被取保候审。\n辩护人吴*甲，**市东元（广州）律师事务所律师。\n**市**区人民检察院以*海检诉刑诉[2016]1193 号起\n诉书指控被告人刘*甲于 2016 年 3 月 7 日 15 时许，在其经营的\n本市**区**村东庆大街 22 号的正豆美食店门口，\n因招牌摆放\n问题与被害人简*甲发生争执和打斗。期间，被告人刘*甲用拳\n头殴打被害人简*甲的脸部致其牙齿受伤（经鉴定，损伤属轻伤\n二级）\n，后被告人刘*甲在现场被民警口头传唤接受调查。于 2016\n年 9 月 21 日向本院以被告人刘*甲犯故意伤害罪提起公诉，\n并附\n随案移送的证据予以证实。本院依法适用速裁程序，实行独任审\n判，公开开庭审理了本案。**市**区人民检察院指派代理检\n察员李婷*甲出庭支持公诉，被告人刘*甲及其辩护人吴*甲到庭\n－－ 1\n参加了诉讼，在开庭审理过程中亦无异议，现已审理终结。

 **省**市**区人民法院\n刑事判决书\n（2017）* 010* 刑初 3131* 号\n公诉机关**市**区人民检察院。\n被告人谢 xx，男，1980 年 1 月 27 日出生，汉族，**省佛山市\n人，文化程度高中，户籍地**省**市**区，现住址**省**市**区。因本案于 2017 年 5 月 4 日被羁押，同日被刑事拘留，同\n年 6 月 2 日被取保候审。\n**市**区人民检察院以*海检诉刑诉[2017]1435 号起诉书\n指控被告人谢 xx 犯故意伤害罪，于 2017 年 10 月 25 日向本院提起公\n诉，本院依法适用速裁程序，实行独任审判，公开开庭审理了本案。\n**市**区人民检察院指派检察员陈*甲出庭支持公诉，被告人谢\nxx 到庭参加了诉讼。现已审理终结。\n**市**区人民检察院指控：被告人谢 xx 伙同同案人谭 xx、\n黄 xx 于 2012 年 4 月 20 日凌晨 3 时许，在本市**区南田路三千里\n大排档门口，因琐事与被害人某*甲、某*乙发生争执，继而相互打斗。\n期间，被告人谢 xx 及同案人谭 xx、黄 xx 采取拳打脚踢的方式，致被\n害人某*甲、某*乙受伤（经法医鉴定，某*甲的损伤属轻伤、某*乙损伤属\n轻微伤）\n。\n－1－\n2017 年 5 月 4 日，被告人向公安机关主动投案。到案后，被告人\n谢 xx 自愿如实供述自己的罪行。\n本院查明事实与上述指控事实一致。\n另查明，案发后，被告人谢 xx 一次性赔偿了被害人某*甲的经济\n损失人民币三万元，被害人某*乙的经济损失人民币一万元，并取得二\n被害人的谅解。\n上述事实，有随案移送的证据予以证实，被告人在开庭审理过程\n中亦无异议，足以认定。\n案件审理中，被告人谢 xx 表示自愿认罪认罚，并提交了认罪认\n罚具结书。

现在请分析以下新案件：

案件详情:
{case['case_detail']}

涉案人员:
{case['person']}

请仔细分析案情，并严格遵守以下量刑规则返回判决结果。你的回答必须是一个JSON格式，包含两个字段：
1. case_judgment：文本描述的判决结果
2. case_judgment_label：每个涉案人员的量刑标签数组，每个标签包含以下字段：
   - predicted_sentence1: 管制，包含value（月数）
   - predicted_sentence2: 拘役，包含value（月数）
   - predicted_sentence3: 有期徒刑，包含value（月数）
   - predicted_sentence4: 缓刑，包含value（月数）
   - predicted_sentence5: 无期徒刑，包含value（true/false）
   - predicted_sentence6: 死刑，包含value（true/false）

请严格按照以下量刑规则进行预测：

1. 主刑选择规则（只能选择一个）：
   a) 拘役（1-6个月）适用情形：
      - 情节较轻
      - 社会危害性小
      - 可以判处缓刑
   b) 有期徒刑适用情形：
      - 情节较重
      - 造成严重后果
      - 主观恶性较大
   c) 其他主刑类型：
      - 管制：限制人身自由较轻
      - 无期徒刑：情节特别严重
      - 死刑：后果特别严重且罪行极其严重

2. 量刑标准：
   a) 故意伤害罪基准刑期：
      - 轻伤二级：6-18个月
      - 轻伤一级：12-36个月
      - 重伤：36-120个月
   b) 从重情节每项加重10%-30%：
      - 犯罪前科
      - 主观恶性大
      - 手段残忍
      - 后果严重
   c) 从轻情节每项减轻10%-30%：
      - 自首
      - 认罪认罚
      - 赔偿谅解
      - 初犯偶犯

3. 缓刑适用条件（必须同时满足多个）：
   - 适用范围：仅拘役或有期徒刑可以缓刑
   - 被告人认罪悔罪态度好
   - 具有从轻处罚情节（自首、立功、初犯、偶犯等）
   - 赔偿被害人损失并取得谅解
   - 犯罪情节较轻
   - 社会危害性较小

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
                {"role": "system", "content": "你是一个专业的法律助手，精通中国刑法和量刑标准。在预测判决结果时，请注意：\n1. 仔细分析案件事实，包括犯罪行为、后果、情节等\n2. 准确认定从重、从轻、减轻情节\n3. 严格遵循法定刑和量刑标准\n4. 特别关注缓刑适用条件\n5. 保持量刑的均衡性和可预测性\n6. 必须按照要求的JSON格式返回结果，包含case_judgment和case_judgment_label两个字段\n7. case_judgment_label中的每个量刑标签必须包含所有required字段，数值型字段使用0表示不适用，布尔型字段使用false表示不适用"},
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

def process_single_case(case, n_shot=N_SHOT):
    """处理单个案件的函数"""
    try:
        prediction = predict_case(case, n_shot)
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

