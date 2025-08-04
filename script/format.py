import json
import os

with open('../output/result_v31_n8.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

data_f = []
for item in data:
    case_judgment_predict = []
    for person in item['case_judgment_label']:
        person_predict = {
            'person_name': person['person_name'],
            'predicted_sentence1': {'value': float(person['predicted_sentence1']['value'])},
            'predicted_sentence2': {'value': float(person['predicted_sentence2']['value'])},
            'predicted_sentence3': {'value': float(person['predicted_sentence3']['value'])},
            'predicted_sentence4': {'value': float(person['predicted_sentence4']['value'])},
            'predicted_sentence5': {'value': bool(person['predicted_sentence5']['value'])},
            'predicted_sentence6': {'value': bool(person['predicted_sentence6']['value'])}
        }
        case_judgment_predict.append(person_predict)
    
    data_f.append({
        'case_id': item['case_id'],
        'case_judgment_predict': case_judgment_predict
    })

test_path = '../data/IntentionalInjury_cases_test.json'
if os.path.exists(test_path):
    with open(test_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    test_case_ids = [item['case_id'] for item in test_cases]
    # 建立case_id到结果的映射
    data_f_dict = {item['case_id']: item for item in data_f}
    # 按test_case_ids顺序排列
    sorted_data_f = [data_f_dict[case_id] for case_id in test_case_ids if case_id in data_f_dict]
    # 输出新文件
    with open('../output/h02.json', 'w', encoding='utf-8') as f:
        json.dump(sorted_data_f, f, ensure_ascii=False, indent=2)