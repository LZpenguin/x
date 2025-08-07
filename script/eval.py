import json
import numpy as np


def evaluate_single_person(true_label, pred_label, max_deviation=42, main_weight=0.5, case_id=None, person_name=None, stats=None, detail=""):
    """
    评估单个被告人的预测结果
    
    Args:
        true_label: 真实标签
        pred_label: 预测标签
        max_deviation: 最大允许偏差
        main_weight: 主刑权重
        case_id: 案件ID（用于统计）
        person_name: 被告人姓名（用于统计）
        stats: 统计信息字典
        
    Returns:
        float: 总偏差值
    """
    # 1. 检查互斥错误
    # 检查多种主刑同时生效
    main_sentences = [
        pred_label["predicted_sentence1"]["value"],  # 管制
        pred_label["predicted_sentence2"]["value"],  # 拘役
        pred_label["predicted_sentence3"]["value"],  # 有期徒刑
        pred_label["predicted_sentence5"]["value"],  # 无期徒刑
        pred_label["predicted_sentence6"]["value"]   # 死刑
    ]
    
    active_sentences = sum([1 for x in main_sentences[:3] if x > 0]) + sum([1 for x in main_sentences[3:] if x])
    if active_sentences > 1 and stats is not None:
        stats["multiple_main_sentences"].append((case_id, person_name, active_sentences))
        return max_deviation
    
    # 检查有缓刑但无主刑
    has_probation = pred_label["predicted_sentence4"]["value"] > 0
    has_valid_sentence = pred_label["predicted_sentence2"]["value"] > 0 or pred_label["predicted_sentence3"]["value"] > 0
    if has_probation and not has_valid_sentence and stats is not None:
        stats["invalid_probation"].append((case_id, person_name, "缓刑存在但无有效主刑"))
        return max_deviation
    
    # 检查缓刑与不适用缓刑的主刑共存
    invalid_probation = (pred_label["predicted_sentence1"]["value"] > 0 or  # 管制
                        pred_label["predicted_sentence5"]["value"] or        # 无期徒刑
                        pred_label["predicted_sentence6"]["value"])          # 死刑
    if has_probation and invalid_probation and stats is not None:
        reason = "缓刑与不适用缓刑的主刑共存"
        if pred_label["predicted_sentence1"]["value"] > 0:
            reason += "（管制）"
        if pred_label["predicted_sentence5"]["value"]:
            reason += "（无期徒刑）"
        if pred_label["predicted_sentence6"]["value"]:
            reason += "（死刑）"
        stats["invalid_probation"].append((case_id, person_name, reason))
        return max_deviation
    
    # 2. 提取真实和预测的主刑类型及数值
    true_type = None
    pred_type = None
    
    # 确定真实主刑类型
    if true_label["predicted_sentence1"]["value"] > 0:
        true_type = "管制"
    elif true_label["predicted_sentence2"]["value"] > 0:
        true_type = "拘役"
    elif true_label["predicted_sentence3"]["value"] > 0:
        true_type = "有期徒刑"
    elif true_label["predicted_sentence5"]["value"]:
        true_type = "无期徒刑"
    elif true_label["predicted_sentence6"]["value"]:
        true_type = "死刑"
    
    # 确定预测主刑类型
    if pred_label["predicted_sentence1"]["value"] > 0:
        pred_type = "管制"
    elif pred_label["predicted_sentence2"]["value"] > 0:
        pred_type = "拘役"
    elif pred_label["predicted_sentence3"]["value"] > 0:
        pred_type = "有期徒刑"
    elif pred_label["predicted_sentence5"]["value"]:
        pred_type = "无期徒刑"
    elif pred_label["predicted_sentence6"]["value"]:
        pred_type = "死刑"
    
    # 3. 比较主刑类型是否一致
    if stats is not None:
        if true_type:
            stats["true_sentence_types"][true_type] += 1
        if pred_type:
            stats["pred_sentence_types"][pred_type] += 1
            
    if true_type != pred_type:
        if stats is not None:
            stats["sentence_type_errors"].append((case_id, person_name, true_type, pred_type))
        return max_deviation
    
    # 4. 计算主刑数值偏差
    main_deviation = 0
    if true_type in ["管制", "拘役", "有期徒刑"]:
        true_value = 0
        pred_value = 0
        
        if true_type == "管制":
            true_value = true_label["predicted_sentence1"]["value"]
            pred_value = pred_label["predicted_sentence1"]["value"]
        elif true_type == "拘役":
            true_value = true_label["predicted_sentence2"]["value"]
            pred_value = pred_label["predicted_sentence2"]["value"]
        elif true_type == "有期徒刑":
            true_value = true_label["predicted_sentence3"]["value"]
            pred_value = pred_label["predicted_sentence3"]["value"]

        
        main_deviation = abs(true_value - min(pred_value, 120))
        if main_deviation > 0 and stats is not None:
            stats["sentence_value_errors"].append((case_id, person_name, true_value, pred_value, main_deviation))
        main_deviation = min(main_deviation, max_deviation)
    
    # 5. 检查缓刑存在性及数值偏差
    probation_deviation = 0
    true_has_probation = true_label["predicted_sentence4"]["value"] > 0
    pred_has_probation = pred_label["predicted_sentence4"]["value"] > 0 if '三级' not in detail else 0
    
    if true_has_probation != pred_has_probation:
        if stats is not None:
            stats["probation_errors"].append((case_id, person_name, 
                true_label["predicted_sentence4"]["value"],
                pred_label["predicted_sentence4"]["value"]))
        probation_deviation = max_deviation
    elif true_has_probation and pred_has_probation:
        probation_deviation = abs(true_label["predicted_sentence4"]["value"] - pred_label["predicted_sentence4"]["value"])
        if probation_deviation > 0 and stats is not None:
            stats["probation_errors"].append((case_id, person_name,
                true_label["predicted_sentence4"]["value"],
                pred_label["predicted_sentence4"]["value"]))
            
        probation_deviation = min(probation_deviation, max_deviation)
    
    # 6. 计算总偏差
    total_deviation = main_deviation * main_weight + probation_deviation * (1 - main_weight)
    
    return total_deviation


def evaluate_predictions(true_file, pred_file, max_deviation=42, main_weight=0.5):
    """
    评估整体预测结果
    
    Args:
        true_file: 真实标签文件路径
        pred_file: 预测标签文件路径
        max_deviation: 最大允许偏差
        main_weight: 主刑权重
        
    Returns:
        float: 平均偏差值
    """
    # 统计信息初始化
    stats = {
        "true_sentence_types": {"管制": 0, "拘役": 0, "有期徒刑": 0, "无期徒刑": 0, "死刑": 0},
        "pred_sentence_types": {"管制": 0, "拘役": 0, "有期徒刑": 0, "无期徒刑": 0, "死刑": 0},
        "sentence_type_errors": [],  # [(case_id, person_name, true_type, pred_type), ...]
        "sentence_value_errors": [],  # [(case_id, person_name, true_value, pred_value, deviation), ...]
        "probation_errors": [],      # [(case_id, person_name, true_prob, pred_prob), ...]
        "multiple_main_sentences": [],  # [(case_id, person_name, active_sentences), ...]
        "invalid_probation": []      # [(case_id, person_name, reason), ...]
    }
    # 读取文件
    with open(true_file, 'r', encoding='utf-8') as f:
        true_data = json.load(f)
    with open(pred_file, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
    
    # 创建案例ID到案例的映射
    true_cases = {case["case_id"]: case for case in true_data}
    pred_cases = {case["case_id"]: case for case in pred_data}
    
    total_deviation = 0
    total_persons = 0
    missing_cases = []
    missing_persons = []
    
    # 遍历所有真实案例
    for case_id, true_case in true_cases.items():
        # 检查案例是否存在于预测中
        if case_id not in pred_cases:
            missing_cases.append(case_id)
            total_deviation += len(true_case["case_judgment_label"]) * max_deviation
            total_persons += len(true_case["case_judgment_label"])
            continue
            
        pred_case = pred_cases[case_id]
        
        # 创建人名到标签的映射
        true_labels = {label["person_name"]: label for label in true_case["case_judgment_label"]}
        pred_labels = {label["person_name"]: label for label in pred_case["case_judgment_label"]}
        detail = pred_case['case_detail']
        
        # 遍历所有真实被告人
        for person_name, true_label in true_labels.items():
            if person_name not in pred_labels:
                missing_persons.append((case_id, person_name))
                total_deviation += max_deviation
                total_persons += 1
                continue
                
            pred_label = pred_labels[person_name]
            deviation = evaluate_single_person(true_label, pred_label, max_deviation, main_weight,
                                            case_id, person_name, stats, detail)
            total_deviation += deviation
            total_persons += 1
            
        # 检查预测中是否有多余的被告人
        for person_name in pred_labels:
            if person_name not in true_labels:
                total_deviation += max_deviation
                total_persons += 1
    
    # 检查预测中是否有多余的案例
    for case_id in pred_cases:
        if case_id not in true_cases:
            pred_case = pred_cases[case_id]
            total_deviation += len(pred_case["case_judgment_label"]) * max_deviation
            total_persons += len(pred_case["case_judgment_label"])
    
    # 计算平均偏差
    average_deviation = total_deviation / total_persons if total_persons > 0 else max_deviation
    
    print(f"\n=== 主刑分布 ===")
    print("真实标签分布：")
    for type_name, count in stats["true_sentence_types"].items():
        print(f"{type_name}: {count} ({count/total_persons*100:.2f}%)")
    print("\n预测标签分布：")
    for type_name, count in stats["pred_sentence_types"].items():
        print(f"{type_name}: {count} ({count/total_persons*100:.2f}%)")
        
    print(f"\n=== 错误统计 ===")
    print(f"1. 主刑类型错误：{len(stats['sentence_type_errors'])} 个")
    if stats['sentence_type_errors']:
        print("示例（前10个）：")
        for case_id, person_name, true_type, pred_type in stats['sentence_type_errors'][:10]:
            print(f"  - 案件 {case_id}, 被告人 {person_name}: {true_type} -> {pred_type}")
            
    print(f"\n2. 主刑刑期错误：{len(stats['sentence_value_errors'])} 个")
    if stats['sentence_value_errors']:
        print("示例（偏差最大的5个）：")
        sorted_errors = sorted(stats['sentence_value_errors'], key=lambda x: x[4], reverse=True)
        for case_id, person_name, true_val, pred_val, dev in sorted_errors[:5]:
            print(f"  - 案件 {case_id}, 被告人 {person_name}: {true_val} -> {pred_val} (偏差: {dev})")
            
    print(f"\n3. 缓刑错误：{len(stats['probation_errors'])} 个")
    if stats['probation_errors']:
        print("示例（前20个）：")
        for case_id, person_name, true_prob, pred_prob in stats['probation_errors'][:20]:
            print(f"  - 案件 {case_id}, 被告人 {person_name}: {true_prob} -> {pred_prob}")
            
    print(f"\n4. 多重主刑错误：{len(stats['multiple_main_sentences'])} 个")
    if stats['multiple_main_sentences']:
        print("示例（前5个）：")
        for case_id, person_name, active in stats['multiple_main_sentences'][:5]:
            print(f"  - 案件 {case_id}, 被告人 {person_name}: {active}个主刑同时生效")
            
    print(f"\n5. 无效缓刑错误：{len(stats['invalid_probation'])} 个")
    if stats['invalid_probation']:
        print("示例（前5个）：")
        for case_id, person_name, reason in stats['invalid_probation'][:5]:
            print(f"  - 案件 {case_id}, 被告人 {person_name}: {reason}")
        
    print(f"\n=== 评估结果 ===")
    print(f"总案例数：{len(true_cases)}")
    print(f"总被告人数：{total_persons}")
    print(f"缺失案例数：{len(missing_cases)}")
    print(f"缺失被告人数：{len(missing_persons)}")
    print(f"平均偏差：{average_deviation:.4f}")
    print(f"分数：{100 - average_deviation}")
    
    return average_deviation


# 评估验证集结果
true_file = "../data/val_b.json"
pred_file = "../output/qwen3-8b-ft-valb-1846_val.json"
average_deviation = evaluate_predictions(true_file, pred_file)
