import os
import json
import torch
import wandb
from tqdm import tqdm
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator
)
import numpy as np
from typing import List, Dict, Any
from torch.utils.data import Dataset
import logging
from dataclasses import dataclass
from typing import Optional, Union

class CourtDataset(Dataset):
    def __init__(self, cases, tokenizer, max_length=2048):
        self.cases = cases
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        case = self.cases[idx]
        prompt = self.create_prompt(case)
        
        # 编码输入
        inputs = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 准备标签
        labels = inputs["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }
    
    def create_prompt(self, case):
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
   - 管制、无期徒刑、死刑不能适用缓刑"""
        
        if "case_judgment" in case and "case_judgment_label" in case:
            # 如果有标签，添加到提示中作为示例
            prompt += f"""

判决结果：
{case['case_judgment']}

量刑标签：
{json.dumps(case['case_judgment_label'], ensure_ascii=False, indent=2)}"""
        
        return prompt

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='训练配置')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, default='../../models/Qwen3-4B',
                        help='模型路径')
    parser.add_argument('--max_length', type=int, default=2048,
                        help='最大序列长度')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='学习率')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='预热比例')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='梯度累积步数')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='梯度裁剪阈值')
    
    # 数据参数
    parser.add_argument('--train_data_path', type=str, default='../data/train.json',
                        help='训练数据路径')
    parser.add_argument('--eval_data_path', type=str, default='../data/val.json',
                        help='验证数据路径')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载的工作进程数')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='../output',
                        help='输出目录')
    parser.add_argument('--save_steps', type=int, default=100,
                        help='保存模型的步数间隔')
    parser.add_argument('--eval_steps', type=int, default=100,
                        help='评估的步数间隔')
    parser.add_argument('--logging_steps', type=int, default=10,
                        help='日志记录的步数间隔')
    
    # Wandb 参数
    parser.add_argument('--use_wandb', action='store_true',
                        help='是否使用 Weights & Biases 记录实验')
    
    args = parser.parse_args()
    return args

# 解析命令行参数
train_args = vars(parse_args())

# 常量
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_DEVIATION = 42
MAIN_WEIGHT = 0.5
OUTPUT_PATH = os.path.join(train_args["output_dir"], 'result_qwen.json')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def load_model_and_tokenizer(accelerator: Accelerator):
    """加载模型和分词器"""
    logging.info("正在加载模型和分词器...")
    
    # 设置环境变量
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    try:
        # 加载tokenizer
        config = {
            "model_name_or_path": MODEL_NAME,
            "trust_remote_code": True,
            "use_fast": True,
            "padding_side": "left"
        }
        
        tokenizer = AutoTokenizer.from_pretrained(**config)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # 加载模型
        model_config = {
            "pretrained_model_name_or_path": MODEL_NAME,
            "trust_remote_code": True,
            "torch_dtype": torch.float16
        }
        
        with accelerator.main_process_first():
            model = AutoModelForCausalLM.from_pretrained(**model_config)
        
        model = accelerator.prepare(model)
        
        return model, tokenizer
    except Exception as e:
        logging.error(f"加载模型时出错: {str(e)}")
        logging.error(f"请确保模型路径 {MODEL_NAME} 存在并包含完整的模型文件")
        raise

def validate_sentence(sentence: Dict[str, Any]) -> tuple[bool, str]:
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

def predict_case(model, tokenizer, accelerator: Accelerator, case: Dict[str, Any]) -> Dict[str, Any]:
    """使用Qwen模型预测单个案件的判决结果"""
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

请返回如下格式的JSON：
{{
    "case_judgment": "本院认为，...",
    "case_judgment_label": [[
        {{
            "person_name": "涉案人员姓名",
            "predicted_sentence1": {{"value": 0, "desc": "管制（月数）"}},
            "predicted_sentence2": {{"value": 0, "desc": "拘役（月数）"}},
            "predicted_sentence3": {{"value": 12, "desc": "有期徒刑（月数）"}},
            "predicted_sentence4": {{"value": 24, "desc": "缓刑（月数）"}},
            "predicted_sentence5": {{"value": false, "desc": "无期徒刑"}},
            "predicted_sentence6": {{"value": false, "desc": "死刑"}}
        }}
    ]]
}}"""

    try:
        # 对输入进行编码
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        inputs = {k: v for k, v in inputs.items()}

        # 生成回答
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取JSON部分
        try:
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                prediction = json.loads(json_str)
            else:
                raise ValueError("未找到有效的JSON格式回答")

            # 验证每个涉案人员的刑罚组合
            valid_sentences = []
            for sentence in prediction['case_judgment_label']:
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

            case['case_judgment'] = prediction['case_judgment']
            case['case_judgment_label'] = valid_sentences
            
        except Exception as e:
            logging.error(f"解析预测结果时出错: {str(e)}")
            case['case_judgment'] = f"Error: {str(e)}"
            case['case_judgment_label'] = []
            
    except Exception as e:
        logging.error(f"生成预测时出错: {str(e)}")
        case['case_judgment'] = f"Error: {str(e)}"
        case['case_judgment_label'] = []
    
    return case

def evaluate_single_person(true_label: Dict[str, Any], pred_label: Dict[str, Any], 
                         max_deviation: float = MAX_DEVIATION, 
                         main_weight: float = MAIN_WEIGHT,
                         case_id: str = None, 
                         person_name: str = None, 
                         stats: Dict[str, Any] = None,
                         detail: str = "") -> float:
    """评估单个被告人的预测结果"""
    # 1. 检查互斥错误
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
    
    # 2. 检查缓刑合法性
    has_probation = pred_label["predicted_sentence4"]["value"] > 0
    has_valid_sentence = (pred_label["predicted_sentence2"]["value"] > 0 or 
                         pred_label["predicted_sentence3"]["value"] > 0)
    
    if has_probation and not has_valid_sentence and stats is not None:
        stats["invalid_probation"].append((case_id, person_name, "缓刑存在但无有效主刑"))
        return max_deviation
    
    invalid_probation = (pred_label["predicted_sentence1"]["value"] > 0 or  # 管制
                        pred_label["predicted_sentence5"]["value"] or        # 无期徒刑
                        pred_label["predicted_sentence6"]["value"])          # 死刑
    if has_probation and invalid_probation and stats is not None:
        reason = "缓刑与不适用缓刑的主刑共存"
        stats["invalid_probation"].append((case_id, person_name, reason))
        return max_deviation
    
    # 3. 确定主刑类型
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
    
    # 4. 统计和比较主刑类型
    if stats is not None:
        if true_type:
            stats["true_sentence_types"][true_type] += 1
        if pred_type:
            stats["pred_sentence_types"][pred_type] += 1
            
    if true_type != pred_type:
        if stats is not None:
            stats["sentence_type_errors"].append((case_id, person_name, true_type, pred_type))
        return max_deviation
    
    # 5. 计算主刑数值偏差
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
    
    # 6. 检查缓刑
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
    
    # 7. 计算总偏差
    total_deviation = main_deviation * main_weight + probation_deviation * (1 - main_weight)
    return total_deviation

def evaluate_predictions(true_data: List[Dict[str, Any]], pred_data: List[Dict[str, Any]],
                       max_deviation: float = MAX_DEVIATION,
                       main_weight: float = MAIN_WEIGHT) -> tuple[float, float]:
    """评估整体预测结果，返回(score, loss)"""
    # 初始化统计信息
    stats = {
        "true_sentence_types": {"管制": 0, "拘役": 0, "有期徒刑": 0, "无期徒刑": 0, "死刑": 0},
        "pred_sentence_types": {"管制": 0, "拘役": 0, "有期徒刑": 0, "无期徒刑": 0, "死刑": 0},
        "sentence_type_errors": [],
        "sentence_value_errors": [],
        "probation_errors": [],
        "multiple_main_sentences": [],
        "invalid_probation": []
    }
    
    # 创建案例ID到案例的映射
    true_cases = {case["case_id"]: case for case in true_data}
    pred_cases = {case["case_id"]: case for case in pred_data}
    
    total_deviation = 0
    total_persons = 0
    missing_cases = []
    missing_persons = []
    
    # 用于计算MSE loss的真实值和预测值列表
    true_values = []
    pred_values = []
    
    # 遍历所有真实案例
    for case_id, true_case in true_cases.items():
        if case_id not in pred_cases:
            missing_cases.append(case_id)
            total_deviation += len(true_case["case_judgment_label"]) * max_deviation
            total_persons += len(true_case["case_judgment_label"])
            continue
            
        pred_case = pred_cases[case_id]
        
        true_labels = {label["person_name"]: label for label in true_case["case_judgment_label"]}
        pred_labels = {label["person_name"]: label for label in pred_case["case_judgment_label"]}
        detail = pred_case['case_detail']
        
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
            
            # 收集用于计算MSE的值
            for i in range(1, 7):
                key = f"predicted_sentence{i}"
                true_val = float(true_label[key]["value"]) if isinstance(true_label[key]["value"], (int, float)) else float(true_label[key]["value"] is True)
                pred_val = float(pred_label[key]["value"]) if isinstance(pred_label[key]["value"], (int, float)) else float(pred_label[key]["value"] is True)
                true_values.append(true_val)
                pred_values.append(pred_val)
            
        # 检查多余的被告人
        for person_name in pred_labels:
            if person_name not in true_labels:
                total_deviation += max_deviation
                total_persons += 1
    
    # 检查多余的案例
    for case_id in pred_cases:
        if case_id not in true_cases:
            pred_case = pred_cases[case_id]
            total_deviation += len(pred_case["case_judgment_label"]) * max_deviation
            total_persons += len(pred_case["case_judgment_label"])
    
    # 计算平均偏差、分数和MSE loss
    average_deviation = total_deviation / total_persons if total_persons > 0 else max_deviation
    score = 100 - average_deviation
    mse_loss = mean_squared_error(true_values, pred_values) if true_values else float('inf')
    
    # 打印评估结果
    logging.info("\n=== 评估结果 ===")
    logging.info(f"总案例数：{len(true_cases)}")
    logging.info(f"总被告人数：{total_persons}")
    logging.info(f"缺失案例数：{len(missing_cases)}")
    logging.info(f"缺失被告人数：{len(missing_persons)}")
    logging.info(f"平均偏差：{average_deviation:.4f}")
    logging.info(f"分数：{score:.2f}")
    logging.info(f"MSE Loss：{mse_loss:.4f}")
    
    return score, mse_loss

@dataclass
class CustomTrainer(Trainer):
    def __init__(self, eval_cases=None, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.eval_cases = eval_cases
        self.tokenizer = tokenizer
        
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        # 首先运行标准的评估循环获取 loss
        eval_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )
        
        # 如果没有评估案例，直接返回
        if not self.eval_cases:
            return eval_output
            
        # 进行案例预测和评分
        self.model.eval()
        results = []
        for case in tqdm(self.eval_cases, desc="评估案例"):
            processed_case = predict_case(self.model, self.tokenizer, case)
            results.append(processed_case)
        
        # 计算评分
        score, _ = evaluate_predictions(self.eval_cases, results)
        
        # 记录到 wandb
        if self.is_world_process_zero():
            wandb.log({
                f"{metric_key_prefix}_loss": eval_output.metrics[f"{metric_key_prefix}_loss"],
                f"{metric_key_prefix}_score": score,
            })
        
        # 更新评估输出的指标
        eval_output.metrics[f"{metric_key_prefix}_score"] = score
        
        return eval_output

def main():
    """主函数"""
    # 初始化 wandb
    if train_args["use_wandb"]:
        wandb.init(project="court-judgment-prediction")
    
    # 加载模型和分词器
    model = AutoModelForCausalLM.from_pretrained(
        train_args["model_name"],
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        train_args["model_name"],
        trust_remote_code=True,
        use_fast=True,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 读取数据
    logging.info("正在读取训练数据...")
    with open(train_args["train_data_path"], 'r', encoding='utf-8') as f:
        train_cases = json.load(f)
    
    logging.info("正在读取验证数据...")
    with open(train_args["eval_data_path"], 'r', encoding='utf-8') as f:
        eval_cases = json.load(f)
    
    # 准备数据集
    train_dataset = CourtDataset(train_cases, tokenizer, max_length=train_args["max_length"])
    eval_dataset = CourtDataset(eval_cases, tokenizer, max_length=train_args["max_length"])
    
    # 准备训练参数
    training_args = TrainingArguments(
        output_dir=train_args["output_dir"],
        num_train_epochs=train_args["num_epochs"],
        per_device_train_batch_size=train_args["batch_size"],
        per_device_eval_batch_size=train_args["batch_size"],
        gradient_accumulation_steps=train_args["gradient_accumulation_steps"],
        learning_rate=train_args["learning_rate"],
        warmup_ratio=train_args["warmup_ratio"],
        max_grad_norm=train_args["max_grad_norm"],
        logging_steps=train_args["logging_steps"],
        evaluation_strategy="steps",
        eval_steps=train_args["eval_steps"],
        save_strategy="steps",
        save_steps=train_args["save_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_score",
        greater_is_better=True,
        fp16=True,
        report_to="wandb" if train_args["use_wandb"] else "none",
    )
    
    # 初始化 trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        eval_cases=eval_cases,
        tokenizer=tokenizer,
    )
    
    for case in tqdm(eval_cases, desc="处理验证集", disable=not accelerator.is_main_process):
        processed_case = predict_case(model, tokenizer, accelerator, case)
        results.append(processed_case)
    
    # 收集所有进程的结果
    all_results = accelerator.gather_for_metrics(results)
    
    if accelerator.is_main_process:
        # 合并结果
        final_results = []
        for process_results in zip(*all_results):
            final_results.extend(process_results)
        
        # 保存预测结果
        logging.info("正在保存预测结果...")
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        # 评估结果
        score, loss = evaluate_predictions(eval_cases, final_results)
        logging.info(f"\n最终评估结果：")
        logging.info(f"分数：{score:.2f}")
        logging.info(f"Loss：{loss:.4f}")

        # 记录到 wandb
        wandb.log({
            "score": score,
            "mse_loss": loss,
        })
        
        # 关闭 wandb
        wandb.finish()
    
    accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()