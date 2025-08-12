from datasets import load_dataset, Dataset, concatenate_datasets, Features, Value
import pandas as pd
from typing import Dict, List, Optional
from tqdm import tqdm
import os
import hashlib
import json

class XingDataset:
    def __init__(self, data_path):
        self.data = self.load_data(data_path)
    
    def process_json_data(self, example: Dict) -> Dict:
        return {
            'prompt': f"你是一个法律助手，请根据以下案件信息预量刑标签(只有拘役和有期徒刑可以有缓刑)\n案件详情:\n{example['case_detail']}\n判决人员:\n{example['person']}\n定罪:\n<|im_start|>{example['case_judgment_label'] if 'case_judgment_label' in example else ''}<|im_end|>"
        }

    def load_json_data(self, data_path: str) -> Dataset:
        """
        加载预训练数据集
        Args:
            data_path: 数据文件路径
        Returns:
            加载的数据集
        """
        with open(data_path, 'r') as f:
            raw_data = json.load(f)

        data_sp = []
        for idx,item in enumerate(raw_data):
            if 'case_judgment_label' not in item:
                item['case_judgment_label'] = [{}]*len(item['person'])
            for person, label in zip(item['person'], item['case_judgment_label']):
                zui = ''
                if 'predicted_sentence4' in label:
                    for i in [1,2,3,5,6]:
                        zuii = label[f'predicted_sentence{i}']
                        if zuii['value']:
                            zui = zuii['desc'].split('（')[0]
                    if label[f'predicted_sentence4']['value']:
                        zui += ' 缓刑'
                data_sp.append({
                    'case_detail': item['case_detail'],
                    'person': person,
                    'case_judgment_label': zui,
                    'idx': idx
                })

        dataset = Dataset.from_dict({key: [d[key] for d in data_sp] for key in data_sp[0]})

        processed = dataset.map(
            self.process_json_data,
            desc=f"处理预训练数据",
        ).select_columns(['prompt', 'idx', 'person'])
            
        result = Dataset.from_list(processed)
        
        return result

    def load_data(self, data_path: str) -> Dataset:
        data = None
        if data_path.endswith('.json'):
            data = self.load_json_data(data_path)
        return data

if __name__ == '__main__':
    ds = XingDataset('../data/val.json')
    print(ds.data[0])