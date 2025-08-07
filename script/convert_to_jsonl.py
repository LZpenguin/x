import json
import pandas as pd  # 新增

def convert_file(input_file, output_file, collect_for_excel=False):
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    prompts = []
    completions = []

    # 创建输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            user_content = f"案件详情：{item['case_detail']}\n涉案人员：{item['person']}"
            assistant_content = json.dumps(item['case_judgment_label'], ensure_ascii=False)
            # 创建消息列表
            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
            # 写入JSONL格式
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + '\n')

            if collect_for_excel:
                prompts.append(user_content)
                completions.append(assistant_content)
    
    if collect_for_excel:
        df = pd.DataFrame({'Prompt': prompts, 'Completion': completions})
        df.to_excel('../data/val.xlsx', index=False)

def main():
    # 转换训练集
    convert_file(
        '../data/train.json',
        '../data/train.jsonl'
    )
    
    # 转换验证集，并生成Excel
    convert_file(
        '../data/val.json',
        '../data/val.jsonl',
        collect_for_excel=True
    )

if __name__ == '__main__':
    main()