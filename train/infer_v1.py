from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
from dataset import XingDataset
from tqdm import tqdm
import json

# val_data_path = '../data/val_b.json'
val_data_path = '../data/IntentionalInjury_cases_test.json'
bs = 1
output_path = f"../output/qwen3-8b-ft-valb-1562{'_val' if 'val' in val_data_path else ''}.json"

model = AutoModelForCausalLM.from_pretrained("/workspace/cache/test-data-train/x/models/Qwen3-8B-ft-full-2048左截断-valb/checkpoint-1562").to('cuda:0')
tokenizer = AutoTokenizer.from_pretrained("../../models/Qwen3-8B", padding_side="left", truncation_side="left")

ps = XingDataset(val_data_path).data
with open(val_data_path, 'r') as f:
    data = json.load(f)

for i in tqdm(range(0, len(ps), bs)):
    j = min(i+bs, len(ps))
    prompts = []
    for k in range(i, j):
        prompts.append(ps[k]['prompt'].split('<|im_start|>')[0]+'<|im_start|>')
    model_inputs = tokenizer(prompts, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=1024)
    for k in range(0, j-i):
        output_id = generated_ids[k][len(model_inputs.input_ids[k]):].tolist()
        output = tokenizer.decode(output_id, skip_special_tokens=True).strip()
        try:
            parsed = json.loads(output.replace("'", '"').replace('False', 'false').replace('True', 'true'))
        except:
            parsed = []

        data[i+k]['case_judgment_label'] = parsed

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
