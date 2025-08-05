import os
# 移除单卡设置，让accelerate管理GPU
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
from datasets import Dataset, concatenate_datasets
import pandas as pd
from peft import get_peft_model, LoraConfig, TaskType
import warnings
from tqdm import tqdm
import logging
import wandb
import numpy as np
import random
import hashlib
from dataset import XingDataset
from accelerate import Accelerator, DistributedDataParallelKwargs


# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 过滤警告
warnings.filterwarnings('ignore')

# 设置随机种子函数
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

# 配置参数
class TrainingConfig:
    # 获取当前工作目录的上级目录作为基础路径
    base_dir = os.path.join(os.getcwd(), "..")
    
    model_path = os.path.join(base_dir, "../models/Qwen3-4B")
    train_files = [
        os.path.join(base_dir, "data/train.json")
    ]
    val_file = os.path.join(base_dir, "data/val.json")

    
    # LoRA配置
    use_lora = False
    lora_r = 32
    lora_alpha = 64
    lora_dropout = 0.1
    
    # 根据训练模式设置输出目录
    training_method = "ft"
    training_mode = "lora" if use_lora else "full"
    output_dir = os.path.join(base_dir, f"models/{training_method}/{model_path.split('/')[-1]}-{training_method}-{training_mode}-2048左截断")
    
    # accelerate配置文件路径
    accelerate_config_file = "accelerate_config.yaml"
    
    
    # 训练配置
    batch_size = 1
    learning_rate = 1e-4
    num_epochs = 3
    max_length = 2048
    gradient_accumulation_steps = 1
    save_total_limit = 100
    eval_steps_ratio = 0.05
    save_only_model = True  # 只保存模型权重，不保存优化器状态等中间参数
    
    # 评估配置
    force_clear_cache = True
    
    # wandb配置
    wandb_project = f"xing"
    wandb_name = f"xing-{training_method}-{training_mode}"
    
    # 添加随机种子
    seed = 42

def get_processed_data_cache_path(config: 'TrainingConfig') -> tuple:
    """
    获取已处理数据的缓存路径
    Args:
        config: 训练配置
    Returns:
        (train_cache_path, val_cache_path)
    """
    train_files_str = "_".join(sorted(config.train_files))  # 排序确保一致性
    train_files_hash = hashlib.md5(train_files_str.encode()).hexdigest()
    val_file_hash = hashlib.md5(config.val_file.encode()).hexdigest()
    cache_id = f"{config.model_path}_{config.max_length}_{config.seed}_{train_files_hash}_{val_file_hash}_processed"
    cache_name = hashlib.md5(cache_id.encode()).hexdigest()
    
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets/processed_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    train_cache = os.path.join(cache_dir, f"{cache_name}_train")
    val_cache = os.path.join(cache_dir, f"{cache_name}_dev")
    
    return train_cache, val_cache

def get_tokenization_cache_path(dataset_name: str, tokenizer_name: str, config: 'TrainingConfig') -> str:
    """
    获取tokenization缓存文件路径，参考dataset.py的缓存方式
    Args:
        dataset_name: 数据集名称 (train/dev)
        tokenizer_name: tokenizer名称
        config: 训练配置
    Returns:
        缓存文件路径
    """
    # 使用关键参数生成唯一的缓存标识，移除rank让所有进程共享缓存
    cache_id = f"{dataset_name}_{tokenizer_name}_{config.max_length}_{config.seed}"
    # 计算哈希值作为缓存文件名
    cache_name = hashlib.md5(cache_id.encode()).hexdigest()
    
    # 确定缓存目录
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets/tokenization_cache")
    
    # 确保缓存目录存在
    os.makedirs(cache_dir, exist_ok=True)
    
    return os.path.join(cache_dir, f"{cache_name}.arrow")

def prepare_data(config):
    """
    根据文件扩展名自动选择对应的数据加载器
    """
    train_datasets = []
    
    # 遍历所有训练文件
    for train_file in config.train_files:
        logger.info(f"加载训练数据: {train_file}")
        
        # 根据文件扩展名选择对应的数据加载器
        if train_file.endswith('.json'):
            dataset = XingDataset(train_file)
        else:
            raise ValueError(f"不支持的文件格式: {train_file}")
        
        dataset = dataset.data
        train_datasets.append(dataset)
        logger.info(f"数据集大小: {len(dataset)}")
    
    # 合并所有训练数据集
    train_dataset = concatenate_datasets(train_datasets)
    logger.info(f"合并后的训练数据集大小: {len(train_dataset)}")
    
    # 加载开发集数据
    logger.info(f"加载开发集数据: {config.val_file}")
    if config.val_file.endswith('.json'):
        val_dataset = XingDataset(config.val_file)
    else:
        raise ValueError(f"不支持的文件格式: {config.val_file}")
    
    val_dataset = val_dataset.data
    logger.info(f"评测数据大小: {len(val_dataset)}")
    
    return train_dataset, val_dataset

def prepare_model_and_tokenizer(config):
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, truncation_side='left', padding_side='left')
    
    # 统一使用bfloat16数据类型
    torch_dtype = torch.bfloat16
    
    # 加载模型，移除device_map让accelerate管理设备分配
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch_dtype,
        attn_implementation='eager'
    )
    
    # 根据开关决定是否应用LoRA
    if config.use_lora:
        # 配置LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        
        # 获取PEFT模型
        model = get_peft_model(model, peft_config)
        logger.info("使用LoRA训练模式")
    else:
        # 全量训练模式下启用梯度检查点以节省内存
        model.gradient_checkpointing_enable()
        logger.info("使用全量训练模式，已启用梯度检查点")
    
    logger.info(f"模型数据类型: {torch_dtype}")
    
    return model, tokenizer

def tokenize_function(examples, tokenizer, config):
    # 所有数据集现在都有统一的prompt字段
    prompts = examples['prompt']
    
    # 编码输入
    inputs = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=config.max_length,
        return_tensors='pt'
    )
    
    labels = inputs['input_ids'].clone()
    
    # 获取冒号的token_id
    colon_token_id = tokenizer.encode('<|im_start|>',add_special_tokens=False)[0]
    
    # 将input_ids转换为numpy数组进行处理
    input_ids_np = inputs['input_ids'].numpy()
    
    # 找到所有冒号的位置
    colon_mask = (input_ids_np == colon_token_id)
    
    # 对每个序列处理
    for i in range(len(input_ids_np)):
        # 找到当前序列中最后一个冒号的位置
        colon_positions = np.where(colon_mask[i])[0]
        if len(colon_positions) > 0:
            last_colon_pos = colon_positions[-1]
            # 将最后一个冒号之前的所有token标记为-100
            labels[i, :last_colon_pos+1] = -100
    
    inputs['labels'] = labels
    labels[labels == tokenizer.pad_token_id] = -100
    
    return inputs

def train(config):
    # 初始化accelerator
    # 统一使用bf16混合精度
    mixed_precision = "bf16"
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs]
    )
    
    # 设置全局随机种子
    set_global_seed(config.seed)
    if accelerator.is_main_process:
        logger.info(f"设置全局随机种子为: {config.seed}")
        logger.info(f"使用GPU: {accelerator.device}")
        logger.info(f"进程数: {accelerator.num_processes}")
        logger.info(f"训练模式: {'LoRA训练' if config.use_lora else '全量训练'}")
        logger.info(f"混合精度: {mixed_precision}")
        
        os.makedirs(config.output_dir, exist_ok=True)

        wandb.init(
            project=config.wandb_project,
            name=config.wandb_name,
            config={
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "epochs": config.num_epochs,
                "model": config.model_path,
                "num_gpus": accelerator.num_processes,
                "use_lora": config.use_lora,
                "training_mode": "LoRA" if config.use_lora else "Full Fine-tuning",
                "mixed_precision": mixed_precision
            }
        )
    
    # 准备数据和模型 - 只在主进程处理，避免重复
    train_cache_path, val_cache_path = get_processed_data_cache_path(config)
    
    # 如果设置了强制清除缓存，删除现有缓存
    if config.force_clear_cache and accelerator.is_main_process:
        import shutil
        if os.path.exists(train_cache_path):
            shutil.rmtree(train_cache_path)
            logger.info(f"已清除训练数据缓存: {train_cache_path}")
        if os.path.exists(val_cache_path):
            shutil.rmtree(val_cache_path)
            logger.info(f"已清除开发数据缓存: {val_cache_path}")
    
    # 等待主进程完成缓存清除
    accelerator.wait_for_everyone()
    
    # 检查是否已有处理好的数据缓存
    if os.path.exists(train_cache_path) and os.path.exists(val_cache_path):
        if accelerator.is_main_process:
            logger.info("发现已处理的数据缓存，直接加载...")
        
        # 所有进程直接加载已处理的数据
        from datasets import load_from_disk
        train_dataset = load_from_disk(train_cache_path)
        val_dataset = load_from_disk(val_cache_path)
        
        if accelerator.is_main_process:
            logger.info(f"从缓存加载训练数据: {len(train_dataset)} 样本")
            logger.info(f"从缓存加载开发数据: {len(val_dataset)} 样本")
        
        # 所有进程加载模型和tokenizer
        model, tokenizer = prepare_model_and_tokenizer(config)
        
    else:
        # 只在主进程处理数据
        if accelerator.is_main_process:
            logger.info("未发现缓存，主进程开始处理数据...")
            
            # 主进程处理原始数据
            train_dataset, val_dataset = prepare_data(config)
            model, tokenizer = prepare_model_and_tokenizer(config)
            
            logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
            logger.info("模型和tokenizer加载完成")
            
            # 主进程进行tokenization
            logger.info("主进程开始tokenization...")
            train_dataset = train_dataset.map(
                lambda x: tokenize_function(x, tokenizer, config),
                batched=True,
                remove_columns=train_dataset.column_names,
                desc="Tokenizing train dataset"
            )
            
            val_dataset = val_dataset.map(
                lambda x: tokenize_function(x, tokenizer, config),
                batched=True,
                remove_columns=val_dataset.column_names,
                desc="Tokenizing dev dataset"
            )
            
            # 保存处理好的数据
            logger.info("保存处理好的数据到缓存...")
            train_dataset.save_to_disk(train_cache_path)
            val_dataset.save_to_disk(val_cache_path)
            logger.info("数据处理完成并已缓存")
            
        else:
            # 其他进程等待主进程完成
            logger.info(f"进程 {accelerator.process_index} 等待主进程处理数据...")
            train_dataset = None
            val_dataset = None
            model, tokenizer = prepare_model_and_tokenizer(config)
    
    # 等待主进程完成数据处理
    accelerator.wait_for_everyone()
    
    # 非主进程加载处理好的数据
    if not accelerator.is_main_process:
        logger.info(f"进程 {accelerator.process_index} 加载处理好的数据...")
        from datasets import load_from_disk
        train_dataset = load_from_disk(train_cache_path)
        val_dataset = load_from_disk(val_cache_path)
    
    if accelerator.is_main_process:
        logger.info("所有进程数据准备完成")
    
    # 计算总步数和评估步数
    total_steps = (len(train_dataset) * config.num_epochs) // (config.batch_size * config.gradient_accumulation_steps * accelerator.num_processes)
    eval_steps = max(1, int(total_steps * config.eval_steps_ratio))  # 根据比例计算评估步数，至少为1
    if accelerator.is_main_process:
        logger.info(f"总训练步数: {total_steps}, 每{eval_steps}步进行一次评估 (比例: {config.eval_steps_ratio})")
        logger.info(f"每个GPU的batch size: {config.batch_size}")
        logger.info(f"总batch size: {config.batch_size * accelerator.num_processes}")
    
    # 设置训练参数，针对accelerate进行配置
    # 统一使用bf16精度设置
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        fp16=False,
        bf16=True,
        logging_dir=f"{config.output_dir}/logs",
        logging_steps=1,
        save_strategy="steps",
        save_steps=eval_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        report_to="wandb" if accelerator.is_main_process else None,  # 只在主进程报告wandb
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=config.save_total_limit,
        save_only_model=config.save_only_model,  # 只保存模型权重，不保存优化器状态等中间参数
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        save_safetensors=True,
        overwrite_output_dir=True,
        seed=config.seed,
        # accelerate相关设置
        dataloader_pin_memory=False,  # 在多GPU环境下可能需要设置为False
        ddp_find_unused_parameters=False,
        # 全量训练时的额外设置
        gradient_checkpointing=not config.use_lora,  # 全量训练启用梯度检查点节省内存
        max_grad_norm=1.0,  # 梯度裁剪
    )
    
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # 检查是否存在完整的检查点
    last_checkpoint = None
    if os.path.exists(config.output_dir):
        checkpoints = [f for f in os.listdir(config.output_dir) if f.startswith("checkpoint")]
        if len(checkpoints) > 0:
            # 按照检查点编号排序
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
            
            # 验证检查点是否完整
            for checkpoint in reversed(checkpoints):  # 从最新的开始检查
                checkpoint_path = os.path.join(config.output_dir, checkpoint)
                trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
                
                # 检查关键文件是否存在
                if os.path.exists(trainer_state_path):
                    last_checkpoint = checkpoint_path
                    if accelerator.is_main_process:
                        logger.info(f"发现完整的检查点: {last_checkpoint}")
                    break
                else:
                    if accelerator.is_main_process:
                        logger.warning(f"检查点 {checkpoint_path} 不完整，跳过")
                        
            if last_checkpoint is None and accelerator.is_main_process:
                logger.info("未找到完整的检查点，将从头开始训练")
    
    # 开始训练，如果存在检查点则从检查点继续训练
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # 只在主进程保存模型
    if accelerator.is_main_process:
        trainer.save_model()
        tokenizer.save_pretrained(config.output_dir)
        logger.info(f"模型保存至: {config.output_dir}")
        # 关闭wandb
        wandb.finish()
    
    # 等待所有进程完成
    accelerator.wait_for_everyone()

if __name__ == "__main__":
    config = TrainingConfig()
    train(config) 