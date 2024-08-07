import argparse
from loguru import logger
import os
from os.path import join
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
from component.collator import PretrainCollator, SFTDataCollator, CompDataCollator
from component.argument import CustomizedArguments
from component.template import template_dict
from component.dataset import (
    UnifiedSFTDataset,
    ChatGLM2SFTDataset,
    ChatGLM3SFTDataset,
    UnifiedDPODataset,
    CompDataset
)
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    Trainer,
    AddedToken
)
from datasets import load_dataset, concatenate_datasets
import datasets
from itertools import chain
from tqdm import tqdm
import json
from trl import DPOTrainer #, get_kbit_device_map
import torch.nn as nn



def setup_everything():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--train_args_file", type=str, default='train_args/pretrain/full/bloom-1b1-pretrain-full.json', help="")
    parser.add_argument("--train_args_file", type=str, default='train_args/sft/qlora/qwen-7b-sft-qlora.json', help="")
    parser.add_argument("--local_rank", type=int, help="")
    args = parser.parse_args()
    train_args_file = args.train_args_file
    # 读取训练的参数配置
    parser = HfArgumentParser((CustomizedArguments, TrainingArguments))
    # 解析得到自定义参数，以及自带参数
    args, training_args = parser.parse_json_file(json_file=train_args_file)
    # 创建输出目录
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    logger.add(join(training_args.output_dir, 'train.log'))
    logger.info("train_args:{}".format(training_args))
    # 加载训练配置文件
    with open(train_args_file, "r") as f:
        train_args = json.load(f)
    # 保存训练参数到输出目录
    with open(join(training_args.output_dir, 'train_args.json'), "w") as f:
        json.dump(train_args, f, indent=4)
    # 设置随机种子
    set_seed(training_args.seed)

    # check some setting
    assert args.task_type in ['pretrain', 'sft', 'dpo', 'comp'], "task_type should be in ['pretrain', 'sft', 'dpo', 'comp']"
    assert args.train_mode in ['full', 'lora', 'qlora'], "task_type should be in ['full', 'lora', 'qlora']"
    assert sum([training_args.fp16, training_args.bf16]) == 1, "only one of fp16 and bf16 can be True"

    return args, training_args


def find_all_linear_names(model, train_mode):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    assert train_mode in ['lora', 'qlora']
    cls = bnb.nn.Linear4bit if train_mode == 'qlora' else nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    lora_module_names = list(lora_module_names)
    logger.info(f'LoRA target module names: {lora_module_names}')
    return lora_module_names


def load_pretrain_dataset(training_args, args, tokenizer):
    """
    多线程预处理预训练数据
    """
    def tokenize_function(examples):
        output = tokenizer(examples["text"])
        output = {'input_ids': output.input_ids}
        return output

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    data_path = args.train_file
    max_seq_length = args.max_seq_length
    # 创建缓存路径
    cache_dir = join(data_path, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    logger.info('Pretraining data path: {}'.format(data_path))

    # 扫描所有jsonl文件
    logger.info('Scanning all the training file...')
    files = []
    for root, dir_names, file_names in os.walk(data_path):
        for file_name in file_names:
            file = join(root, file_name)
            if file_name.endswith('.jsonl'):
                files.append(file)
    logger.info(f'Total num of training file: {len(files)}')

    # 预处理所有文本，将其id化，并且进行packing操作
    with training_args.main_process_first(desc="dataset map tokenization and grouping"):
        pretrain_dataset = []  # 汇总所有dataset
        for idx, file in enumerate(tqdm(files)):
            logger.info(f'Loading file: {file}')
            file_name = os.path.basename(file)
            file_name = file_name.replace('.jsonl', '')
            cache_path = os.path.join(cache_dir, file_name)
            os.makedirs(cache_path, exist_ok=True)

            try:
                processed_dataset = datasets.load_from_disk(cache_path, keep_in_memory=False)
                logger.info(f'Finished loading datasets-{file_name} from cache')
            except Exception:
                tmp_cache_path = join(cache_path, 'tmp')    # 临时缓存目录，会被自动删除
                logger.info(f'There is no cache of file {file_name}, start preprocessing...')
                raw_dataset = load_dataset("json", data_files=file, cache_dir=tmp_cache_path, keep_in_memory=False)
                tokenized_dataset = raw_dataset.map(
                    tokenize_function,
                    batched=True,
                    num_proc=args.tokenize_num_workers,
                    remove_columns="text",
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names={k: os.path.join(tmp_cache_path, 'tokenized.arrow') for k in raw_dataset},
                    desc="Running tokenizer on dataset",
                )
                grouped_datasets = tokenized_dataset.map(
                    group_texts,
                    batched=True,
                    num_proc=args.tokenize_num_workers,
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names={k: os.path.join(tmp_cache_path, 'grouped.arrow') for k in tokenized_dataset},
                    desc=f"Grouping texts in chunks of {max_seq_length}",
                )
                processed_dataset = grouped_datasets
                processed_dataset.save_to_disk(cache_path)
                # 删除临时目录
                # shutil.rmtree(tmp_cache_path)

            logger.info(f"Training number of {file_name}: {len(processed_dataset['train'])}")
            if idx == 0:
                pretrain_dataset = processed_dataset['train']
            else:
                assert pretrain_dataset.features.type == processed_dataset["train"].features.type
                pretrain_dataset = concatenate_datasets([pretrain_dataset, processed_dataset["train"]])
    logger.info(f"Total training number: {len(pretrain_dataset)}")
    return pretrain_dataset


def load_tokenizer(args):
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # 加载tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if config.model_type == 'llama' or config.model_type == 'internlm2' else True
    )

    # 部分模型的base与chat版本的tokenizer存在差异
    if 'internlm2' in args.model_name_or_path.lower():
        tokenizer._added_tokens_encoder.update({'<|im_start|>': 92543})
        tokenizer._added_tokens_encoder.update({'<|im_end|>': 92542})
        tokenizer._added_tokens_decoder.update({92543: AddedToken('<|im_start|>')})
        tokenizer._added_tokens_decoder.update({92542: AddedToken('<|im_end|>')})
        tokenizer.add_special_tokens({'additional_special_tokens': ['<|im_start|>', '<|im_end|>']})
    elif 'orion' in args.model_name_or_path.lower():
        tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>'})
    elif 'gemma' in args.model_name_or_path.lower():
        tokenizer.add_special_tokens({'additional_special_tokens': ['<start_of_turn>', '<end_of_turn>']})

    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    assert tokenizer.eos_token_id is not None, "eos_token_id should not be None"
    logger.info(f'vocab_size of tokenizer: {tokenizer.vocab_size}')
    return tokenizer


def load_model(args, training_args):
    """
    加载模型
    """
    assert training_args.bf16 or training_args.fp16, 'bf16 or fp16 should be True'
    logger.info(f'Loading model from base model: {args.model_name_or_path}')
    logger.info(f'Train model with {args.train_mode}')

    # init model kwargs
    # todo add flash attention
    # attn_implementation = None
    torch_dtype = torch.float16 if training_args.fp16 else torch.bfloat16
    if args.train_mode == 'qlora':
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16 if training_args.fp16 else torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    else:
        quantization_config = None
    
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device_map = {'': local_rank}
    
    model_kwargs = dict(
        trust_remote_code=True,
        # attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=device_map, #get_kbit_device_map() if quantization_config is not None else None
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)

    # moe模型，需要考虑负载均衡的loss
    if 'output_router_logits' in model.config.to_dict():
        logger.info('set output_router_logits as True')
        model.config.output_router_logits = True
    # QLoRA: casts all the non int8 modules to full precision (fp32) for stability
    if args.train_mode == 'qlora' and args.task_type in ['pretrain', 'sft', 'comp']:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    # LoRA: Enables the gradients for the input embeddings
    if args.train_mode == 'lora' and args.task_type in ['pretrain', 'sft', 'comp']:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # init peft_config
    if args.train_mode == 'full':
        peft_config = None
    else:
        # 找到所有需要插入adapter的全连接层
        target_modules = find_all_linear_names(model, args.train_mode)
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

    # init peft model
    if args.train_mode in ['lora', 'qlora'] and args.task_type in ['pretrain', 'sft', 'comp']:
        model = get_peft_model(model, peft_config)
        logger.info(f'memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB')
        model.print_trainable_parameters()

    # init ref_model
    if args.task_type == 'dpo':
        ref_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs) if args.train_mode == 'full' else None
    # pretrain和sft，不需要ref_model
    else:
        ref_model = None

    # 计算模型参数量
    total = sum(p.numel() for p in model.parameters())
    logger.info("Total model params: %.2fM" % (total / 1e6))

    return {
        'model': model,
        'ref_model': ref_model,
        'peft_config': peft_config
    }


def load_sft_dataset(args, tokenizer):
    if args.template_name not in template_dict.keys():
        raise Exception(f"template_name doesn't exist, all template_name: {template_dict.keys()}")
    template = template_dict[args.template_name]
    if 'chatglm2' in args.model_name_or_path.lower():
        logger.info('Loading data with ChatGLM2SFTDataset')
        train_dataset = ChatGLM2SFTDataset(args.train_file, tokenizer, args.max_seq_length, template)
    elif 'chatglm3' in args.model_name_or_path.lower():
        logger.info('Loading data with ChatGLM3SFTDataset')
        train_dataset = ChatGLM3SFTDataset(args.train_file, tokenizer, args.max_seq_length, template)
    else:
        logger.info('Loading data with UnifiedSFTDataset')
        train_dataset = UnifiedSFTDataset(args.train_file, tokenizer, args.max_seq_length, template)
    return train_dataset


def load_comp_dataset(args, tokenizer):
    if args.template_name not in template_dict.keys():
        raise Exception(f"template_name doesn't exist, all template_name: {template_dict.keys()}")
    template = template_dict[args.template_name]
    train_dataset = CompDataset(args.train_file, tokenizer, args.max_seq_length, template)
    return train_dataset


def load_dpo_dataset(args, tokenizer):
    if args.template_name not in template_dict.keys():
        raise Exception(f"template_name doesn't exist, all template_name: {template_dict.keys()}")
    template = template_dict[args.template_name]
    train_dataset = UnifiedDPODataset(args.train_file, tokenizer, args.max_seq_length, args.max_prompt_length, template)
    return train_dataset


class LoRATrainer(Trainer):
    # Pairloss
    def compute_loss(self, model, inputs, return_outputs=False): 
        ignore_index=-100
        log_sigmoid = nn.LogSigmoid()
        # bce_loss = nn.BCELoss() #接受sigmoid之后的结果
        # print('inputs:', [(key,inputs[key].shape) for key in inputs], inputs)
        # 'input_ids', torch.Size([5, 2, 288])
        #([5, 288, 125696]
        outputs_pos = model(input_ids=inputs['input_ids'][:,0,:].squeeze(1), attention_mask=inputs['attention_mask'][:,0,:].squeeze(1), return_dict=True)
        # print("outputs_pos:", outputs_pos['logits'].shape, outputs_pos)
        outputs_neg = model(input_ids=inputs['input_ids'][:,1,:].squeeze(1), attention_mask=inputs['attention_mask'][:,1,:].squeeze(1), return_dict=True)
        # print("outputs_neg:", outputs_neg['logits'].shape, outputs_neg)
        
        # [5, 288, 125696])
        logits_pos = outputs_pos["logits"] if isinstance(outputs_pos, dict) else outputs_pos[0]
        logits_neg = outputs_neg["logits"] if isinstance(outputs_neg, dict) else outputs_neg[0]
        # print("logits_pos:", logits_pos.shape)
        # print("logits_neg:", logits_neg.shape)
        
        #[5, 288]
        # labels_pos = torch.where(inputs['target_mask'][:,0,:].squeeze(1) == 1, inputs['input_ids'][:,0,:].squeeze(1), ignore_index)
        # labels_neg = torch.where(inputs['target_mask'][:,1,:].squeeze(1) == 1, inputs['input_ids'][:,1,:].squeeze(1), ignore_index)
        #在datacollator中已经转换过了
        labels_pos = inputs['labels'][:,0,:].squeeze(1)
        labels_neg = inputs['labels'][:,1,:].squeeze(1)
        
        # print("labels_pos:", labels_pos.shape, labels_pos)
        
        shift_logits_pos = logits_pos[..., :-1, :].contiguous()
        shift_labels_pos = labels_pos[..., 1:].contiguous()
        # print("shift_logits_pos:", shift_logits_pos.shape, shift_logits_pos)
      
        shift_logits_neg = logits_neg[..., :-1, :].contiguous()
        shift_labels_neg = labels_neg[..., 1:].contiguous()    
        
        pos_loss_mask = shift_labels_pos != ignore_index
        neg_loss_mask = shift_labels_neg != ignore_index
        # print("pos_loss_mask:", pos_loss_mask.shape, pos_loss_mask)
        
        shift_labels_pos[shift_labels_pos == ignore_index] = 0
        shift_labels_neg[shift_labels_neg == ignore_index] = 0
        
        # gather用于按照给定的索引index从输入张量中收集指定的元素
        pos_per_token_logps = torch.gather(shift_logits_pos.softmax(-1), dim=2, index=shift_labels_pos.unsqueeze(2)).squeeze(2)
        neg_per_token_logps = torch.gather(shift_logits_neg.softmax(-1), dim=2, index=shift_labels_neg.unsqueeze(2)).squeeze(2)
        # print("pos_per_token_logps:", pos_per_token_logps.shape, pos_per_token_logps)
        
        pos_log_p = (pos_per_token_logps * pos_loss_mask).sum(-1) / pos_loss_mask.sum(-1)
        neg_log_p = (neg_per_token_logps * neg_loss_mask).sum(-1) / neg_loss_mask.sum(-1)
        # print("pos_log_p:", pos_log_p.shape, pos_log_p)
        # print("neg_log_p:", neg_log_p.shape, neg_log_p)
      
        loss = -log_sigmoid(pos_log_p-neg_log_p)
        # print("loss:", loss)
    
        return (loss.sum(-1), outputs_pos) if return_outputs else loss.sum(-1)


def init_components(args, training_args):
    """
    初始化各个组件
    """
    training_args.ddp_find_unused_parameters = False
    logger.info('Initializing components...')

    # 加载tokenizer
    tokenizer = load_tokenizer(args)
    # 加载model
    components = load_model(args, training_args)
    model = components['model']
    ref_model = components['ref_model']
    peft_config = components['peft_config']

    # 初始化dataset和collator
    if args.task_type == 'pretrain':
        logger.info('Train model with pretrain task')
        train_dataset = load_pretrain_dataset(training_args, args, tokenizer)
        data_collator = PretrainCollator(tokenizer, args.max_seq_length)
    elif args.task_type == 'sft':
        logger.info('Train model with sft task')
        train_dataset = load_sft_dataset(args, tokenizer)
        data_collator = SFTDataCollator(tokenizer, args.max_seq_length)
    elif args.task_type == 'comp':
        train_dataset = load_comp_dataset(args, tokenizer)
        data_collator = CompDataCollator(tokenizer, args.max_seq_length)        
    else:
        logger.info('Train model with dpo task')
        train_dataset = load_dpo_dataset(args, tokenizer)
        data_collator = None
    print("train_dataset:", train_dataset.__getitem__(0))

    # dpo
    if args.task_type == 'dpo':
        trainer = DPOTrainer(
            model,
            ref_model,
            args=training_args,
            beta=args.beta,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            peft_config=peft_config
        )
    # pretrain or sft
    elif args.task_type == 'comp':
        trainer = LoRATrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    return trainer



def main():
    # 进行一些配置和检查
    args, training_args = setup_everything()
    # 加载各种组件
    trainer = init_components(args, training_args)
    # 开始训练
    logger.info("*** starting training ***")
    train_result = trainer.train()
    # 保存最好的checkpoint
    final_save_path = join(training_args.output_dir)
    trainer.save_model(final_save_path)  # Saves the tokenizer too
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
