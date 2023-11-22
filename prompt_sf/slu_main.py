from re import template
import random
import numpy as np
import torch
import pdb
import logging
import os
import sys
from torch.nn.functional import embedding
from tqdm import tqdm
from config import get_params
from data_reader import snips_data_reader,pretrain_snips_data_reader,universal_snips_data_reader
from openprompt.prompts import PrefixTuningTemplate
from openprompt.plms import load_plm
from openprompt import PromptDataLoader
from openprompt import PromptForGeneration
from openprompt.prompts import T5TemplateGenerator
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from evaluate import evaluate_origin,evaluate_snips, evaluate_snips_test,universal_evaluate_snips
from preprecessor import domain2slot
import math
from evaluate import count_repeat,delete_repeat,calcualate_f1_score
import pickle
import json
from torch.nn.parallel import DistributedDataParallel as DDP
from data_processor import domain2slot

args = get_params()

# 固定随机数种子
if args.random_seed >= 0:
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)


# 模型保存路径
model_saved_dir = os.path.join(args.model_saved_path,args.model_name)
if not os.path.exists(model_saved_dir):
    os.makedirs(model_saved_dir)

params_json = args.__dict__
with open(os.path.join(model_saved_dir,'params.json'), 'w')as fout:
    json.dump(params_json, fout,indent=4)
    
# log信息设置
log = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
fh = logging.FileHandler(os.path.join(model_saved_dir,"log.txt"))
log.addHandler(fh)

# 初始数据集
if args.pretrain:
    dataset = pretrain_snips_data_reader()
else:
    dataset = snips_data_reader()

# 获取预训练模型及相关配置文件
plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

# 构建模板
if args.pretrain:
    mytemplate = PrefixTuningTemplate(
        model=plm,
        tokenizer=tokenizer,
        text='{"placeholder":"text_a"} {"placeholder":"text_b"} {"special": "<eos>"} {"mask"}',
        prefix_dropout=args.prefix_dp,
        using_decoder_past_key_values=True,  # 解码端也使用prompt token
        num_token=args.prefix_num_tokens
        )
else:
    mytemplate = PrefixTuningTemplate(
        model=plm,
        tokenizer=tokenizer,
        text='{"placeholder":"text_a"} music_item best_rating facility poi genre service country current_location city object_type object_name geographic_poi timeRange restaurant_name year movie_type track object_location_type spatial_relation cuisine object_part_of_series_type condition_description playlist restaurant_type sort movie_name rating_value condition_temperature party_size_description location_name party_size_number playlist_owner state entity_name served_dish rating_unit artist album object_select {"placeholder":"text_b"} {"special": "<eos>"} {"mask"}',
        prefix_dropout=args.prefix_dp,
        using_decoder_past_key_values=True,  # 解码端也使用prompt token
        num_token=args.prefix_num_tokens,
        # using_pretrained_weights=True,
        # weights = [encoder_embs,decoder_embs]
        )

# 目标域的slot数
slot_nums = len(domain2slot[args.target_domain])

# # 数据集加载
if not args.test_only:
    train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256, 
        batch_size=args.batch_size,shuffle=True, teacher_forcing=True, predict_eos_token=True,
        truncate_method="head")

    validation_dataloader = PromptDataLoader(dataset=dataset["val"], template=mytemplate, tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256, 
        batch_size=slot_nums,shuffle=False, teacher_forcing=False, predict_eos_token=True,
        truncate_method="head")

test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256, 
    batch_size=slot_nums,shuffle=False, teacher_forcing=False, predict_eos_token=True,
    truncate_method="head")

# Using different optimizer for prompt parameters and model parameters
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
{
    "params": [p for n, p in mytemplate.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
    "weight_decay": 0.0,
},
{
    "params": [p for n, p in mytemplate.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
    "weight_decay": 0.0,
},
]

optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
if not args.test_only:
    tot_step  = len(train_dataloader)*5
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)


# 测试
if args.test_only:
    template_model = torch.load(os.path.join(model_saved_dir,"prefix_tuning_best_model.pth"))
    mytemplate.load_state_dict(template_model['template']) # 预训练模型参数未改动，加载模板相关参数即可
    prompt_model = PromptForGeneration(plm=plm,template=mytemplate,tokenizer=tokenizer)
    if args.use_cuda:
        prompt_model = prompt_model.cuda()

    log.info("evaluating begin!!!")
    # res = evaluate_snips_test(prompt_model,test_dataloader)
    res = universal_evaluate_snips(prompt_model,test_dataloader)
    print(f"best result is {res}")
    print("-------------------------------------------------------")

# 训练 
else:
    if args.using_full_weights:
        model_dir = args.model_dir
        template_model = torch.load(os.path.join(model_dir,"prefix_tuning_best_model.pth"))
        mytemplate.load_state_dict(template_model["template"])
    prompt_model = PromptForGeneration(plm=plm,template=mytemplate, freeze_plm=True,tokenizer=tokenizer, plm_eval_mode=args.plm_eval_mode)
    if args.use_cuda:
        prompt_model.cuda()
    tot_loss = 0 
    log_loss = 0
    # training and generation.
    tot_loss = 0
    best_dev_acc = 0.0
    cur_dev_acc = 0.0
    stop_steps_count = 0
    stop_flags = False

    for epoch in range(args.max_epochs):
        pbar = tqdm(enumerate(train_dataloader),total=len(train_dataloader))
        log.info("-----------------------------------------")
        log.info("training begining!")
        log.info("-----------------------------------------")
        prompt_model.train()
        global_step = 0
        if stop_flags:
            break
        else:
            for step, inputs in pbar:
                global_step +=1
                if args.use_cuda:
                    input_sentence = inputs.cuda()

                loss,logits = prompt_model(input_sentence)
                loss.backward()
                # input_sentence = input_sentence["input_ids"]

                tot_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(mytemplate.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                pbar.set_description(f"Epoch {epoch}, global_step {global_step}, loss: {'%.2f'%loss.item()}")
                if global_step %2000 ==0 or global_step==len(train_dataloader)-1: 
                    print("Epoch {}, global_step {} average loss: {} lr: {}".format(epoch, global_step, (tot_loss-log_loss)/2000, scheduler.get_last_lr()[0]), flush=True)
                    log_loss = tot_loss
                    # torch.save(prompt_model.state_dict(),os.path.join(model_saved_dir,"prefix_tuning_best_model.pth"))
                    # 保存最佳模型
                    cur_dev_acc = universal_evaluate_snips(prompt_model,validation_dataloader)
                    log.info(f"cur_dev_acc is {cur_dev_acc}")
                    if cur_dev_acc>best_dev_acc:
                        best_dev_acc = cur_dev_acc
                        log.info(f"found better model!!!")
                        torch.save(prompt_model.state_dict(),os.path.join(model_saved_dir,"prefix_tuning_best_model.pth"))
                        cur_test_acc = universal_evaluate_snips(prompt_model,test_dataloader)
                        log.info("-----------------------------------------")
                        log.info(f"best result is {cur_test_acc}")
                        log.info("-----------------------------------------")
                        stop_steps_count = 0
                    
                    else:
                        stop_steps_count+=1
                        log.info(f"No better model found {stop_steps_count}/{args.early_stop}")
                        if stop_steps_count==args.early_stop:
                            stop_flags = True
                            log.info(f"training finished!!!")
                            break

