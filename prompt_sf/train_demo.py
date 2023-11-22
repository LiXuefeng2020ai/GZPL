import argparse
from re import template
import torch
import pdb
import logging
import os
import sys
from tqdm import tqdm

parser = argparse.ArgumentParser("")
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--model", type=str, default='t5')  # tested model are gpt2/t5
parser.add_argument("--model_name_or_path", default='t5-base')
parser.add_argument("--model_saved_path", default='test_model_save')
parser.add_argument("--batch_size", type=int, default=5)
args = parser.parse_args()

log = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
fh = logging.FileHandler(os.path.join(args.model_saved_path,"log.txt"))
log.addHandler(fh)

from openprompt.data_utils.conditional_generation_dataset import WebNLGProcessor


dataset = {}
dataset['train'] = WebNLGProcessor().get_train_examples("../data/CondGen/webnlg_2017/")
dataset['validation'] = WebNLGProcessor().get_dev_examples("../data/CondGen/webnlg_2017/")
dataset['test'] = WebNLGProcessor().get_test_examples("../data/CondGen/webnlg_2017/")



# ## Construct Template
# 
# A template can be constructed from the yaml config, but it can also be constructed by directly passing arguments.
# You can load the plm related things provided by openprompt simply by calling:

# %%
from openprompt.plms import load_plm

plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)


# # Try more prompt!

# You can use templates other than manual template, for example the mixedtemplate is a good place to start.
# In MixedTemplate, you can use {"soft"} to denote a tunable template. 



# Or use a mix template
from openprompt.prompts import SoftTemplate,PrefixTuningTemplate

mytemplate = PrefixTuningTemplate(model=plm, tokenizer=tokenizer, text='{"placeholder":"text_a"} {"special": "<eos>"} {"mask"}',prefix_dropout=0.2,using_decoder_past_key_values=False)

# mytemplate = SoftTemplate(model=plm, tokenizer=tokenizer, text='{"placeholder":"text_a"} {"soft"} {"soft"} {"soft"} {"placeholder":"text_b"} {"soft"} {"mask"}.')

# mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text='{"placeholder":"text_a"} {"soft": "Question:"} {"placeholder":"text_b"}? Is it correct? {"mask"}.')


# To better understand how does the template wrap the example, we visualize one instance.

wrapped_example = mytemplate.wrap_one_example(dataset['train'][0]) 
# print(wrapped_example)
# pdb.set_trace()

# We provide a `PromptDataLoader` class to help you do all the above matters and wrap them into an `torch.DataLoader` style iterator.


from openprompt import PromptDataLoader

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256, 
    batch_size=5,shuffle=True, teacher_forcing=True, predict_eos_token=True,
    truncate_method="head")

validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256, 
    batch_size=5,shuffle=False, teacher_forcing=False, predict_eos_token=True,
    truncate_method="head")

test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256, 
    batch_size=5,shuffle=False, teacher_forcing=False, predict_eos_token=True,
    truncate_method="head")


# ## Now is time to build your prompt model!
# In this section we introduce using prompt to do classification, for other kinds of format, please see
# `generation_tutorial.ipynb`, `probing_tutorial.ipynb`.
# 


from openprompt import PromptForGeneration

use_cuda = True
prompt_model = PromptForGeneration(plm=plm,template=mytemplate, freeze_plm=True,tokenizer=tokenizer, plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model=  prompt_model.cuda()

from transformers import AdamW


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

# optimizer1 = AdamW(optimizer_grouped_parameters1, lr=1e-4)
optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)

from transformers.optimization import get_linear_schedule_with_warmup

tot_step  = len(train_dataloader)*5
scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)
# We provide generation a generation metric, you can also define your own. Note that it's not directly comparable to WebNLG's scripts evaluation.
from openprompt.utils.metrics import generation_metric
# Define evaluate function 
def evaluate(prompt_model, dataloader):
    generated_sentence = []
    groundtruth_sentence = []
    prompt_model.eval()
    
    for step, inputs in enumerate(dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        _, output_sentence = prompt_model.generate(inputs, **generation_arguments)
        # pdb.set_trace()
        generated_sentence.extend(output_sentence)
        groundtruth_sentence.extend(inputs['tgt_text'])
    score = generation_metric(generated_sentence, groundtruth_sentence, "sentence_bleu")
    return score
    # print("test_score", score, flush=True)


generation_arguments = {
    "max_length": 512,
    "max_new_tokens": None,
    "min_length": 5,
    "temperature": 1.0,
    "do_sample": False,
    "top_k": 0,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "num_beams": 5,
    "bad_words_ids": None
}
# pbar = tqdm(enumerate(train_dataloader),total=len(train_dataloader))


# global_step = 0 
# tot_loss = 0 
# log_loss = 0
# # training and generation.
# tot_loss = 0 
# for epoch in range(5):
#     best_dev_acc = 0.0
#     cur_dev_acc = 0.0
#     log.info("training begining!")
#     prompt_model.train()
#     for step, inputs in pbar:
#         global_step +=1
#         if use_cuda:
#             inputs = inputs.cuda()
#         loss = prompt_model(inputs)
#         loss.backward()
#         # tot_loss += loss.item()
#         torch.nn.utils.clip_grad_norm_(mytemplate.parameters(), 1.0)
#         optimizer.step()
#         scheduler.step()
#         optimizer.zero_grad()

#         pbar.set_description(f"Epoch {epoch}, global_step {global_step}, loss: {'%.2f'%loss.item()}")
#         # if global_step %500 ==0: 
#         #     print("Epoch {}, global_step {} average loss: {} lr: {}".format(epoch, global_step, (tot_loss-log_loss)/500, scheduler.get_last_lr()[0]), flush=True)
#         #     log_loss = tot_loss
#     cur_dev_acc = evaluate(prompt_model,validation_dataloader)
#     log.info(f"cur_dev_acc is {cur_dev_acc}")
#     if cur_dev_acc>best_dev_acc:
#         best_dev_acc = cur_dev_acc
#         log.info(f"found better model!!!")
#         torch.save(prompt_model.state_dict(),os.path.join(args.model_saved_path,"prefix_tuning_best_model.pth"))
#         cur_test_acc = evaluate(prompt_model,test_dataloader)
#         log.info(f"best result is {cur_test_acc}")
        

# prompt_model_new = prompt_model.load_state_dict(torch.load(os.path.join(args.model_saved_path,"best_model_new.pth")))
template_model = torch.load(os.path.join(args.model_saved_path,"prefix_tuning_best_model.pth"))
mytemplate.load_state_dict(template_model['template'])
# pdb.set_trace()

# model(test_dataloader[0])
prompt_model = PromptForGeneration(plm=plm,template=mytemplate,tokenizer=tokenizer)
# for batch in test_dataloader:
#     if use_cuda:
#         batch.cuda()
#     _,output = prompt_model.generate(batch,**generation_arguments)
    # pdb.set_trace()
# pdb.set_trace()
log.info("evaluating begin!!!")
res = evaluate(prompt_model, test_dataloader)
log.info(f"best result is {res}")
