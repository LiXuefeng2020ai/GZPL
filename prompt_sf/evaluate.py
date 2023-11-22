from itertools import repeat
from nltk.util import guess_encoding
import torch
from openprompt.utils.metrics import generation_metric
import pdb
import numpy as np
import math
from tqdm import tqdm
from collections import defaultdict
from preprecessor import domain2slot,unseen_slot,seen_slot
from config import get_params

args = get_params()


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

def evaluate_origin(prompt_model, dataloader):
    '''
    模型评估函数,使用bleu进行评价
    '''
    generated_sentence = []
    groundtruth_sentence = []
    prompt_model.eval()
    
    for step, inputs in enumerate(dataloader):
        inputs = inputs.cuda()
        _, output_sentence = prompt_model.generate(inputs, **generation_arguments)
        # pdb.set_trace()
        generated_sentence.extend(output_sentence)
        groundtruth_sentence.extend(inputs['tgt_text'])
    score = generation_metric(generated_sentence, groundtruth_sentence, "sentence_bleu")
    return score


def evaluate_snips(prompt_model, dataloader):
    '''
    针对于snips的slot filling的评价方法(平均f1分数)
    '''
    generated_sentence = []  # 模型预测的span
    groundtruth_sentence = [] # gold span
    
    gold_seen = 0
    gold_unseen = 0
    predict_seen = 0
    predict_unseen = 0
    correct_seen = 0
    correct_unseen = 0
    
    gold_dict = defaultdict(int)
    predict_dict = defaultdict(int)
    correct_dict =defaultdict(int)
    
    slot_list = domain2slot[args.target_domain]
    prompt_model.eval()
    
    pbar = tqdm(enumerate(dataloader),total=len(dataloader))
    for step, inputs in pbar:
        inputs = inputs.cuda()
        # pdb.set_trace()
        # _, logits = prompt_model(inputs) # 生成logits的熵作为判别依据
        _, output_sentence = prompt_model.generate(inputs) 
        # pdb.set_trace()
        # generated_sentence.extend(output_sentence)
        groundtruth = inputs["tgt_text"]           
        groundtruth_sentence.extend(inputs['tgt_text'])
        
        # repeat_dict = count_repeat(output_sentence) # 预测重复的部分
        # if repeat_dict:
        #     output_sentence = delete_repeat(repeat_dict,logits,output_sentence)
        generated_sentence.extend(output_sentence)
        # print(generated_sentence)
        # pdb.set_trace()
        
        assert len(groundtruth) == len(output_sentence)
        
        # 分类别的统计
        for i,count in enumerate(zip(groundtruth,output_sentence)):
            cur_slot = slot_list[i]
            if count[0]!="none":
                gold_dict[cur_slot] +=1
            if count[1]!="none":
                predict_dict[cur_slot] +=1
            if count[0]==count[1]!="none":
                correct_dict[cur_slot] +=1
    
    guess_trunk_nums = 0
    true_trunk_nums = 0
    correct_trunk_nums = 0
    
    assert len(generated_sentence) == len(groundtruth_sentence)
    
    for i in range(len(generated_sentence)):
        if generated_sentence[i] != "none":
            guess_trunk_nums +=1
        if groundtruth_sentence[i] != "none":
            true_trunk_nums +=1
        if generated_sentence[i]==groundtruth_sentence[i]!="none":
            correct_trunk_nums +=1
            
    tp = correct_trunk_nums
    fp = guess_trunk_nums-tp
    fn = true_trunk_nums-tp
    
    print(f"total number of slots is {true_trunk_nums}")
    print(f"guessed number of slots is {guess_trunk_nums}")
    print(f"correct number of guessed slots is {correct_trunk_nums}")
    
    # print("----------------------------------")
    # print(f"gold_dict : {gold_dict}")
    # print(f"predict_dict : {predict_dict}")
    # print(f"correct_dict : {correct_dict}")
    # print("----------------------------------")
    
    # 统计seen和unseen的f1分数
    for i,dict in enumerate([gold_dict,predict_dict,correct_dict]):
        for slot,nums in dict.items():
            if slot in seen_slot:
                if i==0:
                    gold_seen +=nums
                elif i==1:
                    predict_seen +=nums
                else:
                    correct_seen +=nums
            elif slot in unseen_slot:
                if i==0:
                    gold_unseen +=nums
                elif i==1:
                    predict_unseen +=nums
                else:
                    correct_unseen +=nums
            else:
                print("slot error!")
    
    seen_slot_f1 = calcualate_f1_score(correct_seen,predict_seen-correct_seen,gold_seen-correct_seen)
    print(f"seen slot num is : {gold_seen}")
    print(f"seen slot f1 score is : {seen_slot_f1}")
    unseen_slot_f1 = calcualate_f1_score(correct_unseen,predict_unseen-correct_unseen,gold_unseen-correct_unseen)
    print(f"unseen slot num is : {gold_unseen}")
    print(f"unseen slot f1 score is {unseen_slot_f1}")
    print("--------------------------------------------")
    
    return calcualate_f1_score(tp,fp,fn)


def evaluate_snips_test(prompt_model, dataloader):
    '''
    针对于snips的slot filling的评价方法(平均f1分数)
    '''
    generated_sentence = []  # 模型预测的span
    groundtruth_sentence = [] # gold span
    
    gold_seen = 0
    gold_unseen = 0
    predict_seen = 0
    predict_unseen = 0
    correct_seen = 0
    correct_unseen = 0
    
    gold_dict = defaultdict(int)
    predict_dict = defaultdict(int)
    correct_dict =defaultdict(int)
    
    repeat_predict_nums = 0
    
    slot_list = domain2slot[args.target_domain]
    prompt_model.eval()
    
    pbar = tqdm(enumerate(dataloader),total=len(dataloader))
    for step, inputs in pbar:
        inputs = inputs.cuda()
        # pdb.set_trace()
        # _, logits = prompt_model(inputs) # 生成logits的熵作为判别依据
        scores, output_sentence = prompt_model.generate(inputs)
        # print(output_sentence)
        # pdb.set_trace()
        # generated_sentence.extend(output_sentence)
        groundtruth = inputs["tgt_text"]           
        groundtruth_sentence.extend(inputs['tgt_text'])
        
        repeat_dict = count_repeat(output_sentence) # 预测重复的部分
        for i in repeat_dict.values():
            repeat_predict_nums+=(len(i)-1)
        # pdb.set_trace()
        # if repeat_dict:
        #     output_sentence = delete_repeat(repeat_dict,scores,output_sentence)
        generated_sentence.extend(output_sentence)
        
        assert len(groundtruth) == len(output_sentence)
        
        # 分类别的统计
        for i,count in enumerate(zip(groundtruth,output_sentence)):
            cur_slot = slot_list[i]
            if count[0]!="none":
                gold_dict[cur_slot] +=1
            if count[1]!="none":
                predict_dict[cur_slot] +=1
            if count[0]==count[1]!="none":
                correct_dict[cur_slot] +=1
    
    guess_trunk_nums = 0
    true_trunk_nums = 0
    correct_trunk_nums = 0
    # pdb.set_trace()
    assert len(generated_sentence) == len(groundtruth_sentence)
    
    for i in range(len(generated_sentence)):
        if generated_sentence[i] != "none":
            guess_trunk_nums +=1
        if groundtruth_sentence[i] != "none":
            true_trunk_nums +=1
        if generated_sentence[i]==groundtruth_sentence[i]!="none":
            correct_trunk_nums +=1
            
    tp = correct_trunk_nums
    fp = guess_trunk_nums-tp
    fn = true_trunk_nums-tp
    
    print(f"total number of slots is {true_trunk_nums}")
    print(f"guessed number of slots is {guess_trunk_nums}")
    print(f"correct number of guessed slots is {correct_trunk_nums}")
    
    # print("----------------------------------")
    # print(f"gold_dict : {gold_dict}")
    # print(f"predict_dict : {predict_dict}")
    # print(f"correct_dict : {correct_dict}")
    # print("----------------------------------")
    
    # 统计seen和unseen的f1分数
    for i,dict in enumerate([gold_dict,predict_dict,correct_dict]):
        for slot,nums in dict.items():
            if slot in seen_slot:
                if i==0:
                    gold_seen +=nums
                elif i==1:
                    predict_seen +=nums
                else:
                    correct_seen +=nums
            elif slot in unseen_slot:
                if i==0:
                    gold_unseen +=nums
                elif i==1:
                    predict_unseen +=nums
                else:
                    correct_unseen +=nums
            else:
                print("slot error!")
    
    seen_slot_f1 = calcualate_f1_score(correct_seen,predict_seen-correct_seen,gold_seen-correct_seen)
    # print(f"seen slot num is : {gold_seen}")
    print(f"seen slot f1 score is : {seen_slot_f1}")
    unseen_slot_f1 = calcualate_f1_score(correct_unseen,predict_unseen-correct_unseen,gold_unseen-correct_unseen)
    # print(f"unseen slot num is : {gold_unseen}")
    print(f"unseen slot f1 score is {unseen_slot_f1}")
    # print("--------------------------------------------")
    print(f"repeat nums: {repeat_predict_nums}")
    
    return calcualate_f1_score(tp,fp,fn)


def calcualate_f1_score(tp, fp, fn):
    p = 0 if tp + fp == 0 else 1.*tp / (tp + fp)
    r = 0 if tp + fn == 0 else 1.*tp / (tp + fn)
    f = 0 if p + r == 0 else 2 * p * r / (p + r)
    print(f"precision is {p} , recall is {r}")
    return f

def entropy(c):
    result=-1;
    if(len(c)>0):
        result=0
    for x in c:
        # print(x)
        result+=(-x)*math.log(x,2)
    return result


def count_repeat(predict_list):
    res = {}
    for index,i in enumerate(predict_list):
        if i == "none":
            continue
        else:
            if i in res:
                res[i].append(index)
            else:
                res[i] = [index]
    key_set = list(res.keys())
    for j in key_set:
        if len(res[j])==1: # 只出现的一次不算重复
            res.pop(j)
    return res

def delete_repeat(repeat_dict,predict_logits,predict_sentence):
    '''
    先用基于输出熵的方式进行梳理
    '''
    for span,repeat_list in repeat_dict.items():
        span_length = len(span)
        max_index = -1
        max_entropy = 0
        max_probs = 0
        for index in repeat_list: 
            # 寻找最大熵
            # logits_to_list = torch.nn.functional.softmax(predict_logits[index][2:2+span_length],dim=-1).cpu().detach().numpy()
            # entropy_list = []
            # for i in logits_to_list:
            #     cur_entropy = entropy(np.squeeze(i))
            #     entropy_list.append(cur_entropy)
            # mean_entropy = sum(entropy_list)/len(entropy_list)
            # if mean_entropy>max_entropy:
            #     max_entropy = cur_entropy
            #     max_index = index
            
            # 使用最大概率
            logits_to_list = torch.nn.functional.softmax(predict_logits[0][index],dim=-1)
            # pdb.set_trace()
            # if span_length == 1:
            cur_max = torch.max(logits_to_list).cpu().detach().numpy()
            # else:
            #     cur_max = torch.max(torch.max(logits_to_list,1)[0],0)[0].cpu().detach().numpy()
            if cur_max>max_probs:
                max_probs = cur_max
                max_index = index
                    
                
        for j in repeat_list: # 最大熵处保留，其他位置换位none
            if j!=max_index:
                predict_sentence[j] = "none"
    # pdb.set_trace()
    return predict_sentence


def universal_evaluate_snips(prompt_model, dataloader):
    '''
    针对于snips的slot filling的评价方法(平均f1分数)
    '''
    generated_sentence = []  # 模型预测的span
    groundtruth_sentence = [] # gold span
    
    gold_seen = 0
    gold_unseen = 0
    predict_seen = 0
    predict_unseen = 0
    correct_seen = 0
    correct_unseen = 0
    
    gold_dict = defaultdict(int)
    predict_dict = defaultdict(int)
    correct_dict =defaultdict(int)
    
    slot_list = domain2slot[args.target_domain]
    prompt_model.eval()
    
    pbar = tqdm(enumerate(dataloader),total=len(dataloader))
    for step, inputs in pbar:
        inputs = inputs.cuda()
        _, output_sentence = prompt_model.generate(inputs)
        # print(output_sentence)
        # pdb.set_trace()
        # generated_sentence.extend(output_sentence)
        assert len(output_sentence)==len(inputs['tgt_text'])
        for groundtruth,output in zip(inputs['tgt_text'],output_sentence):
        # groundtruth = inputs["tgt_text"][0].split(';')
            groundtruth = groundtruth.split(";")         
            groundtruth_sentence.extend(groundtruth)
            output = output.split(';')
            # print(groundtruth)
            # print(output)
            # pdb.set_trace()
            if len(output)>len(groundtruth):
                output = output[:len(groundtruth)]
            elif len(output)<len(groundtruth):
                output.extend(['none']*(len(groundtruth)-len(output)))
            assert len(groundtruth) == len(output)
            generated_sentence.extend(output)
            # 分类别的统计
            for i,count in enumerate(zip(groundtruth,output)):
                cur_slot = slot_list[i]
                if count[0]!="none":
                    gold_dict[cur_slot] +=1
                if count[1]!="none":
                    predict_dict[cur_slot] +=1
                if count[0]==count[1]!="none":
                    correct_dict[cur_slot] +=1
    
    guess_trunk_nums = 0
    true_trunk_nums = 0
    correct_trunk_nums = 0
    
    assert len(generated_sentence) == len(groundtruth_sentence)
    
    for i in range(len(generated_sentence)):
        if generated_sentence[i] != "none":
            guess_trunk_nums +=1
        if groundtruth_sentence[i] != "none":
            true_trunk_nums +=1
        if generated_sentence[i]==groundtruth_sentence[i]!="none":
            correct_trunk_nums +=1
            
    tp = correct_trunk_nums
    fp = guess_trunk_nums-tp
    fn = true_trunk_nums-tp
    
    print(f"total number of slots is {true_trunk_nums}")
    print(f"guessed number of slots is {guess_trunk_nums}")
    print(f"correct number of guessed slots is {correct_trunk_nums}")
    
    # print("----------------------------------")
    # print(f"gold_dict : {gold_dict}")
    # print(f"predict_dict : {predict_dict}")
    # print(f"correct_dict : {correct_dict}")
    # print("----------------------------------")
    
    # 统计seen和unseen的f1分数
    for i,dict in enumerate([gold_dict,predict_dict,correct_dict]):
        for slot,nums in dict.items():
            if slot in seen_slot:
                if i==0:
                    gold_seen +=nums
                elif i==1:
                    predict_seen +=nums
                else:
                    correct_seen +=nums
            elif slot in unseen_slot:
                if i==0:
                    gold_unseen +=nums
                elif i==1:
                    predict_unseen +=nums
                else:
                    correct_unseen +=nums
            else:
                print("slot error!")
    
    seen_slot_f1 = calcualate_f1_score(correct_seen,predict_seen-correct_seen,gold_seen-correct_seen)
    print(f"seen slot num is : {gold_seen}")
    print(f"seen slot f1 score is : {seen_slot_f1}")
    unseen_slot_f1 = calcualate_f1_score(correct_unseen,predict_unseen-correct_unseen,gold_unseen-correct_unseen)
    print(f"unseen slot num is : {gold_unseen}")
    print(f"unseen slot f1 score is {unseen_slot_f1}")
    print("--------------------------------------------")
    
    return calcualate_f1_score(tp,fp,fn)




