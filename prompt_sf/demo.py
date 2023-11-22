
# from build.lib.openprompt.prompt_base import Verbalizer
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate,SoftTemplate,PrefixTuningTemplate,PTRTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptModel,PromptForClassification,PromptForGeneration
from openprompt.pipeline_base import PromptDataLoader
import torch
import torch.nn.functional as F
import pdb
from transformers import BertModel,BertTokenizer,GPT2PreTrainedModel
import logging
import sys
import math




log = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
fh = logging.FileHandler('log.txt')
log.addHandler(fh)
print("?")
log.info("test")
log.removeHandler(fh)
# pdb.set_trace()
dataset_new = [
    InputExample(
    guid=0,
    text_a="add camille to the this is lady antebellum playlist."
)]

plm, tokenizer, model_config, WrapperClass = load_plm("t5", "t5-base")

# input_ids = tokenizer("add another song to the acoustic soul playlist .",return_tensors="pt").input_ids
# res = plm.generate(input_ids)
# decode_res = tokenizer.decode(res[0],skip_special_tokens=True)
# print(res)
# print(decode_res)
# pdb.set_trace()
# prompt_template = ManualTemplate(
#     text = '{"placeholder":"text_a"} {"soft"} Playlist refers to ',
#     tokenizer=tokenizer
# )

prompt_template = PTRTemplate(
    model=plm,
    text = '{"placeholder":"text_a"} {"soft"} Playlist refers to {"mask"} ',
    tokenizer= tokenizer
)

wrapped_example = prompt_template.wrap_one_example(dataset_new[0]) 
print(wrapped_example)
# pdb.set_trace()

wrapped_t5tokenizer = WrapperClass(max_seq_length=128, decoder_max_length=3, tokenizer=tokenizer,truncate_method="head")
tokenized_example = wrapped_t5tokenizer.tokenize_one_example(wrapped_example, teacher_forcing=False)
print(tokenized_example)
print(tokenizer.convert_ids_to_tokens(tokenized_example['input_ids']))
print(tokenizer.convert_ids_to_tokens(tokenized_example['decoder_input_ids']))
# pdb.set_trace()

prompt_model = PromptForGeneration(
    template=prompt_template,
    plm=plm,
)

# print(prompt_model)

data_loader = PromptDataLoader(
    dataset = dataset_new,
    tokenizer = tokenizer,
    template = prompt_template, 
    tokenizer_wrapper_class=WrapperClass,
)

# pdb.set_trace()

prompt_model.eval()
with torch.no_grad():
    for batch in data_loader:
        pdb.set_trace()
        logits = prompt_model.generate(batch)
        print(logits[1])
        pdb.set_trace()
        res = tokenizer.decode(logits[1],skip_special_tokens=True)
        print(res)
        
class test():
    def __init__(self,number) -> None:
        super().__init__()
        self.number = number
        print("init finished")



if __name__ == "__main__":
    print(1)
    pdb.set_trace()
    test1 = test(3)
    print(test1.number)

