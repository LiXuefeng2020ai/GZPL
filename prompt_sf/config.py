import argparse

def get_params():
    parser = argparse.ArgumentParser(description="Prompt_based_SLU")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--plm_eval_mode", action="store_true")
    parser.add_argument("--model", type=str, default='t5')  # tested model are gpt2/t5
    parser.add_argument("--model_name_or_path", default='t5-base')
    parser.add_argument("--model_saved_path", default='test_model_save')
    parser.add_argument("--model_name", default='snips')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--target_domain", type=str, default="PlayMusic")
    parser.add_argument("--dataset_dir",type=str,default="data/snips/")
    parser.add_argument("--test_only",default=False,action="store_true")
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument("--use_cuda",default=True,action="store_false")
    parser.add_argument('--max_epochs', type=int, default=15)
    parser.add_argument('--early_stop', type=int, default=5)
    parser.add_argument("--prefix_num_tokens", type=int, default=5)
    parser.add_argument("--prefix_dp", type=float, default=0.2)
    parser.add_argument("--n_samples", type=int, default=0)
    parser.add_argument("--source_samples", type=int, default=0)
    parser.add_argument("--slots_emb_file", type=str, default="../data/snips/emb/slot_word_char_embs_based_on_each_domain.dict")

    parser.add_argument("--temprature", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=2)
    parser.add_argument("--beta", type=float, default=0.1)
    
    #pretrain参数
    parser.add_argument("--pretrain", default=False,action="store_true" )
    parser.add_argument("--model_dir", type=str, default="experiments/new_dgt/0shot/new_rb_0")
    parser.add_argument("--using_pretrained_weights",default=False,action="store_true")
    parser.add_argument("--using_full_weights", default=False, action="store_true")
    
    args = parser.parse_args()
    
    return args