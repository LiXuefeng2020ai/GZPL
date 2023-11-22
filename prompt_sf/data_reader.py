from data_processor import SnipsProcessor
from config import get_params
import pdb
from preprecessor import domain2slot
args = get_params()


domains = ["AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]

def snips_data_reader(n_samples=0):
    dataset = {}
    snips_process = SnipsProcessor()

    dataset["train"] = []
    dataset["val"] = []
    dataset["test"] = []
    
    print(args.target_domain)
    assert args.target_domain in domains
    n_samples = args.n_samples
    for domain in domains:
        if domain!= args.target_domain:
            single_domain_dataset = snips_process.get_examples(data_dir=args.dataset_dir+domain,domain=domain)
            if args.source_samples:
                num_train = args.source_samples*len(domain2slot[domain])
                dataset["train"].extend(single_domain_dataset[:num_train])
            else:
                dataset["train"].extend(single_domain_dataset)
        else:
            val_test_dataset = snips_process.get_examples(data_dir=args.dataset_dir+domain,domain=domain)
            num_val = 500*len(domain2slot[args.target_domain])
            num_train = n_samples*len(domain2slot[args.target_domain]) # 目标域前n条数据作为少样本
            dataset["train"].extend(val_test_dataset[:num_train])
            dataset["val"].extend(val_test_dataset[num_train:num_val])  # 目标领域前500条数据中剩下部分作为验证集
            dataset["test"].extend(val_test_dataset[num_val:]) # 其余数据作为测试集


    return dataset

def pretrain_snips_data_reader():
    dataset = {}
    snips_process = SnipsProcessor()

    dataset["train"] = []
    dataset["pos_example"] = []
    dataset["val"] = []
    dataset["test"] = []
    assert args.target_domain in domains
    print(f"target domain : {args.target_domain}")
    for domain in domains:
        if domain!= args.target_domain:
            single_domain_dataset = snips_process.pretrain_get_examples(data_dir=args.dataset_dir+domain,domain=domain,key_words="train")
            if args.source_samples:
                num_train = args.source_samples*len(domain2slot[domain])
                dataset["train"].extend(single_domain_dataset[:num_train])
            else:
                dataset["train"].extend(single_domain_dataset)
            
        else:
            val_test_dataset = snips_process.pretrain_get_examples(data_dir=args.dataset_dir+domain,domain=domain,key_words="val")
            num_val = 1500
            num_few_shot_samples = args.n_samples*len(domain2slot[args.target_domain])
            dataset["train"].extend(val_test_dataset[:num_few_shot_samples])
            dataset["val"].extend(val_test_dataset[num_few_shot_samples:num_val])  # 目标领域前500条数据作为验证集
            dataset["test"].extend(val_test_dataset[num_val:]) # 其余数据作为测试集
    # pdb.set_trace()
    return dataset

def universal_snips_data_reader():

    dataset = {}
    snips_process = SnipsProcessor()

    dataset["train"] = []
    dataset["val"] = []
    dataset["test"] = []
    assert args.target_domain in domains
    print(f"target domain : {args.target_domain}")
    for domain in domains:
        if domain!= args.target_domain:
            single_domain_dataset = snips_process.concate_get_examples(data_dir=args.dataset_dir+domain,domain=domain,key_words="train")
            if args.source_samples:
                # num_train = args.source_samples*len(domain2slot[domain])
                dataset["train"].extend(single_domain_dataset[:args.source_samples])
            else:
                dataset["train"].extend(single_domain_dataset)
            
        else:
            val_test_dataset = snips_process.concate_get_examples(data_dir=args.dataset_dir+domain,domain=domain,key_words="val")
            # num_val = 1500
            num_few_shot_samples = args.n_samples
            num_val = num_few_shot_samples+500
            dataset["train"].extend(val_test_dataset[:num_few_shot_samples])
            dataset["val"].extend(val_test_dataset[num_few_shot_samples:num_val])  # 目标领域前500条数据作为验证集
            dataset["test"].extend(val_test_dataset[num_val:]) # 其余数据作为测试集
    return dataset

    

if __name__ == "__main__":
    dataset = snips_data_reader()
    # pdb.set_trace()