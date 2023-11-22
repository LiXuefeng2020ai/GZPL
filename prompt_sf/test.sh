#train
CUDA_VISIBLE_DEVICES=0 python slu_main.py --target_domain PlayMusic --model_saved_path experiments/test --model_name PlayMusic_0 --n_samples 0 --prefix_num_tokens 100

#test
CUDA_VISIBLE_DEVICES=0 python slu_main.py --target_domain PlayMusic --model_saved_path experiments/test --model_name PlayMusic_0  --n_samples 0 --prefix_num_tokens 100 --test_only
