# for dp in 0 0.1 0.3 0.5 0.8 1.0
# do    
#     CUDA_VISIBLE_DEVICES=0 python slu_main_dgt.py --target_domain GetWeather --model_saved_path experiments/12_31_parameters_refine/0shot --model_name GetWeather_0_dp_${dp} --model_name_or_path t5-base --early_stop 10 --prefix_dp ${dp} --n_samples 0 --using_pretrained_weights --model_dir experiments/12_29_new_temmplate_pre/0shot/GetWeather_0
# done

      
# for domain in RateBook
# do   
#     CUDA_VISIBLE_DEVICES=0 python slu_main_dgt.py --target_domain ${domain} --model_saved_path experiments/dgt_50shots_pretrain/ --model_name ${domain}_50_pre --model_name_or_path t5-base --early_stop 10 --prefix_dp 0.1 --n_samples 50 --pretrain 
# done

# for dp in 0.1
# do
#     for domain in SearchCreativeWork
#     do  
#         CUDA_VISIBLE_DEVICES=1 python slu_main_dgt.py --target_domain ${domain} --model_saved_path experiments/dgt_50shots_final/ --model_name ${domain}_50_dp_${dp} --model_name_or_path t5-base --early_stop 10 --prefix_dp ${dp} --n_samples 50 --using_pretrained_weights --model_dir experiments/dgt_50shots_pretrain/${domain}_50_pre --lr 1e-4
#     done
# done

# for domain in SearchScreeningEvent
# do  
#     CUDA_VISIBLE_DEVICES=0 python slu_main_dgt.py --target_domain ${domain} --model_saved_path experiments/100_dgt_50shots_baseline/ --model_name ${domain}_50_dp_${dp} --model_name_or_path t5-base --early_stop 10 --prefix_dp 0.1 --n_samples 50  --model_dir experiments/dgt_50shots_pretrain/${domain}_50_pre --lr 1e-4 --prefix_num_tokens 100
# done

# for domain in RateBook SearchCreativeWork
# do
#     CUDA_VISIBLE_DEVICES=0 python slu_main_dgt.py --target_domain ${domain} --model_saved_path experiments/1_3_allfintune_50shots/ --model_name ${domain}_50_dp_0.1 --model_name_or_path t5-base --early_stop 10 --prefix_dp 0.1 --n_samples 50  --model_dir experiments/dgt_50shots_pretrain/${domain}_50_pre --lr 1e-4 --prefix_num_tokens 100
# done


#全参数加载，调gw br
# for domain in GetWeather BookRestaurant
# do
#     CUDA_VISIBLE_DEVICES=0 python slu_main_dgt.py --target_domain ${domain} --model_saved_path experiments/BR_GW_full_retrain/ --model_name ${domain}_50_dp_0.1 --model_name_or_path t5-base --early_stop 10 --prefix_dp 0.1 --n_samples 50 --using_full_weights --model_dir experiments/dgt_50shots_pretrain/${domain}_50_pre --lr 1e-4 --prefix_num_tokens 5
# done


# for domain in RateBook SearchCreativeWork SearchScreeningEvent
# do
#     CUDA_VISIBLE_DEVICES=0 python slu_main_dgt.py --target_domain ${domain} --model_saved_path experiments/t5-base-qasf-0/0shot --model_name ${domain}_dp_0.1 --model_name_or_path mrm8488/t5-base-finetuned-qasc --early_stop 10 --prefix_dp 0.1 --n_samples 0 --using_pretrained_weights --model_dir experiments/12_29_new_temmplate_pre/0shot/${domain}_0 --lr 1e-4 --prefix_num_tokens 5
# done

# for num in 10 20 40 60 80 100
# do
#     for domain in GetWeather 
#     do
#         CUDA_VISIBLE_DEVICES=0 python slu_main_dgt.py --target_domain ${domain} --model_saved_path experiments/BR_GW_SCW_50shots/ --model_name ${domain}_50_num_${num} --model_name_or_path t5-base --early_stop 10 --prefix_dp 0.1 --n_samples 50 --lr 1e-4 --prefix_num_tokens ${num}
#     done
# done
# for domain in GetWeather SearchScreeningEvent
# do
#     CUDA_VISIBLE_DEVICES=0 python slu_main_dgt.py --target_domain ${domain} --model_saved_path experiments/all_refine_result/ --model_name ${domain}_0 --test_only --n_samples 50 --prefix_num_tokens 10 >> 1.9.txt
# done

# for domain in PlayMusic SearchCreativeWork
# do
#     CUDA_VISIBLE_DEVICES=0 python slu_main_dgt.py --target_domain ${domain} --model_saved_path experiments/all_refine_result/ --model_name ${domain}_0 --test_only --n_samples 50 --prefix_num_tokens 40 >> 1.9.txt
# done

for num in 1 5 50
do
    for domain in RateBook 
    do
        CUDA_VISIBLE_DEVICES=0 python slu_main_dgt.py --target_domain ${domain} --model_saved_path experiments/1.8_prefix_num/ --model_name ${domain}_${num} --model_name_or_path t5-base --early_stop 10 --prefix_dp 0.1 --n_samples 0 --lr 1e-4 --prefix_num_tokens ${num}
    done
done

for num in 40 60 80
do
    for domain in BookRestaurant  
    do
        CUDA_VISIBLE_DEVICES=0 python slu_main_dgt.py --target_domain ${domain} --model_saved_path experiments/1.8_prefix_num/ --model_name ${domain}_${num} --model_name_or_path t5-base --early_stop 10 --prefix_dp 0.1 --n_samples 0 --lr 1e-4 --prefix_num_tokens ${num}
    done
done
