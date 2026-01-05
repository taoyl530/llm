## LLM for CDSR (Semantic-Structure Co-Learning Version)
gpu_id=0
dataset="amazon"
seed_list=(42)
inter_file="cloth_sport"
llm_emb_file="itm_emb_np"
user_emb_file="usr_profile_emb"

model_name="llm4cdsr"
alpha=0.1
beta=0.1

# === [NEW] Semantic-Behavior Alignment Config (语义对齐参数) ===
# 开启对齐损失，权重建议 0.1~0.5，温度系数 0.2
align_weight=0.1
align_tau=0.2

# === [UPGRADED] Global Semantic Graph Config (全局语义图参数) ===
# 开启 GNN (use_gnn=1) 以启用图学习
use_gnn=1           
gnn_layer=2         

# 动态图学习参数
# graph_aug_k: 为每个物品/用户寻找 Top-K 个语义邻居 (建议 2-5)
# graph_learn_weight: 图结构蒸馏损失权重
graph_aug_enable=1
graph_aug_k=2
graph_learn_tau=1.0
graph_learn_weight=0.01

# 注意：graph_cold_threshold 参数在全局图模式下可能不再强制依赖，但保留以兼容代码

for seed in ${seed_list[@]}
do
    # 构建 GNN 和 新模块的参数字符串
    EXTRA_ARGS=""
    if [ "$use_gnn" = "1" ]; then
        EXTRA_ARGS="--use_gnn --layer_num ${gnn_layer}"
        
        # 传入图学习参数
        EXTRA_ARGS="${EXTRA_ARGS} --graph_aug_enable ${graph_aug_enable} --graph_aug_k ${graph_aug_k}"
        EXTRA_ARGS="${EXTRA_ARGS} --graph_learn_tau ${graph_learn_tau} --graph_learn_weight ${graph_learn_weight}"
        
        # 传入对齐参数
        EXTRA_ARGS="${EXTRA_ARGS} --align_weight ${align_weight} --align_tau ${align_tau}"
    fi

    python main.py --dataset ${dataset} \
                --inter_file ${inter_file} \
                --model_name ${model_name} \
                --hidden_size 128 \
                --train_batch_size 128 \
                --max_len 200 \
                --gpu_id ${gpu_id} \
                --num_workers 8 \
                --num_train_epochs 80 \
                --seed ${seed} \
                --check_path "" \
                --patience 20 \
                --ts_user 12 \
                --ts_item 13 \
                --log \
                --domain "AB" \
                --local_emb \
                --global_emb \
                --freeze_emb \
                --llm_emb_file ${llm_emb_file} \
                --user_emb_file ${user_emb_file} \
                ${EXTRA_ARGS} \
                --alpha ${alpha} \
                --beta ${beta} 
done