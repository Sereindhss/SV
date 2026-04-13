#!/bin/bash

M=$1  

METHOD_LIST=('baseline' 'securevector' 'ase' 'ironmask' 'sfm' 'securevector_cluster' 'sv_dj')
METHOD=${METHOD_LIST[$M]}

# --- 日志配置开始 ---
BASE_FOLD="results"
LOG_DIR="${BASE_FOLD}/${METHOD}"
mkdir -p "${LOG_DIR}"
# 日志加上 ijb 前缀，方便区分
LOG_FILE="${LOG_DIR}/eval_ijb_$(date +%Y%m%d_%H%M%S).log"

# 使用 exec 将此后的所有 stdout (标准输出) 和 stderr (标准错误) 
# 同时输出到控制台和日志文件
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "开始任务: ${METHOD} (IJB-B / IJB-C)"
echo "日志文件保存至: ${LOG_FILE}"
echo "--------------------------------------"
# --- 日志配置结束 ---

# 第一部分：生成分数
for BM in 'c' # 'b'
do
    # Convert ijbx feature to id-template feature
    IJBX_BASE_FOLD=data/ijb/

    FEAT_LIST=${IJBX_BASE_FOLD}/ijb${BM}_feat.list
    TEMP_FEAT_LIST=${IJBX_BASE_FOLD}/ijb${BM}_template.list
    PAIR_LIST=${IJBX_BASE_FOLD}/ijb${BM}.pair.list

    FOLD=${BASE_FOLD}/${METHOD}/${BM}
    SCORE_LIST=${FOLD}/score.list  

    # 自动创建结果目录
    mkdir -p "${FOLD}"

    if [ ! -f  ${TEMP_FEAT_LIST} ]; then
        python3 eval/ijbx_template_feature.py \
            --feat_list ${FEAT_LIST} \
            --base_dir ${IJBX_BASE_FOLD} \
            --type ${BM} \
            --template_feature ${TEMP_FEAT_LIST} \
            --pair_list ${PAIR_LIST}
    fi
    # use the converted id-template feature file.

    if [[ $M == 0 ]]
    then
        # generate similarities
        python3 libs/baseline/gen_sim.py --feat_list ${TEMP_FEAT_LIST} --pair_list ${PAIR_LIST} --score_list ${SCORE_LIST}

    elif [[ $M == 1 ]]
    then
        KS=512
        K=64
        # enrollment
        if [ ! -f libs/SecureVector/keys/privatekey_${KS} ]; then
            echo 'generate paillier keys...'
            mkdir -p libs/SecureVector/keys/
            python3 libs/SecureVector/crypto_system.py --genkey 1 --key_size ${KS} 
        fi
        python3 libs/SecureVector/enrollment.py --feat_list ${TEMP_FEAT_LIST} --key_size ${KS} --K ${K} --folder ${FOLD}
        # generate similarities
        python3 libs/SecureVector/crypto_system.py --key_size ${KS} --K ${K} --folder ${FOLD} --pair_list ${PAIR_LIST} --score_list ${SCORE_LIST}
    
    elif [[ $M == 2 ]]
    then 
        ASE_DIM=4
        # enrollment
        python3 libs/ASE/enrollment.py --feat_list ${TEMP_FEAT_LIST} --folder ${FOLD} --ase_dim ${ASE_DIM}
        # generate similarities
        python3 libs/ASE/gen_sim.py --folder ${FOLD} --pair_list ${PAIR_LIST} --score_list ${SCORE_LIST}  

    elif [[ $M == 3 ]]
    then 
        ALPHA=16
        # enrollment
        python3 libs/IronMask/enrollment.py --feat_list ${TEMP_FEAT_LIST} --folder ${FOLD} --alpha ${ALPHA}        
        # generate similarities
        python3 libs/IronMask/gen_sim.py --folder ${FOLD} --pair_list ${PAIR_LIST} --score_list ${SCORE_LIST}  --alpha ${ALPHA} --feat_list ${FEAT_LIST}

    elif [[ $M == 4 ]]
    then 
        PRECISION=125                
        if [ ! -f libs/SFM/keys/gal_key ]; then
            echo 'generate SFM keys...'
            mkdir -p libs/SFM/keys/
            python3 libs/SFM/gen_sim.py  --genkey 1
        fi                
         # enrollment
        python3 libs/SFM/enrollment.py --feat_list ${FEAT_LIST} --folder ${FOLD} --precision ${PRECISION}        
        # generate similarities
        python3 libs/SFM/gen_sim.py --folder ${FOLD} --pair_list ${PAIR_LIST} --score_list ${SCORE_LIST}  --precision ${PRECISION}

    elif [[ $M == 5 ]]
    then
        KS=512
        K=64
        N_CLUSTERS=${2:-100}
        TOP_K=${3:-5}
        # key generation (reuse SecureVector keys)
        if [ ! -f libs/SecureVector/keys/privatekey_${KS}.npy ]; then
            echo 'generate paillier keys...'
            mkdir -p libs/SecureVector/keys/
            python3 libs/SecureVector/crypto_system.py --genkey 1 --key_size ${KS}
        fi
        # Step 1: Standard enrollment (all features)
        python3 libs/SecureVector/enrollment.py --feat_list ${TEMP_FEAT_LIST} --key_size ${KS} --K ${K} --folder ${FOLD}
        # Step 2: Build cluster index
        python3 libs/SV_cluster/build_index.py --feat_list ${TEMP_FEAT_LIST} --pair_list ${PAIR_LIST} --folder ${FOLD} --n_clusters ${N_CLUSTERS} --key_size ${KS} --K ${K} --metrics_output ${FOLD}/metrics_build.json
        # Step 3: Two-stage matching
        python3 libs/SV_cluster/cluster_match.py --folder ${FOLD} --pair_list ${PAIR_LIST} --score_list ${SCORE_LIST} --top_k ${TOP_K} --key_size ${KS} --K ${K} --metrics_output ${FOLD}/metrics_match.json

    elif [[ $M == 6 ]]
    then
        KS=512
        K=128
        S=2
        # key generation
        if [ ! -f libs/SV_DJ/keys/privatekey_${KS}.npy ]; then
            echo 'generate Damgard-Jurik keys...'
            python3 libs/SV_DJ/crypto_system.py --genkey 1 --key_size ${KS} --s ${S}
        fi
        # enrollment
        python3 libs/SV_DJ/enrollment.py --public_key libs/SV_DJ/keys/publickey --feat_list ${TEMP_FEAT_LIST} --key_size ${KS} --K ${K} --s ${S} --folder ${FOLD}
        # generate similarities
        python3 libs/SV_DJ/crypto_system.py --key_size ${KS} --K ${K} --s ${S} --folder ${FOLD} --pair_list ${PAIR_LIST} --score_list ${SCORE_LIST} --feat_list ${TEMP_FEAT_LIST}

    else
        echo 'key error'
    fi
done


echo "===== 即将进入评估阶段 ====="
# 第二部分：评估
for BM in 'c' # 'b'
do 
    echo "--------------------------------------"
    echo "[${METHOD}] 正在评估数据集: IJB-${BM}"
    IJBX_BASE_FOLD=data/ijb/
    PAIR_LIST=${IJBX_BASE_FOLD}/ijb${BM}.pair.list
    FOLD=${BASE_FOLD}/${METHOD}/${BM}
    SCORE_LIST=${FOLD}/score.list 
    
    if [ -f "${SCORE_LIST}" ]; then
        if [[ $M == 5 ]]; then
            python3 eval/eval_1vn.py --pair_list ${PAIR_LIST} --score_list ${SCORE_LIST} --metrics_output ${FOLD}/metrics_eval.json
            echo ""
            echo "[${METHOD}] 指标报告已保存至:"
            echo "  ${FOLD}/metrics_build.json"
            echo "  ${FOLD}/metrics_match.json"
            echo "  ${FOLD}/metrics_eval.json"
        elif [[ $M == 6 ]]; then
            python3 eval/eval_1vn.py --pair_list ${PAIR_LIST} --score_list ${SCORE_LIST} --metrics_output ${FOLD}/metrics_eval.json
            echo ""
            echo "[${METHOD}] 指标报告已保存至:"
            echo "  ${FOLD}/metrics_dj_enroll.json"
            echo "  ${FOLD}/metrics_dj.json"
            echo "  ${FOLD}/metrics_eval.json"
        else
            python3 eval/eval_1vn.py --pair_list ${PAIR_LIST} --score_list ${SCORE_LIST}
        fi
    else
        echo "跳过 IJB-${BM}: 分数文件未生成"
    fi
done
