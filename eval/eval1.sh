#!/bin/bash

M=$1  

METHOD_LIST=('baseline' 'securevector' 'ase' 'ironmask' 'sfm')
METHOD=${METHOD_LIST[$M]}

# --- 日志配置开始 ---
BASE_FOLD="results"
LOG_DIR="${BASE_FOLD}/${METHOD}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"

# 使用 exec 将此后的所有 stdout (标准输出) 和 stderr (标准错误) 
# 同时输出到控制台和日志文件
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "开始任务: ${METHOD}"
echo "日志文件保存至: ${LOG_FILE}"
echo "--------------------------------------"
# --- 日志配置结束 ---

# 第一部分：生成分数
# 注意：如果你需要跑所有数据集，请确保 'cfp' 'agedb' 没有被注释
for BM in 'lfw' 'cfp' 'agedb'
do 
    FEAT_LIST=data/${BM}/${BM}_feat.list
    PAIR_LIST=data/${BM}/pair.list

    FOLD=${BASE_FOLD}/${METHOD}/${BM}
    SCORE_LIST=${FOLD}/score.list

    # 自动创建结果目录
    mkdir -p "${FOLD}"

    if [[ $M == 0 ]]
    then
        # generate similarities for baseline
        python3 libs/baseline/gen_sim.py --feat_list ${FEAT_LIST} --pair_list ${PAIR_LIST} --score_list ${SCORE_LIST}

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
        python3 libs/SecureVector/enrollment.py --feat_list ${FEAT_LIST} --key_size ${KS} --K ${K} --folder ${FOLD}
        # generate similarities
        python3 libs/SecureVector/crypto_system.py --key_size ${KS} --K ${K} --folder ${FOLD} --pair_list ${PAIR_LIST} --score_list ${SCORE_LIST}
    
    elif [[ $M == 2 ]]
    then 
        ASE_DIM=4
        # enrollment
        python3 libs/ASE/enrollment.py --feat_list ${FEAT_LIST} --folder ${FOLD} --ase_dim ${ASE_DIM}
        # generate similarities
        python3 libs/ASE/gen_sim.py --folder ${FOLD} --pair_list ${PAIR_LIST} --score_list ${SCORE_LIST}  

    elif [[ $M == 3 ]]
    then 
        ALPHA=16
        # enrollment
        python3 libs/IronMask/enrollment.py --feat_list ${FEAT_LIST} --folder ${FOLD} --alpha ${ALPHA}        
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

    else
        echo 'key error'
    fi
done

# 第二部分：评估
for BM in 'lfw' 'cfp' 'agedb'  
do 
    echo "--------------------------------------"
    echo "[${METHOD}] 正在评估数据集: ${BM}"
    PAIR_LIST=data/${BM}/pair.list
    FOLD=${BASE_FOLD}/${METHOD}/${BM}
    SCORE_LIST=${FOLD}/score.list
    
    if [ -f "${SCORE_LIST}" ]; then
        python3 eval/eval_1v1.py --pair_list ${PAIR_LIST} --score_list ${SCORE_LIST}
    else
        echo "跳过 ${BM}: 分数文件未生成"
    fi
done
