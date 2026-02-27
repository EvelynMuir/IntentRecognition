#!/bin/bash

# =======================
# 用户可修改参数
# =======================
TOTAL_TASKS=24       # 总任务数
BATCH_SIZE=4         # 每批提交作业数（根据提交上限调整）
MAX_RUNNING=2        # 集群允许同时运行的任务数
USER=$(whoami)       # 当前用户名，用于查询队列
# =======================

for START in $(seq 4 $BATCH_SIZE $((TOTAL_TASKS-1))); do
    END=$((START + BATCH_SIZE - 1))
    if [ $END -ge $TOTAL_TASKS ]; then
        END=$((TOTAL_TASKS-1))
    fi

    # 等待队列空位
    while true; do
        # 查询当前用户运行或排队的任务数
        RUNNING=$(squeue -u $USER -h -r -t R,PD | wc -l)
        if [ $RUNNING -lt $MAX_RUNNING ]; then
            break
        fi
        echo "当前运行/排队任务数: $RUNNING, 等待空位..."
        sleep 3600  # 每小时检查一次
    done

    echo "提交任务 $START 到 $END"
    sbatch --array=${START}-${END}%${MAX_RUNNING} scripts/intentonomy_clip_vit_layer_cls_all.slurm
done

echo "所有任务提交完成！"
