# UABL: Uncertainty-Aware Boundary Learning
Step 1: baseline prediction (cls + mean patch)
z=fθ​(x)
Step 2: uncertainty estimation
从特征和 logits 预测样本-类别 uncertainty：
u=gϕ​(x,z)
训练时可用 soft annotation level 监督 u。
Step 3: boundary adaptation
根据 u 对 logits 做自适应变换：
zic′​=a(uic​)zic​+b(uic​)
其中 a,b 是小网络输出，或离散专家加权输出。
Step 4: classification loss
用 z′做最终 ASL。

正则
加 identity regularization：
Lid​=∥a(u)−1∥+∥b(u)∥
保证模型只做最小必要修正。