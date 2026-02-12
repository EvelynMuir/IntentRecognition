# most borrow from: https://github.com/KMnP/intentonomy/blob/master/eval_utils.py
# 参考 HLEG/data_utils/metrics.py 的计算方式

#!/usr/bin/env python3
"""
evaluation metrics for multi-label classification
"""
import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics import f1_score

SUBSET2IDS = {
    'easy': [0, 7, 19],
    'medium': [1, 3, 4, 6, 9, 10, 11, 12, 13, 14, 15, 16, 18, 22, 26],
    'hard': [2, 5, 8, 17, 20, 21, 23, 24, 25, 27],
    'object': [0, 3, 10, 11, 12, 16, 23],
    'context': [7, 8],
    "other": [1, 2, 4, 5, 6, 9, 13, 17, 18, 19, 20, 22, 25, 27, 14, 15, 21,  24, 26],
}


def voc_ap(rec, prec, true_num):
    """计算单个类别的平均精度 (AP)
    
    Args:
        rec: 召回率数组
        prec: 精确率数组
        true_num: 真实正样本数量
    
    Returns:
        ap: 平均精度值
    """
    if true_num == 0:
        return 0.0
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_mAP(scores: np.ndarray, targets: np.ndarray, return_each: bool = False) -> float:
    """计算多标签分类的平均精度均值 (mAP)
    
    参考 HLEG/utils/metric.py 的实现方式
    
    Args:
        scores: 预测分数，shape (num_samples, num_classes)
        targets: 真实标签（multihot编码），shape (num_samples, num_classes)
        return_each: 是否返回每个类别的AP
    
    Returns:
        mAP: 平均精度均值，如果return_each=True，则返回(mAP, aps)
    """
    sample_num = len(targets)
    class_num = scores.shape[1]
    aps = []

    for class_id in range(class_num):
        confidence = scores[:, class_id]
        sorted_ind = np.argsort(-confidence)
        sorted_label = targets[sorted_ind, class_id]

        tp = np.zeros(sample_num)
        fp = np.zeros(sample_num)
        for i in range(sample_num):
            tp[i] = (sorted_label[i] > 0)
            fp[i] = (sorted_label[i] <= 0)
        true_num = sum(tp)
        
        if true_num == 0:
            aps.append(0.0)
            continue
            
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(true_num)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, true_num)
        aps.append(ap)

    np.set_printoptions(precision=6, suppress=True)
    aps = np.array(aps) * 100
    mAP = np.mean(aps)
    if return_each:
        return mAP, aps
    return mAP


def eval_all_metrics(
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    val_targets: List[List[int]],
    test_targets: List[List[int]]
) -> dict:
    """
    compute validation and test results
    args:
        val_scores: np.ndarray of shape (val_num, num_classes),
        test_scores: np.ndarray of shape (test_num, num_classes),
        val_targets: List[List[int]],
        test_targets: List[List[int]]
    """
    # get optimal threshold using val set
    multihot_targets = multihot(val_targets, 28)
    f1_dict = get_best_f1_scores(multihot_targets, val_scores)

    # get results using the threshold found
    multihot_targets = multihot(test_targets, 28)
    test_micro, test_samples, test_macro, test_none = compute_f1(multihot_targets, test_scores, f1_dict["threshold"])
    return {
        "val_micro": f1_dict["micro"], "val_samples": f1_dict["samples"],
        "val_macro": f1_dict["macro"], "val_none": f1_dict["none"],
        "test_micro": test_micro, "test_samples": test_samples,
        "test_macro": test_macro, "test_none": test_none,
    }


def get_best_f1_scores(
    multihot_targets: np.ndarray,
    scores: np.ndarray,
    threshold_end: float = 0.05,
    use_inference_strategy: bool = False,
    class_thresholds: np.ndarray = None
) -> Dict[str, float]:
    """
    get the optimal macro f1 score by tuning threshold
    If class_thresholds is provided, use class-level thresholds directly without searching.
    """
    if class_thresholds is not None:
        # 如果提供了类别级阈值，直接使用，跳过阈值搜索
        _micros, _samples, _macros, _none = compute_f1(
            multihot_targets, scores, threshold=0.5, 
            use_inference_strategy=use_inference_strategy, 
            class_thresholds=class_thresholds
        )
        f1 = {}
        f1["micro"] = _micros
        f1["macro"] = _macros
        f1["samples"] = _samples
        f1["none"] = _none
        # 对于类别级阈值，返回平均阈值作为参考
        f1["threshold"] = float(np.mean(class_thresholds))
        return f1
    
    # 原有逻辑：搜索最优单一阈值
    thrs = np.linspace(
        threshold_end, 0.95, int(np.round((0.95 - threshold_end) / 0.01)) + 1,
        endpoint=True
    )
    f1_micros = []
    f1_macros = []
    f1_samples = []
    f1_none = []
    for thr in thrs:
        _micros, _samples, _macros, _none = compute_f1(multihot_targets, scores, thr, use_inference_strategy)
        f1_micros.append(_micros)
        f1_samples.append(_samples)
        f1_macros.append(_macros)
        f1_none.append(_none)

    f1_macros_m = max(f1_macros)
    b_thr = np.argmax(f1_macros)
    # f1_micros_m = max(f1_micros)
    # b_thr = np.argmax(f1_micros)

    f1_micros_m = f1_micros[b_thr]
    # f1_macros_m = f1_macros[b_thr]
    f1_samples_m = f1_samples[b_thr]
    f1_none_m = f1_none[b_thr]
    f1 = {}
    f1["micro"] = f1_micros_m
    f1["macro"] = f1_macros_m
    f1["samples"] = f1_samples_m
    f1["threshold"] = thrs[b_thr]
    f1["none"] = f1_none_m
    
    return f1


def compute_f1(
        multihot_targets: np.ndarray, scores: np.ndarray, threshold: float = 0.5,
        use_inference_strategy: bool = False, class_thresholds: np.ndarray = None
) -> Tuple[float, float, float, np.ndarray]:
    # change scores to predict_labels
    if class_thresholds is not None:
        # 使用类别级阈值：每个类别使用自己的阈值
        # scores: [num_samples, num_classes], class_thresholds: [num_classes]
        # 广播机制：scores > class_thresholds
        predict_labels = scores > class_thresholds
    else:
        # 使用单一阈值（原有逻辑）
        predict_labels = scores > threshold
    # 只有当 use_inference_strategy=True 时才应用保底机制
    if use_inference_strategy:
        for i in range(predict_labels.shape[0]):
            if predict_labels[i].sum() == 0:
                max_idx = np.argmax(scores[i])
                predict_labels[i, max_idx] = 1.0
    predict_labels = predict_labels.astype(int)

    # get f1 scores
    f1 = {}
    f1["micro"] = f1_score(
        y_true=multihot_targets,
        y_pred=predict_labels,
        average="micro"
    )
    f1["samples"] = f1_score(
        y_true=multihot_targets,
        y_pred=predict_labels,
        average="samples"
    )
    f1["macro"] = f1_score(
        y_true=multihot_targets,
        y_pred=predict_labels,
        average="macro"
    )
    f1["none"] = f1_score(
        y_true=multihot_targets,
        y_pred=predict_labels,
        average=None
    )
    return f1["micro"], f1["samples"], f1["macro"], f1["none"]


def multihot(x: List[List[int]], nb_classes: int) -> np.ndarray:
    """transform to multihot encoding

    Arguments:
        x: list of multi-class integer labels, in the range
            [0, nb_classes-1]
        nb_classes: number of classes for the multi-hot vector

    Returns:
        multihot: multihot vector of type int, (num_samples, nb_classes)
    """
    num_samples = len(x)

    multihot = np.zeros((num_samples, nb_classes), dtype=np.int32)
    for idx, labs in enumerate(x):
        for lab in labs:
            multihot[idx, lab] = 1

    return multihot.astype(int)


def eval_validation_set(
    val_scores: np.ndarray,
    val_targets: np.ndarray,
    use_inference_strategy: bool = False,
    class_thresholds: np.ndarray = None
) -> dict:
    """
    compute validation results
    args:
        val_scores: np.ndarray of shape (val_num, num_classes),
        val_targets: np.ndarray of shape (val_num, num_classes) - multihot encoded targets
        use_inference_strategy: whether to use inference strategy (fallback mechanism for all-zero predictions)
        class_thresholds: optional np.ndarray of shape (num_classes,) - class-level thresholds to use directly
    """
    # get optimal threshold using val set
    multihot_targets = val_targets
    f1_dict = get_best_f1_scores(
        multihot_targets, val_scores, 
        use_inference_strategy=use_inference_strategy,
        class_thresholds=class_thresholds
    )

    # compute mAP
    mAP = compute_mAP(val_scores, multihot_targets)

    # get results using the threshold found
    return {
        "val_micro": f1_dict["micro"], 
        "val_samples": f1_dict["samples"],
        "val_macro": f1_dict["macro"], 
        "val_none": f1_dict["none"],
        "val_mAP": mAP,
        "threshold": f1_dict["threshold"],
    }

