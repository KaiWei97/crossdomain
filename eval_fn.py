import math
import numpy as np

def eval_score(pred_list, target_list, options):
    RC, MRR, NDCG = [], [], []
    pred_list = pred_list.argsort()
    for k in options:  # k:5
        RC.append(0)
        MRR.append(0)
        NDCG.append(0)
        temp_list = pred_list[:, -k:]
        search_index = 0
        while search_index < len(target_list):
            pos = np.argwhere(temp_list[search_index] == target_list[search_index])
            if len(pos) > 0:
                RC[-1] += 1
                MRR[-1] += 1 / (k - pos[0][0])
                NDCG[-1] += math.log(2) / math.log(k - pos[0][0] + 2)
            else:
                RC[-1] += 0
                MRR[-1] += 0
                NDCG[-1] += 0
            search_index += 1
    return RC, MRR, NDCG
