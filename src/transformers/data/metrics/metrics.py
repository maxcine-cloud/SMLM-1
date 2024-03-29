
import collections
import json
import logging
import math
import re
import string

from transformers.tokenization_bert import BasicTokenizer

import numpy as np
logger = logging.getLogger(__name__)

def calPRAUC(orderScores_index,TP_index, y_p_num, topN, save_plot=0, labels=None, probs=None):
    cumPRAUC = 0
    topNyhat = list()
    for i in range(topN):
        if orderScores_index[i] in TP_index:
            topNyhat.append(1)
        else:
            topNyhat.append(0)
    curSum = topNyhat[0]

    prevRecall = round(topNyhat[0] / y_p_num, 4)
    prevPrec = round(topNyhat[0], 4)
    recalist=[prevRecall]
    prelist=[prevPrec]
    auprlist=[0]
    threshold=[0]    
    for i in range(1, topN):
        curSum += topNyhat[i]
        recall = round(curSum / y_p_num, 4)
        prec   = round(curSum / (i+1), 4)
        cumPRAUC += ((recall - prevRecall) * (prevPrec + prec) / 2)
        prevRecall = recall
        prevPrec = prec
        recalist.append(recall)
        prelist.append(prec)
        threshold.append(i)
        auprlist.append(cumPRAUC)
    cumPRAUC = round(cumPRAUC, 4)
    pre=np.array(prelist)
    rec=np.array(recalist)
    f1 = 2*pre*rec/(pre + rec)
    f1[pre + rec == 0] = 0
    if save_plot:
        np.save('./output/metrics_res.npy',[recalist, prelist, auprlist, threshold, f1, cumPRAUC, labels, probs])
    return cumPRAUC

def evaluate_model(preds, labels, probs,last_eval,is_val):
    if is_val:
        topN = int(labels.sum()*1)
    else:
        topN = labels.sum()
    if topN==0:
        return 0,0,0,0,0,0
    p_rate = probs[labels == 1].sum()/probs.sum()
    avg_p = probs[labels == 1].mean()
    avg_u = probs[labels == 0].mean()
    
    y_p_Index=np.argwhere(labels == 1).flatten()
    orderScores_index = np.argsort(-probs)
    topNIndex = orderScores_index[:topN]
    TP_index = np.intersect1d(topNIndex, y_p_Index, assume_unique=True)
    recall = TP_index.shape[0] / y_p_Index.shape[0]
    precision = TP_index.shape[0] / topN    
    if last_eval:
        aupr = calPRAUC(orderScores_index, TP_index, y_p_Index.shape[0], topN, last_eval,labels, probs)        
    else:
        aupr = calPRAUC(orderScores_index, TP_index, y_p_Index.shape[0], topN)
    return aupr, avg_p, avg_u, p_rate, precision, recall