import sklearn
import numpy as np
import copy as cp
import sklearn.metrics

def get_performance_metrics(num_classes, preds, label_list, metrics=['acc','auprc','auroc'], rounding=4):
    '''
    num_classes: integer
    metrics: list of strings. subset of ['acc','auprc','auroc'] 
    '''

    results = dict()
    if 'auprc' in metrics or 'auroc' in metrics:
        if num_classes>2:
            Y_byclass = [np.array([[0,1] if row==l else [1,0] for row in label_list]) for l in range(num_classes)]
            pred_byclass = [cp.deepcopy(preds) for i in range(num_classes)]
            for j in range(num_classes):
                other = np.sum(pred_byclass[j][:,[i for i in range(num_classes) if i!=j]],axis=1)
                pred_byclass[j][:,1] = pred_byclass[j][:,j]
                pred_byclass[j][:,0] = other
                pred_byclass[j] = pred_byclass[j][:,[0,1]]
                
    for metric in metrics:
        if metric=='acc': results[metric] = round(sklearn.metrics.accuracy_score(label_list,np.argmax(preds,axis=1)),rounding)
        elif metric=='auprc': 
            if num_classes>2: results[metric] = round(np.mean([sklearn.metrics.average_precision_score(Y_byclass[cl],pred_byclass[cl]) for cl in range(num_classes)]),rounding)
            else: results[metric] = round(sklearn.metrics.average_precision_score(label_list,preds[:,1]),rounding)
        elif metric=='auroc':
            if num_classes>2: results[metric] = round(np.mean([sklearn.metrics.roc_auc_score(Y_byclass[cl],pred_byclass[cl]) for cl in range(num_classes)]),rounding)
            else: results[metric] = round(sklearn.metrics.roc_auc_score(label_list,preds[:,1]),rounding)
                
    return results
