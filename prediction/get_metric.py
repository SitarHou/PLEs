from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix,matthews_corrcoef,roc_auc_score, recall_score, precision_score
from sklearn.metrics import average_precision_score
from sklearn.utils import resample
from sklearn.metrics import cohen_kappa_score
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, type, classes, dirpath,title='Confusion matrix',cmap=plt.cm.Blues):
    """
       plot confusion matrix

       Parameters
       ----------
       type: train,valid,test
       classes: default [0,1]
       dirpath: cm save path
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(i, j, cm[j, i],
                     horizontalalignment="center",
                     color="white" if cm[j, i] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    path = dirpath + str(type) + '_con.png'
    plt.savefig(path, dpi=500, bbox_inches='tight', transparent=True)
    plt.show()


def bootstrap_metric_multi(metric, average, y_test, y_pred, n_bootstraps=1000, alpha=0.95):
    """
       confidence interval

       Parameters
       ----------
       metric: acc, f1, recall, precision
       average: weighted, macro, micro
    """

    bootstrapped_scores = []
    rng = np.random.RandomState(42)

    for _ in range(n_bootstraps):
        # Bootstrap
        indices = resample(np.arange(len(y_test)), random_state=rng)
        if len(np.unique(y_test[indices])) == 2:
            if metric == 'acc':
                score = accuracy_score(y_test[indices], y_pred[indices])
            elif metric == 'f1':
                score = f1_score(y_test[indices], y_pred[indices], average=average)
            elif metric == 'recall':
                score = recall_score(y_test[indices], y_pred[indices], average=average)
            elif metric == 'precision':
                score = precision_score(y_test[indices], y_pred[indices],  average=average)
            elif metric =='kappa':
                score = cohen_kappa_score(y_test[indices], y_pred[indices])

            bootstrapped_scores.append(score)

    if metric == 'acc':
        index = accuracy_score(y_test, y_pred)
    elif metric == 'f1':
        index = f1_score(y_test, y_pred,  average=average)
    elif metric == 'recall':
        index = recall_score(y_test, y_pred,  average=average)
    elif metric == 'precision':
        index = precision_score(y_test, y_pred,  average=average)
    elif metric == 'kappa':
        index = cohen_kappa_score(y_test, y_pred)


    # confidence interval
    sorted_scores = np.sort(bootstrapped_scores)

    lower_bound = np.percentile(sorted_scores, (1 - alpha) / 2 * 100)
    upper_bound = np.percentile(sorted_scores, (1 + alpha) / 2 * 100)

    return lower_bound, upper_bound, index

def bootstrap_metric_binary(metric, y_test, y_pred, y_pred_prob, n_bootstraps=1000, alpha=0.95):
    """
       confidence interval

       Parameters
       ----------
       metric: auc, acc, f1, mcc, recall, precision, pr
    """

    bootstrapped_scores = []
    rng = np.random.RandomState(42)

    for _ in range(n_bootstraps):
        # Bootstrap
        indices = resample(np.arange(len(y_test)), random_state=rng)
        if len(np.unique(y_test[indices])) == 2:
            if metric == 'auc':
                score = roc_auc_score(y_test[indices],  y_pred_prob[indices])
            elif metric == 'acc':
                score = accuracy_score(y_test[indices], y_pred[indices])
            elif metric == 'f1':
                score = f1_score(y_test[indices], y_pred[indices])
            elif metric == 'mcc':
                score = matthews_corrcoef(y_test[indices], y_pred[indices])
            elif metric == 'recall':
                score = recall_score(y_test[indices], y_pred[indices])
            elif metric == 'precision':
                score = precision_score(y_test[indices], y_pred[indices])
            elif metric =='pr':
                score = average_precision_score(y_test[indices], y_pred_prob[indices])
            elif metric =='kappa':
                score = cohen_kappa_score(y_test[indices], y_pred[indices])

            bootstrapped_scores.append(score)

    if metric == 'auc':
        index = roc_auc_score(y_test, y_pred_prob)
    elif metric == 'acc':
        index = accuracy_score(y_test, y_pred)
    elif metric == 'f1':
        index = f1_score(y_test, y_pred)
    elif metric == 'mcc':
        index = matthews_corrcoef(y_test, y_pred)
    elif metric == 'recall':
        index = recall_score(y_test, y_pred)
    elif metric == 'precision':
        index = precision_score(y_test, y_pred)
    elif metric == 'pr':
        index = average_precision_score(y_test, y_pred_prob)
    elif metric == 'kappa':
        index = cohen_kappa_score(y_test, y_pred)

    # confidence interval
    sorted_scores = np.sort(bootstrapped_scores)

    lower_bound = np.percentile(sorted_scores, (1 - alpha) / 2 * 100)
    upper_bound = np.percentile(sorted_scores, (1 + alpha) / 2 * 100)

    return lower_bound, upper_bound, index

def C_interval_multi(y_test, y_pred, y_pred_prob, dirpath, n_bootstraps=1000, alpha=0.95):
    recall_ci_weighted = bootstrap_metric_multi('recall','weighted', y_test, y_pred, n_bootstraps,alpha)
    precision_ci_weighted = bootstrap_metric_multi('precision','weighted', y_test, y_pred,n_bootstraps,alpha)
    f1_ci_weighted = bootstrap_metric_multi('f1','weighted', y_test, y_pred, n_bootstraps, alpha)

    recall_ci_macro = bootstrap_metric_multi('recall', 'macro', y_test, y_pred, n_bootstraps, alpha)
    precision_ci_macro = bootstrap_metric_multi('precision', 'macro', y_test, y_pred, n_bootstraps, alpha)
    f1_ci_macro = bootstrap_metric_multi('f1', 'macro', y_test, y_pred,  n_bootstraps, alpha)

    recall_ci_micro = bootstrap_metric_multi('recall', 'micro', y_test, y_pred, n_bootstraps, alpha)
    precision_ci_micro = bootstrap_metric_multi('precision', 'micro', y_test, y_pred, n_bootstraps, alpha)
    f1_ci_micro = bootstrap_metric_multi('f1', 'micro', y_test, y_pred, n_bootstraps, alpha)

    acc_ci = bootstrap_metric_multi('acc','weighted', y_test, y_pred,  n_bootstraps, alpha)
    kappa_ci = bootstrap_metric_multi('kappa','weighted', y_test, y_pred, n_bootstraps, alpha)
    path = dirpath + '/metric_confidence_interval.txt'

    with open(path,'w') as f:

        f.write(f'acc_ci: {acc_ci[0]:.3f}, {acc_ci[1]:.3f}, {acc_ci[2]:.3f}\n')
        f.write(f'kappa_ci: {kappa_ci[0]:.3f}, {kappa_ci[1]:.3f}, {kappa_ci[2]:.3f}\n')
        f.write(f'f1_ci_weighted: {f1_ci_weighted[0]:.3f}, {f1_ci_weighted[1]:.3f}, {f1_ci_weighted[2]:.3f}\n')
        f.write(f'recall_ci_weighted: {recall_ci_weighted[0]:.3f}, {recall_ci_weighted[1]:.3f}, {recall_ci_weighted[2]:.3f}\n')
        f.write(f'precision_ci_weighted: {precision_ci_weighted[0]:.3f}, {precision_ci_weighted[1]:.3f}, {precision_ci_weighted[2]:.3f}\n')

        f.write(f'f1_ci_macro: {f1_ci_macro[0]:.3f}, {f1_ci_macro[1]:.3f}, {f1_ci_macro[2]:.3f}\n')
        f.write(f'recall_ci_macro: {recall_ci_macro[0]:.3f}, {recall_ci_macro[1]:.3f}, {recall_ci_macro[2]:.3f}\n')
        f.write(f'precision_ci_macro: {precision_ci_macro[0]:.3f}, {precision_ci_macro[1]:.3f}, {precision_ci_macro[2]:.3f}\n')

        f.write(f'f1_ci_micro: {f1_ci_micro[0]:.3f}, {f1_ci_micro[1]:.3f}, {f1_ci_micro[2]:.3f}\n')
        f.write(f'recall_ci_micro: {recall_ci_micro[0]:.3f}, {recall_ci_micro[1]:.3f}, {recall_ci_micro[2]:.3f}\n')
        f.write(f'precision_ci_micro: {precision_ci_micro[0]:.3f}, {precision_ci_micro[1]:.3f}, {precision_ci_micro[2]:.3f}\n')

        f.close
    return 0

def C_interval_binary(y_test, y_pred, y_pred_prob, dirpath, n_bootstraps=1000, alpha=0.95):

    auc_ci = bootstrap_metric_binary('auc', y_test, y_pred, y_pred_prob,n_bootstraps, alpha)
    mcc_ci = bootstrap_metric_binary('mcc', y_test, y_pred, y_pred_prob,n_bootstraps, alpha)
    recall_ci = bootstrap_metric_binary('recall', y_test, y_pred, y_pred_prob,n_bootstraps, alpha)
    precision_ci = bootstrap_metric_binary('precision', y_test, y_pred, y_pred_prob,n_bootstraps, alpha)
    pr_ci = bootstrap_metric_binary('pr', y_test, y_pred, y_pred_prob,n_bootstraps, alpha)
    f1_ci = bootstrap_metric_binary('f1', y_test, y_pred, y_pred_prob, n_bootstraps, alpha)
    acc_ci = bootstrap_metric_binary('acc', y_test, y_pred, y_pred_prob, n_bootstraps, alpha)
    kappa_ci = bootstrap_metric_binary('kappa', y_test, y_pred, y_pred_prob, n_bootstraps, alpha)
    path = dirpath + '/C_interval.txt'

    with open(path,'w') as f:

        f.write(f'auc_ci: {auc_ci[0]:.3f}, {auc_ci[1]:.3f}, {auc_ci[2]:.3f}\n')
        f.write(f'kappa_ci: {kappa_ci[0]:.3f}, {kappa_ci[1]:.3f}, {kappa_ci[2]:.3f}\n')
        f.write(f'mcc_ci: {mcc_ci[0]:.3f}, {mcc_ci[1]:.3f}, {mcc_ci[2]:.3f}\n')
        f.write(f'recall_ci: {recall_ci[0]:.3f}, {recall_ci[1]:.3f}, {recall_ci[2]:.3f}\n')
        f.write(f'precision_ci: {precision_ci[0]:.3f}, {precision_ci[1]:.3f}, {precision_ci[2]:.3f}\n')
        f.write(f'pr_ci: {pr_ci[0]:.3f}, {pr_ci[1]:.3f}, {pr_ci[2]:.3f}\n')
        f.write(f'f1_ci: {f1_ci[0]:.3f}, {f1_ci[1]:.3f}, {f1_ci[2]:.3f}\n')
        f.write(f'acc_ci: {acc_ci[0]:.3f}, {acc_ci[1]:.3f}, {acc_ci[2]:.3f}\n')

        f.close
    return 0

def model_metric(X_test, y_test,y_pred, pr,  model, feature_columns, dirpath, type, class_names = [0, 1]):
    """
      get all model result

      Parameters
      ----------
      X_test,y_test/X_valid,y_valid
      model:fit model
      feature_columns: X is numpy ,pass into plot shap to draw the pic
      dirpath: the path to save result
      type: train, valid, test
      classes: default [0,1]
       """
    #predict_y_test
    #y_pred = model.predict(X_test)
    #pr = model.predict_proba(X_test)
    #print metric: class report

    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)

    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    report = classification_report(y_test, y_pred, target_names=class_names)

    print(report)
    with open(dirpath + 'classification_report.txt', 'w') as f:
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, type, class_names, dirpath, title='Confusion matrix', cmap=plt.cm.Blues)
    
    #save C_interval of mcc auc pr recall precision
    if y_test.nunique()>2:
        C_interval_multi(y_test, y_pred, pr, dirpath, n_bootstraps=1000, alpha=0.95)
        #plot_shap(model, X_test, feature_columns, dirpath,type)
    else:
        if pr.ndim >1:
            C_interval_binary(y_test, y_pred, pr[:,1], dirpath,  n_bootstraps=1000, alpha=0.95)
        else:
            C_interval_binary(y_test, y_pred, pr, dirpath, n_bootstraps=1000, alpha=0.95)

    return y_test,y_pred,pr
