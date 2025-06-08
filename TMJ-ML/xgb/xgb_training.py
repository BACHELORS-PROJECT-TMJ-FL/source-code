import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def extract_test_error(s):
    prefix = "Test-error:"
    try:
        start = s.index(prefix) + len(prefix)
        return float(s[start:])
    except (ValueError, IndexError):
        return None

def createAndTrainModel(dmatrix_train, test_dmatrix, params, num_boost_round=50):
    labels = test_dmatrix.get_label()
    bst = xgb.train(
        params,
        dmatrix_train,
        num_boost_round=1
    )
    outputs = np.array(bst.predict(test_dmatrix) > 0.5, dtype=float)
    accuracy = [accuracy_score(labels, outputs)]
    precision = [precision_score(labels, outputs)]
    recall = [recall_score(labels, outputs)]
    auc = [roc_auc_score(labels, outputs)]
    f1 = [f1_score(labels, outputs)]
    for i in range(1, num_boost_round):
        bst.update(dmatrix_train, i)
        outputs = np.array(bst.predict(test_dmatrix) > 0.5, dtype=float)
        accuracy.append(accuracy_score(labels, outputs))
        precision.append(precision_score(labels, outputs))
        recall.append(recall_score(labels, outputs))
        f1.append(f1_score(labels, outputs))
        auc.append(roc_auc_score(labels, outputs))
    return accuracy, precision, recall, f1, auc