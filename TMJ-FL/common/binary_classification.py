import torch

from sklearn.metrics import confusion_matrix
from typing import Tuple
import json
import numpy as np
import torch

class TestResultBinClassifcation: 
    def __init__(self, criterion:torch.nn.BCELoss=None):
        self.acc_loss = 0.0
        self.TN : int = 0
        self.FP : int = 0
        self.FN : int = 0
        self.TP : int = 0
        self.batches = 0
        self.criterion = criterion
        self.label_distribution = [0,0]
        
    def __str__(self):
        return f"Loss: {self.loss()}, Accuracy: {self.accuracy()} TN: {self.TN}, FP: {self.FP}, FN: {self.FN}, TP: {self.TP}"
    
    def as_dict(self) -> dict:
        return {
            "loss": self.loss(),
            "accuracy": self.accuracy(),
            "precision": self.precision(),
            "specificity": self.specificity(),
            "recall": self.recall(),
            "f1": self.f1(),
            # "batches_compared": self.batches,
            # "label_0": self.label_distribution[0],
            # "label_1": self.label_distribution[1],
        }
    
    def to_json(self) -> str:
        return json.dumps(self.as_dict())
    
    def tuple_dict(self) -> Tuple[str, float]:
        return ("accuracy", self.accuracy())
        
    def compare_batch(self, predicts:np.ndarray, labels:np.ndarray) -> None:
        if self.criterion is not None:
            self.acc_loss += self.criterion(predicts, labels).item()
            
        predicts = (predicts >= 0.5)
        labels = (labels >= 0.5)
        tn, fp, fn, tp = confusion_matrix(labels, predicts, labels=[0, 1]).ravel()

        # print(labels)
        # TODO: meget langsomt og dumt dette
        for l in labels:
            if l:
                self.label_distribution[1] += 1
            else:
                self.label_distribution[0] += 1
        
        self.TN += tn
        self.FP += fp
        self.FN += fn
        self.TP += tp
        self.batches += 1
    
    def accuracy(self) -> float:
        if self.batches == 0:
            return 0
        
        return float(self.TP + self.TN) / (float(self.TP + self.TN + self.FP + self.FN))
    
    def loss(self) -> float:
        if self.criterion is None or self.batches == 0:
            return 0
        
        return float(self.acc_loss) / float(self.batches)

    def precision(self) -> float:
        if self.TP + self.FP == 0:
            return 0
        return float(self.TP) / float(self.TP + self.FP)

    def recall(self) -> float:
        if self.TP + self.FN == 0:
            return 0
        return float(self.TP) / float(self.TP + self.FN)
    
    def specificity(self) -> float:
        if self.TN + self.FP == 0:
            return 0
        return float(self.TN) / float(self.TN + self.FP)
    
    def f1(self) -> float:
        if self.precision() + self.recall() == 0:
            return 0
        return 2 * (self.precision() * self.recall()) / (self.precision() + self.recall())
        
    