import numpy as np
import datasets
from common import report as r

class ClientsPartitioner:
    def __init__(self, num_partitions: int, partition_method: str = "equal"):
        """The partition_splits should be a list of floats that sum to 1.0."""
        
        self.dataset = None
        self._num_partitions = num_partitions
        if partition_method == "equal":
            self._partition_splits = [1.0 / num_partitions] * num_partitions
        elif partition_method == "big-guy":
            self._partition_splits = [0.5] + [0.5 / (num_partitions - 1)] * (num_partitions - 1)
        elif partition_method == "random":
            self._partition_splits = np.random.dirichlet(np.ones(num_partitions), size=1)[0]
        else :
            raise ValueError("Unknown partition method: {}".format(partition_method))
            
        r.write_to_report("partition_splits", self._partition_splits)
    
    def load_partition(self, partition_id: int):
        """
        Returns the wanted partition of the dataset. The split is defined by partition_splits.
        """
        if partition_id < 0 or partition_id >= self._num_partitions:
            raise ValueError("partition_id must be between 0 and {}".format(self._num_partitions - 1))
        
        
        if self.dataset is None:
            raise ValueError("Dataset not set. Please load the dataset first.")
        if round(sum(self._partition_splits), 2) != 1.0:
            raise ValueError("Partition splits must sum to 1.0., got: {}".format(sum(self._partition_splits)))

        ds = self.dataset

        features = ds["features"]
        labels = ds["labels"]

        partition_splits = np.array(self._partition_splits)

        start = int(len(features) * sum(partition_splits[:partition_id]))
        end = int(len(features) * sum(partition_splits[:partition_id + 1]))
                
        features_partitioned = features[start:end]
        labels_partitioned = labels[start:end]

        return datasets.Dataset.from_dict({
            "features": features_partitioned,
            "labels": labels_partitioned,
        })