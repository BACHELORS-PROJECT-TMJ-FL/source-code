from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import datasets
import pandas as pd
import numpy as np
from PIL import Image

# ========================================
# MNIST
# ========================================


class MNISTPartitioner(IidPartitioner):
    """Partitioner for splitting MNIST into 5 centers."""

    def __init__(self):
        super().__init__(num_partitions=5)

    def load_partition(self, partition_id: int) -> datasets.Dataset:
        """
        Creates 5 partitions of the dataset:
            1. 0's and 1's
            2. 2's and 3's
            3. 4's and 5's
            4. 6's and 7's
            5. 8's and 9's
        """

        # Return the entire dataset if partition_id is -1
        if partition_id == -1:
            return self.dataset

        df = pd.DataFrame(
            {
                "image": [np.array(img) for img in self.dataset["image"]],
                "label": self.dataset["label"],
            }
        )

        df1 = df[df["label"].isin([0, 1])]
        df2 = df[df["label"].isin([2, 3])]
        df3 = df[df["label"].isin([4, 5])]
        df4 = df[df["label"].isin([6, 7])]
        df5 = df[df["label"].isin([8, 9])]

        def convertPDtoDS(df):
            return datasets.Dataset.from_dict(
                {
                    "image": [Image.fromarray(np.array(img)) for img in df["image"]],
                    "label": df["label"],
                }
            )

        splitDataset = [
            convertPDtoDS(df1),
            convertPDtoDS(df2),
            convertPDtoDS(df3),
            convertPDtoDS(df4),
            convertPDtoDS(df5),
        ]

        return splitDataset[partition_id]


trainPartitionerMNIST = None
testPartitionerMNIST = None


def load_data_mnist(partition_id: int, split: int, batch_size: int = 32):
    """Load partition MNIST data."""
    # Only initialize `FederatedDataset` once
    global trainPartitionerMNIST, testPartitionerMNIST
    if trainPartitionerMNIST is None or testPartitionerMNIST is None:
        trainPartitionerMNIST = MNISTPartitioner()
        testPartitionerMNIST = MNISTPartitioner()
        ds = datasets.load_dataset(path="ylecun/mnist")
        ds = datasets.concatenate_datasets([ds["train"], ds["test"]])
        trainPartitionerMNIST.dataset = datasets.concatenate_datasets(
            [ds.shard(5, i) for i in list(filter(lambda x: x != split, range(5)))]
        )
        testPartitionerMNIST.dataset = ds.shard(5, split)
    partition_train = trainPartitionerMNIST.load_partition(partition_id)
    partition_test = testPartitionerMNIST.load_partition(partition_id)

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["image"] = [
            (np.array(img, dtype=np.float32) / 256).flatten() for img in batch["image"]
        ]  # Transform images to float and normalize
        return batch

    partition_train = partition_train.with_transform(apply_transforms)
    partition_test = partition_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(partition_test, batch_size=batch_size)
    return trainloader, testloader


# ========================================
# FEMNIST
# ========================================


class FEMNISTPartitioner(IidPartitioner):
    """Partitioner for splitting MNIST into 5 centers."""

    def __init__(self):
        super().__init__(num_partitions=5)

    def load_partition(self, partition_id: int) -> datasets.Dataset:
        """
        Creates 5 partitions of the dataset:
            1. 0's and 1's from one person
            2. 2's and 3's from another person
            3. etc.
            4. etc.
            5. etc.
        """

        # Return the entire dataset if partition_id is -1
        if partition_id == -1:
            return self.dataset

        df = pd.DataFrame(
            {
                "image": [np.array(img) for img in self.dataset["image"]],
                "writer_id": self.dataset["writer_id"],
                "label": self.dataset["label"],
            }
        )

        # Create a dataframe only containing one writer_id 
        df1 = df[df["writer_id"] == "f0421_33"]
        df2 = df[df["writer_id"] == "f0611_34"]
        df3 = df[df["writer_id"] == "f0667_05"]
        df4 = df[df["writer_id"] == "f0317_44"]
        df5 = df[df["writer_id"] == "f0191_22"]

        splitDataset = [
            convertPDtoDSFEMNIST(df1),
            convertPDtoDSFEMNIST(df2),
            convertPDtoDSFEMNIST(df3),
            convertPDtoDSFEMNIST(df4),
            convertPDtoDSFEMNIST(df5),
        ]

        if len(splitDataset[partition_id]) == 0:
            raise ValueError(
                f"No datapoints in ds"
            )

        return splitDataset[partition_id]


def convertPDtoDSFEMNIST(df):
    return datasets.Dataset.from_dict(
        {
            "image": [Image.fromarray(np.array(img)) for img in df["image"]],
            "writer_id": df["writer_id"],
            "label": df["label"],
        }
    )


trainPartitionerFEMNIST = None
testPartitionerFEMNIST = None


def load_data_femnist(partition_id: int, split: int, batch_size: int = 32):
    """Load partition MNIST data."""
    # Only initialize `FederatedDataset` once
    global trainPartitionerFEMNIST, testPartitionerFEMNIST
    if trainPartitionerFEMNIST is None or testPartitionerFEMNIST is None:
        trainPartitionerFEMNIST = FEMNISTPartitioner()
        testPartitionerFEMNIST = FEMNISTPartitioner()
        ds = datasets.load_dataset(path="flwrlabs/femnist")
        ds = ds["train"]
        ds = ds.shuffle(seed=42)
        # Only keep 0's and 1's
        df = pd.DataFrame(
            {
                "image": [np.array(img) for img in ds["image"]],
                "writer_id": ds["writer_id"],
                "label": ds["character"],
            }
        )
        df = df[df["label"].isin([0, 1])]
        ds = convertPDtoDSFEMNIST(df)
        trainPartitionerFEMNIST.dataset = datasets.concatenate_datasets(
            [ds.shard(5, i) for i in list(filter(lambda x: x != split, range(5)))]
        )
        testPartitionerFEMNIST.dataset = ds.shard(5, split)

    partition_train = trainPartitionerFEMNIST.load_partition(partition_id)
    partition_test = testPartitionerFEMNIST.load_partition(partition_id)

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["image"] = [
            (np.array(img, dtype=np.float32) / 256).flatten() for img in batch["image"]
        ]  # Transform images to float and normalize
        return batch
    
    partition_train = partition_train.with_transform(apply_transforms)
    partition_test = partition_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(partition_test, batch_size=batch_size)
    return trainloader, testloader


def load_data(
    partition_id: int, split: int, batch_size: int = 32, dataset_name: str = "MNIST"
):
    if dataset_name == "MNIST":
        return load_data_mnist(partition_id, split, batch_size)
    elif dataset_name == "FEMNIST":
        return load_data_femnist(partition_id, split, batch_size)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

