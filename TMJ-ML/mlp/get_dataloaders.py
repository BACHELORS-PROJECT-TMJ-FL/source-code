from ..get_data import splitSegmentations
import torch
import datasets
from torch.utils.data import DataLoader

# Define function for converting matrices to data loaders
def convertToDataloaders(trainSets, testSets):
    # Define collate function for making list into stacked pytorch tensor
    def collate_fn(batch):
        features = torch.tensor([item['features'] for item in batch])
        labels = torch.tensor([item['labels'] for item in batch])
        return {'features': features, 'labels': labels}

    trainloaders = []
    for trainset in trainSets:
        trainloaders.append(DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=collate_fn))

    testloaders = []
    for testset in testSets:
        testloaders.append(DataLoader(testset, batch_size=32, shuffle=True, collate_fn=collate_fn))
    return trainloaders, testloaders

# Define function for creating every necessary data loader for a segmentation
def createDataLoader(segmentation):
    trainsets = [
        datasets.Dataset.from_dict(
            {
                "features": segmentation["X_train_10p"],
                "labels": segmentation["y_train_10p"],
            }
        ),
        datasets.Dataset.from_dict(
            {
                "features": segmentation["X_train_50p"],
                "labels": segmentation["y_train_50p"],
            }
        ),
        datasets.Dataset.from_dict(
            {
                "features": segmentation["X_train_100p"],
                "labels": segmentation["y_train_100p"],
            }
        ),
    ]

    testSets = [
        datasets.Dataset.from_dict(
            {"features": segmentation["X_test_10p"], "labels": segmentation["y_test_10p"]}
        ),
        datasets.Dataset.from_dict(
            {"features": segmentation["X_test_50p"], "labels": segmentation["y_test_50p"]}
        ),
        datasets.Dataset.from_dict(
            {"features": segmentation["X_test_100p"], "labels": segmentation["y_test_100p"]}
        ),
    ]

    trainloaders, testloaders = convertToDataloaders(trainsets, testSets)
    trainloader10p = trainloaders[0]
    trainloader50p = trainloaders[1]
    trainloader100p = trainloaders[2]
    testloader10p = testloaders[0]
    testloader50p = testloaders[1]
    testloader100p = testloaders[2]

    return {
        "train_10p": trainloader10p,
        "train_50p": trainloader50p,
        "train_100p": trainloader100p,
        "test_10p": testloader10p,
        "test_50p": testloader50p,
        "test_100p": testloader100p,
    }

# Create segmentations as dataloaders
dataloaderSegmentations = list(map(lambda x: createDataLoader(x), splitSegmentations))