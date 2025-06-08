import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Define training function for single epoch for neural network
def train(net, trainloader, lr, device):
    """Train the net on the training set."""
    net.to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    net.train()
    epoch_loss = 0.0
    for batch in trainloader:
        data, targets = batch['features'], batch['labels']
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = net(data)[:, 0]

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(trainloader)
    
# Define test function for neural network
def test(net, testloader, device):
    """Validate the net on the test set."""
    net.to(device)
    criterion = torch.nn.BCELoss()
    correct, loss = 0, 0.0 
    all_outputs = np.array([]) [:, np.newaxis]
    all_labels = np.array([]) [:, np.newaxis]
    with torch.no_grad():
        for batch in testloader:
            data = batch["features"]
            labels = batch["labels"] [:, torch.newaxis]
            outputs = net(data.to(device))
            all_outputs = np.vstack([all_outputs, outputs.cpu().numpy()])
            all_labels = np.vstack([all_labels, labels.cpu().numpy()])
            loss += criterion(outputs, labels.to(device)).item()
    loss = loss / len(testloader)
    accuracy = accuracy_score(all_labels, all_outputs > 0.5)
    precision = precision_score(all_labels, all_outputs > 0.5, average='binary')
    recall = recall_score(all_labels, all_outputs > 0.5, average='binary')
    f1 = f1_score(all_labels, all_outputs > 0.5, average='binary')
    
    return loss, accuracy, precision, recall, f1

# Function for training and evaluating the network
def trainAndEvaluateNetwork(net, epochs, trainloader, testloader, lr, device):
    trainloss = []
    testloss = []
    testaccuracy = []
    testprecision = []
    testrecall = []
    testf1 = []
    
    # Get initial loss and accuracy on all test sets
    inittestLoss, initaccuracy, precision, recall, f1 = test(net, testloader, device)
    print(f"Initial - Test Loss: {inittestLoss}, Accuracy: {initaccuracy}")
    testloss.append(inittestLoss)
    testaccuracy.append(initaccuracy)
    testprecision.append(precision)
    testrecall.append(recall)
    testf1.append(f1)

    # get initial train loss
    initTrainLoss, _, _, _, _ = test(net, trainloader, device)
    trainloss.append(initTrainLoss)
    
    # Run thrugh the given amount of epochs
    for epoch in range(epochs):
        trainingLoss = train(net, trainloader, lr, device)
        print(f"Epoch {epoch+1}/{epochs} - Trainloss: {trainingLoss:.4f}")
        trainloss.append(trainingLoss)
        
        # Evaluate the updated model on the test sets
        testLoss, accuracy, precision, recall, f1 = test(net, testloader, device)
        print(f"Test Loss: {testLoss}, Accuracy: {accuracy}")
        testloss.append(testLoss)
        testaccuracy.append(accuracy)
        testprecision.append(precision)
        testrecall.append(recall)
        testf1.append(f1)
    
    return trainloss, testloss, testaccuracy, testrecall, testprecision, testf1 
    