# ML A1
#Section 1 - A Standard Multi-layer Perceptron
# Julyan van der Westhuizen - VWSJUL003

import torch
import torchvision
import torchvision.transforms as transforms
import ssl

# HYPER PARAMETERS
BATCH_SIZE = 128

INPUT_SIZE = 32 * 32 * 3
FIRST_LAYER_SIZE = 2048
SECOND_LAYER_SIZE = 2048
OUTPUT_SIZE = 10

LEARNING_RATE = 1e-2
MOMENTUM = 0.9

# Bypass SSL isseus the OS might have
ssl._create_default_https_context = ssl._create_unverified_context

#region DATA DOWNLOADING AND TRANSFORM
# Define the transformation to apply to the data
torch.manual_seed(1234)

transform = transforms.Compose([
    
    # Data augmentation
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-5, 5)),

    transforms.ToTensor(),  # Convert to Tensor
    # Normalize Image to [-1, 1] first number is mean, second is std deviation
    transforms.Normalize((0.5,), (0.5,)) 
])

# Download and load the CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

# Create data loaders to iterate over the data
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False)
#endregion 

#region DEFINING MODEL
import torch.nn as nn  # Layers
import torch.nn.functional as F # Activation Functions

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten() # For flattening the 2D image
        self.fc1 = nn.Linear(INPUT_SIZE, FIRST_LAYER_SIZE)  
        self.fc2 = nn.Linear(FIRST_LAYER_SIZE, SECOND_LAYER_SIZE)  # First HL
        self.fc3= nn.Linear(SECOND_LAYER_SIZE, OUTPUT_SIZE) # Second HL
        self.output = nn.LogSoftmax(dim=1)

    def forward(self, x):
      x = self.flatten(x)
      x = F.relu(self.fc1(x))  # First Hidden Layer
      x = F.relu(self.fc2(x))  # Second Hidden Layer
      x = self.fc3(x)  # Output Layer
      x = self.output(x)  # For multi-class classification
      return x  

# Identify device
device = ("cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
#endregion

#region DEFINE TRAINING THE MODEL
import torch.optim as optim # Optimizers

# Define the training and testing functions
def train(net, train_loader, criterion, optimizer, device):
    net.train()  # Set model to training mode.
    running_loss = 0.0  # To calculate loss across the batches
    for data in train_loader:
        inputs, labels = data  # Get input and labels for batch
        inputs, labels = inputs.to(device), labels.to(device)  # Send to device
        optimizer.zero_grad()  # Zero out the gradients of the ntwork i.e. reset
        outputs = net(inputs)  # Get predictions
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Propagate loss backwards
        optimizer.step()  # Update weights
        running_loss += loss.item()  # Update loss
    return running_loss / len(train_loader)

def test(net, test_loader, device):
    net.eval()  # We are in evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Don't accumulate gradients
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) # Send to device
            outputs = net(inputs)  # Get predictions
            _, predicted = torch.max(outputs.data, 1)  # Get max value
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # How many are correct?
    return correct / total

#endregion

# TRAINING THE MODEL

def trainModel():
    mlp = MLP().to(device)
    # Define the loss function, optimizer, and learning rate scheduler
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(mlp.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # define the learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for epoch in range(15):
        train_loss = train(mlp, train_loader, criterion, optimizer, device)
        test_acc = test(mlp, test_loader, device)
        print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}")
        scheduler.step()

trainModel()
