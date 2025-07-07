# ML A1
#Section 3 - A RESNET
# Julyan van der Westhuizen - VWSJUL003

import torch
import torchvision
import torchvision.transforms as transforms
import ssl

# Bypass SSL isseus the OS might have
ssl._create_default_https_context = ssl._create_unverified_context

#region CREAING TRANSFORMS AND DOWNLOADING DATA
# Create the transform sequence
torch.manual_seed(1234)

transform = transforms.Compose([
    # Data augmentation
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-1, 1)),

    transforms.ToTensor(),  # Convert to Tensor
    # Normalize Image to [-1, 1] first number is mean, second is std deviation
    transforms.Normalize((0.5,), (0.5,)) 
])

# Load  dataset
# Train
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                      download=True, transform=transform)
# Test
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform)

# Send data to the data loaders
BATCH_SIZE = 200
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                          shuffle=False)
#endregion

#region SELECTING DEVICE, DEFINING TRAIN AND TEST METHODS
# Identify device
device = ("cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

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

#region CREATING RESNET and RESIDUAL BLOCK
import torch.nn as nn  # Layers
import torch.nn.functional as F # Activation Functions

# Define the RESNET architecture
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.stride != 1 or identity.shape[1] != out.shape[1]:
            identity = F.avg_pool2d(identity, kernel_size=self.stride, stride=self.stride)
            identity = torch.cat((identity, torch.zeros(identity.shape[0], out.shape[1] - identity.shape[1], identity.shape[2], identity.shape[3], device=identity.device)), dim=1)

        out += identity
        out = F.relu(out)

        return out

class RESNET(nn.Module):
    def __init__(self):
        super(RESNET, self).__init__()

        # Convolusion Design
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, stride=2)  # For pooling

        # Residual Block
        self.resblock1 = ResidualBlock(in_channels=16, out_channels=16)

        # MLP Design
        self.flatten = nn.Flatten() # For flattening the 2D image
        self.fc1 = nn.Linear(400, 120)  # First FC HL
        self.fc2= nn.Linear(120, 84) # Output layer
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Convolusion Design
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Residual Block
        x = self.resblock1(x)

        # MLP Design
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#endregion 

# TRAINING THE MODEL 

def trainModel():
    rsn = RESNET().to(device)

    LEARNING_RATE = 1e-1
    MOMENTUM = 0.9

    # Define the loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss() # Use this if not using softmax layer
    optimizer = optim.SGD(rsn.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # define the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Train the MLP for 5 epochs
    for epoch in range(15):
        train_loss = train(rsn, train_loader, criterion, optimizer, device)
        test_acc = test(rsn, test_loader, device)
        print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}")
        scheduler.step()

trainModel()


