import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import time


def load_data(batch_size=128):
    transform = transforms.Compose([
        transforms.Resize(32),  # Resizing to 32x32 (CIFAR-10 default size)
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalization
    ])

    # CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

# 2. Training and Evaluation Functions

def train(model, device, trainloader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    start_time = time.time()
    
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Compute accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i % 100 == 99:  # Print every 100 mini-batches
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

    accuracy = 100 * correct / total
    print(f'Epoch {epoch + 1} finished with accuracy: {accuracy:.2f}% in {time.time() - start_time:.2f}s')

def test(model, device, testloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test accuracy: {accuracy:.2f}%')

# 3. Main Training Loop

def main():
    # Device configuration (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    trainloader, testloader = load_data(batch_size=128)
    
    # Initialize the model, loss function, optimizer, and scheduler
    model = VisionTransformer(img_size=32, patch_size=4, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Learning rate scheduler

    num_epochs = 20

    # Training loop
    for epoch in range(num_epochs):
        train(model, device, trainloader, criterion, optimizer, epoch)
        test(model, device, testloader, criterion)
        scheduler.step()

    print('Finished Training')

    # Save the trained model
    torch.save(model.state_dict(), 'vit_cifar10.pth')
    print('Model saved as vit_cifar10.pth')

if __name__ == '__main__':
    main()
