import torch
import torchvision 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from transformers import ViTForImageClassification
import numpy as np



torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

train_size = int(0.8 * len(full_trainset))
val_size = len(full_trainset) - train_size
trainset, valset = random_split(full_trainset, [train_size, val_size])

trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
valloader = DataLoader(valset, batch_size=32, shuffle=False)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                     download=True, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False)


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
model.classifier = torch.nn.Linear(model.config.hidden_size, 10)
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 5

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0 
    total = 0
    running_loss = 0.0
    with torch.no_grad():
      for data in dataloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100*correct / total
    avg_loss = running_loss / len(dataloader)

    return accuracy, avg_loss

def train():
  best_val_acc, patience, patience_counter = 0, 3, 0

  for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
      inputs, labels = data[0].to(device), data[1].to(device)

      optimizer.zero_grad()
      outputs = model(inputs).logits
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      if i%100 == 99:
        print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
        running_loss = 0.0

    val_acc, val_loss = evaluate_model(model, valloader)
    print(f'Epoch {epoch + 1}:')
    print(f'Validation Accuracy: {val_acc:.2f}%')
    print(f'Validation Loss: {val_loss:.3f}')
    
    if val_acc > best_val_acc:
      best_val_acc = val_acc
      patience_counter = 0
      torch.save(model.state_dict(), 'best_model_imgclassification.pth')
    else:
      patience_counter += 1
      if patience_counter >= patience:
        break

if __name__=="__main__":
  print("Training Started....")
  train()

  model.load_state_dict(torch.load('best_model_imgclassification.pth'))
  test_acc, test_loss = evaluate_model(model, testloader)
  print(f'Test Accuracy: {test_acc:.2f}%')
  print(f'Test Loss: {test_loss:.3f}')

