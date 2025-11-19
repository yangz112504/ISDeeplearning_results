import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class MLP(nn.Module):
    def __init__(self, act_fun):
        super().__init__()
        self.fc1 = nn.Linear(32*32*3, 512)  
        self.fc2 = nn.Linear(512, 256)     
        self.fc3 = nn.Linear(256, 128)      
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)       # output layer (10 classes)
        self.act_fun = act_fun

    def forward(self, x):
        x = torch.flatten(x, 1)        # flatten each image to [batch, 3072]
        if self.act_fun == "relu":
            x = F.relu(self.fc1(x))        # hidden layer 1 with ReLU activation
            x = F.relu(self.fc2(x))        # hidden layer 2 with ReLU
            x = F.relu(self.fc3(x))        # hidden layer 3 with ReLU
            x = F.relu(self.fc4(x))
            x = self.fc5(x)                # output layer (no activation, raw logits)
        if self.act_fun == "gelu":
            x = F.gelu(self.fc1(x))        # hidden layer 1 with ReLU activation
            x = F.gelu(self.fc2(x))        # hidden layer 2 with ReLU
            x = F.gelu(self.fc3(x))        # hidden layer 3 with ReLU
            x = F.gelu(self.fc4(x))
            x = self.fc5(x)  
        if self.act_fun == "silu":
            x = F.silu(self.fc1(x))        # hidden layer 1 with ReLU activation
            x = F.silu(self.fc2(x))        # hidden layer 2 with ReLU
            x = F.silu(self.fc3(x))        # hidden layer 3 with ReLU
            x = F.silu(self.fc4(x))
            x = self.fc5(x) 
        return x
    

model_relu = MLP("relu")
model_gelu = MLP("gelu")
model_silu = MLP("silu")

model_silu.load_state_dict(model_relu.state_dict())
model_gelu.load_state_dict(model_relu.state_dict())

def train_net(model, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    epoch_losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)   # Calculates how far off the predictions are from the true labels using a loss function.
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if(i % 2000 == 1999):    # print every 2000 mini-batches
                print(f"[Epoch {epoch + 1}, Batch {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0
        
        epoch_loss = running_loss / len(trainloader)
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    print('Finished Training')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    return epoch_losses, accuracy


loss_relu, acc_relu = train_net(model_relu, epochs=50)
loss_gelu, acc_gelu = train_net(model_gelu, epochs=50)
loss_silu, acc_silu = train_net(model_silu, epochs=50)

# Plot loss curves
plt.figure(figsize=(10,6))
plt.plot(loss_relu, label=f"ReLU (Acc {acc_relu:.2f}%)")
plt.plot(loss_gelu, label=f"GELU (Acc {acc_gelu:.2f}%)")
plt.plot(loss_silu, label=f"SiLU (Acc {acc_silu:.2f}%)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curves (MLP on CIFAR-10)")
plt.legend()
plt.show()

# Model accuracies
print(f"ReLU Model Accuracy: {acc_relu:.2f}%")
print(f"GELU Model Accuracy: {acc_gelu:.2f}%")
print(f"SiLU Model Accuracy: {acc_silu:.2f}%")


#CIFAR
#change activation function between relu, gelu, siu
#check loss per epoch
#aim for 40% 50% accuracy with MLP
#instead of use CNN, use the MLP instead of the CNN in this so dont copy and paste exactly  

# functions to show an image
