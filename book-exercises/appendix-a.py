import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# creating pytorch tensors

tensor0d = torch.tensor(1)
tensor1d = torch.tensor([1, 2, 3])
tensor2d = torch.tensor([[1, 2, 3],
                         [4, 5, 6]])
tensor3d = torch.tensor([[[1, 2], [3, 4]],
                         [[5, 6], [7, 8]]])


# intro to tensor types and conversions

# prints torch.int64
print(tensor1d.dtype)
floatvec = torch.tensor([1.0, 2.0, 3.0])
# prints torch.float32
print(floatvec.dtype)
floatvec = tensor1d.to(torch.float32)
# prints torch.float32
print(floatvec.dtype)
# prints tensor([[1, 2, 3],
#                [4, 5, 6]])
print(tensor2d)
# prints torch.Size([2, 3])
print(tensor2d.shape)
# prints tensor([[1, 2],
#                [3, 4],
#                [5, 6]])
print(tensor2d.reshape(3, 2))
# prints tensor([[1, 2],
#                [3, 4],
#                [5, 6]])
print(tensor2d.view(3, 2))
# prints tensor([[1, 2],
#                [3, 4],
#                [5, 6]])
print(tensor2d.T)
# prints tensor([[14, 32],
#                [32, 77]])
print(tensor2d.matmul(tensor2d.T))
# prints tensor([[14, 32],
#                [32, 77]])
print(tensor2d @ tensor2d.T)

# computational graphs

y = torch.tensor([1.0]) # true label
x1 = torch.tensor([1.1]) # input feature
w1 = torch.tensor([2.2], requires_grad=True) # weight parameter
b = torch.tensor([0.0], requires_grad=True) # bias unit
z = x1 * w1 + b # net input
a = torch.sigmoid(z) # activation and output
loss = F.binary_cross_entropy(a, y)
grad_L_w1 = grad(loss, w1, retain_graph=True)
grad_L_b = grad(loss, b, retain_graph=True)

# prints (tensor([-0.0898]),)
print(grad_L_w1)
# prints (tensor([-0.0817]),)
print(grad_L_b)

loss.backward() # syntactic sugar for above process of computing the gradients of leaf nodes of the graph and storing in `.grad`
# prints tensor([-0.0898])
print(w1.grad)
# prints tensor([-0.0817])
print(b.grad)

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_imputs, num_outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_imputs, 30),
            torch.nn.ReLU(), # non-linear activation functions are placed between the hidden layers

            # 2nd hidden layer
            torch.nn.Linear(30, 20), # number of output nodes of one output layer has to match the number of inputs in the next layer
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, num_outputs)
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits # outputs of the last layer are called logits

model = NeuralNetwork(50, 3)
# prints NeuralNetwork(
#         (layers): Sequential(
#            (0): Linear(in_features=50, out_features=30, bias=True)
#            (1): ReLU()
#            (2): Linear(in_features=30, out_features=20, bias=True)
#            (3): ReLU()
#            (4): Linear(in_features=20, out_features=3, bias=True)
#          )
#       )
print(model)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # each parameter for which `requires_grad=True` counts as a trainable parameter and will be updated during training
# prints Total number of trainable model parameters: 2213
print("Total number of trainable model parameters:", num_params)
# prints Parameter containing:
#        tensor([[-0.0933,  0.0919,  0.0443,  ...,  0.0254, -0.0993,  0.0926],
#                [ 0.0856, -0.0370, -0.0238,  ...,  0.0526, -0.0084,  0.1058],
#                [ 0.0044, -0.1190,  0.1152,  ..., -0.1342,  0.0319, -0.0674],
#                ...,
#                [-0.1349, -0.0294, -0.0006,  ...,  0.0070, -0.0830,  0.0808],
#                [-0.1034,  0.0821, -0.0271,  ...,  0.0918, -0.0268, -0.0440],
#                [-0.1166, -0.0403, -0.1321,  ...,  0.0298, -0.1393, -0.1158]],
#               requires_grad=True)
print(model.layers[0].weight)
# prints torch.Size([30, 50])
print(model.layers[0].weight.shape)
# prints Parameter containing:
#        tensor([-0.0014,  0.0131,  0.0502, -0.1295,  0.1140, -0.0187,  0.1392, -0.0768,
#                -0.0024,  0.1245,  0.0730, -0.0296,  0.0786, -0.0618,  0.0710,  0.1131,
#                -0.0678,  0.1010,  0.0292,  0.0313,  0.0298, -0.0462, -0.1369,  0.1205,
#                 0.0777, -0.0068, -0.1413, -0.0177,  0.0994, -0.1301],
#               requires_grad=True)
print(model.layers[0].bias)

torch.manual_seed(123) # use a seed to create reproducable results
model = NeuralNetwork(50, 3)
# prints Parameter containing:
#        tensor([[-0.0577,  0.0047, -0.0702,  ...,  0.0222,  0.1260,  0.0865],
#                [ 0.0502,  0.0307,  0.0333,  ...,  0.0951,  0.1134, -0.0297],
#                [ 0.1077, -0.1108,  0.0122,  ...,  0.0108, -0.1049, -0.1063],
#                ...,
#                [-0.0787,  0.1259,  0.0803,  ...,  0.1218,  0.1303, -0.1351],
#                [ 0.1359,  0.0175, -0.0673,  ...,  0.0674,  0.0676,  0.1058],
#                [ 0.0790,  0.1343, -0.0293,  ...,  0.0344, -0.0971, -0.0509]],
#               requires_grad=True)
print(model.layers[0].weight)

X = torch.rand((1, 50))
out = model(X)
# prints tensor([[-0.1670,  0.1001, -0.1219]], grad_fn=<AddmmBackward0>)
print(out)

with torch.no_grad():
    out = model(X)
# prints tensor([[-0.1670,  0.1001, -0.1219]])
print(out)

with torch.no_grad():
    out = torch.softmax(model(X), dim=1)
# prints tensor([[0.2983, 0.3896, 0.3121]])
print(out)

X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])
y_train = torch.tensor([0, 0, 0, 1, 1])
X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])
y_test = torch.tensor([0, 1])

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y 

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y # instructions for rertrieving exactly one data record and the corresponding label
    
    def __len__(self):
        return self.labels.shape[0] # instructions for returning the total length of the dataset

train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)
# prints 5
print(len(train_ds))

train_loader = DataLoader(
    dataset=train_ds, # the ToyDataset serves as input to the data loader
    batch_size=2,
    shuffle=True,
    num_workers=0 # number of background processes
)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False, # it's not necessary to shuffle a test dataset
    num_workers=0
)

# prints Batch 1: tensor([[ 2.7000, -1.5000],
#                [ 2.3000, -1.1000]]) tensor([1, 1])
#        Batch 2: tensor([[-0.9000,  2.9000],
#                [-1.2000,  3.1000]]) tensor([0, 0])
#        Batch 3: tensor([[-0.5000,  2.6000]]) tensor([0])
for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}:", x, y)

train_loader = DataLoader(
    dataset=train_ds, # the ToyDataset serves as input to the data loader
    batch_size=2,
    shuffle=True,
    num_workers=0, # number of background processes
    drop_last=True # drop uneven data batches
)

# prints Batch 1: tensor([[ 2.3000, -1.1000],
#                [-1.2000,  3.1000]]) tensor([1, 0])
#        Batch 2: tensor([[-0.9000,  2.9000],
#                [ 2.7000, -1.5000]]) tensor([0, 1])
for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}:", x, y)

# a typical training loop

model = NeuralNetwork(num_imputs=2, num_outputs=2) # this dataset has two features and two classes
optimizer = torch.optim.SGD(
    model.parameters(),  # stochiastic gradient descent
    lr=0.5 # learning rate
)
num_epochs = 3

# prints:
    # Epoch: 001/003 | Batch 001/002 | Train Loss: 0.47
    # Epoch: 001/003 | Batch 002/002 | Train Loss: 1.08
    # Epoch: 002/003 | Batch 001/002 | Train Loss: 0.35
    # Epoch: 002/003 | Batch 002/002 | Train Loss: 0.08
    # Epoch: 003/003 | Batch 001/002 | Train Loss: 0.05
    # Epoch: 003/003 | Batch 002/002 | Train Loss: 0.02
for epoch in range(num_epochs):
    model.train()

    for batch_idx, (features, labels) in enumerate(train_loader):
        logits = model(features)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad() # sets the gradients from the previous round to 0 to prevent unintended gradient accumulation
        loss.backward() # compute gradients of the loss given the model parameters
        optimizer.step() # optimizer uses the gradients to update the model parameters
        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
            f" | Batch {batch_idx+1:03d}/{len(train_loader):03d}"
            f" | Train Loss: {loss:.02f}")

model.eval()
with torch.no_grad():
    outputs = model(X_train)
# prints
    # tensor([[ 2.5896, -2.4775],
    #         [ 2.4005, -2.3052],
    #         [ 2.1214, -2.0500],
    #         [-2.0564,  1.2990],
    #         [-2.4409,  1.5332]])
print(outputs)

torch.set_printoptions(sci_mode=False)
probas = torch.softmax(outputs, dim=1)

# prints
    # tensor([[0.9937, 0.0063],
    #         [0.9910, 0.0090],
    #         [0.9848, 0.0152],
    #         [0.0337, 0.9663],
    #         [0.0184, 0.9816]])
print(probas)

predictions = torch.argmax(probas, dim=1)
# prints tensor([0, 0, 0, 1, 1])
print(predictions)

predictions = torch.argmax(outputs, dim=1)
# prints tensor([0, 0, 0, 1, 1])
print(predictions)
# prints tensor([True, True, True, True, True])
print(predictions == y_train)
# prints tensor(5)
print(torch.sum(predictions == y_train))

def compute_accuracy(model, dataloader): 
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        with torch.no_grad(): 
            logits = model(features)
        
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions # returns true/false depending on if the labels match
        correct += torch.sum(compare) # counts the number of True values
        total_examples += len(compare)
    return (correct / total_examples).item() # fraction of correct prediction, returns the value of the tensor as a float

# prints 1.0
print(compute_accuracy(model, train_loader))

# saving and loading models

torch.save(model.state_dict(), "appendix-a-model.pth")
model = NeuralNetwork(2, 2)
model.load_state_dict(torch.load("appendix-a-model.pth"))

# prints False 
# editor's note: on my computer, a macbook M2
print(torch.cuda.is_available())
# prints True 
# editor's note: on my computer, a macbook M2
print(torch.backends.mps.is_available())

tensor_1 = torch.tensor([1., 2., 3.])
tensor_2 = torch.tensor([4., 5., 6.])
# prints tensor([5., 7., 9.])
print(tensor_1 + tensor_2)

tensor_1 = tensor_1.to("mps")
tensor_2 = tensor_2.to("mps")
# prints tensor([5., 7., 9.], device='mps:0')
# editor's note: the '0' refers to which compute item, in this case, the first one. you can have 'mps:1' or any other number if your computer can handle that
print(tensor_1 + tensor_2)

# single compute item training 

model = NeuralNetwork(num_imputs=2, num_outputs=2)
device = torch.device("mps")
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
num_epochs = 3
# prints
    # Epoch: 001/003 | Batch 001/002 | Train/Val Loss: 0.79
    # Epoch: 001/003 | Batch 002/002 | Train/Val Loss: 0.47
    # Epoch: 002/003 | Batch 001/002 | Train/Val Loss: 0.32
    # Epoch: 002/003 | Batch 002/002 | Train/Val Loss: 0.20
    # Epoch: 003/003 | Batch 001/002 | Train/Val Loss: 0.02
    # Epoch: 003/003 | Batch 002/002 | Train/Val Loss: 0.01
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device) 
        logits = model(features)
        loss = F.cross_entropy(logits, labels) # loss function
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
            f" | Batch {batch_idx+1:03d}/{len(train_loader):03d}"
            f" | Train/Val Loss: {loss:.2f}")
    model.eval()