import torch as pt
import torch.optim as optim
import torch.nn as nn

from Dataset import DeepONetDataset
from DeepONet import DeepONet

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Just some sanity pytorch settings
pt.set_grad_enabled(True)
pt.set_default_dtype(pt.float64)

# Load the data in memory
print('Loading Training Data ...')
dataset = DeepONetDataset()
batch_size = 1024
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print('Done.')

# Initialize the Network and the Optimizer (Adam)
print('\nSetting Up DeepONet Neural Net...')
p = 100
branch_layers = [3, 100, 100, 2*p]
trunk_layers = [1, 100, 100, 2*p]
network = DeepONet(branch_layers=branch_layers, trunk_layers=trunk_layers)
optimizer = optim.Adam(network.parameters(), lr=1.e-4)
print('Data Size / Number of Parameters:', len(dataset) / (1.0*network.getNumberOfParameters()))

# Training Routine
loss_fn = nn.MSELoss()
train_losses = []
train_counter = []
store_directory = './Results/'
log_rate = 100
def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        # Compute Loss
        output = network(data)
        loss = loss_fn(output, target)

        # Compute loss gradient and do one optimization step
        loss.backward()
        optimizer.step()

        # Some housekeeping
        train_losses.append(loss.item())
        train_counter.append((batch_idx) + ((epoch-1)*len(train_loader.dataset)))
        if batch_idx % log_rate == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))

            pt.save(network.state_dict(), store_directory + 'model_deeponet_fhn.pth')
            pt.save(optimizer.state_dict(), store_directory + 'optimizer_deeponet_fhn.pth')

# Do the actual training
print('\nStarting Training Procedure...')
n_epochs = 50000
try:
    for epoch in range(1, n_epochs + 1):
        train(epoch)
except KeyboardInterrupt:
    print('Aborting Training. Plotting Training Convergence.')

# Show the training results
fig = plt.figure()
plt.semilogy(train_counter, train_losses, color='blue', label='Training Loss')
plt.legend()
plt.xlabel('Number of training examples seen')
plt.ylabel('Loss')
plt.show()