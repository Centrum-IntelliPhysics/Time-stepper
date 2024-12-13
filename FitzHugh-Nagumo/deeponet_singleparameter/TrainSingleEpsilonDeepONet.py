import torch as pt
import torch.optim as optim
import torch.optim.lr_scheduler as sch
import torch.nn as nn

from Dataset import SingleEpsilonDeepONetDataset
from DeepONet import DeepONet

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Just some sanity pytorch settings
pt.set_grad_enabled(True)
pt.set_default_dtype(pt.float64)

# Load the data in memory
print('Loading Training Data ...')
dataset = SingleEpsilonDeepONetDataset()
batch_size = 128
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print('Done.')

# Initialize the Network and the Optimizer (Adam)
print('\nSetting Up DeepONet Neural Net...')
p = 200
branch_input_size = 400
trunk_input_size = 1
branch_layers = [branch_input_size, 400, 400, 400, 400, 2*p]
trunk_layers  = [trunk_input_size,  400, 400, 400, 400, 2*p]
network = DeepONet(branch_layers=branch_layers, trunk_layers=trunk_layers)
optimizer = optim.Adam(network.parameters(), lr=1.e-4)
step = 250
scheduler = sch.StepLR(optimizer, step_size=step, gamma=0.1)
print('Data Size / Number of Parameters:', len(dataset) / (1.0*network.getNumberOfParameters()))

# Training Routine
loss_fn = nn.MSELoss()
train_losses = []
train_grads = []
train_counter = []
store_directory = './Results/'
log_rate = 100

def getGradient():
    grads = []
    for param in network.parameters():
        grads.append(param.grad.view(-1))
    grads = pt.cat(grads)
    return pt.norm(grads)

def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        # Compute Loss
        output = network(data)
        loss = loss_fn(output, target)

        # Compute loss gradient and do one optimization step
        loss.backward()
        grad = getGradient()
        optimizer.step()

        # Some housekeeping
        train_losses.append(loss.item())
        train_grads.append(grad)
        train_counter.append((1.0*batch_idx)/len(train_loader) + epoch-1)
        if batch_idx % log_rate == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tLoss Gradient: {:.6f} \tlr: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item(), grad, scheduler.get_last_lr()[0]))

            pt.save(network.state_dict(), store_directory + 'model_deeponet_fhn.pth')
            pt.save(optimizer.state_dict(), store_directory + 'optimizer_deeponet_fhn.pth')

# Do the actual training
print('\nStarting Training Procedure...')
n_epochs = 4 * step
try:
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        scheduler.step()
except KeyboardInterrupt:
    print('Aborting Training. Plotting Training Convergence.')

# Show the training results
plt.semilogy(train_counter, train_losses, color='tab:blue', label='Training Loss', alpha=0.5)
plt.semilogy(train_counter, train_grads, color='tab:orange', label='Loss Gradient', alpha=0.5)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()