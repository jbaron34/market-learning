import torch
from torchvision import datasets, transforms
import ops

# Define a transform to normalize the data
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

# Download and load the training data
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

# Download and load the test data
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

num_models = 10

models = torch.nn.ModuleList(
    [
        torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28 * 28, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 10),
        )
        for _ in range(num_models)
    ]
)


loss_fn = ops.calculate_profit


optimizer = torch.optim.Adam(models.parameters(), lr=1e-3)
epochs = 10

for epoch in range(epochs):
    for x, y in train_loader:
        optimizer.zero_grad()
        os = torch.stack([model(x) for model in models], dim=-2)
        loss = loss_fn(os, y)
        loss.backward()
        optimizer.step()
        print(
            "acc:",
            (os.sum(dim=-2).argmax(dim=-1) == y).float().mean().item(),
            "iaccmax:",
            (os.argmax(dim=-1) == y.unsqueeze(-1)).float().mean(dim=0).max().item(),
            "iaccmin:",
            (os.argmax(dim=-1) == y.unsqueeze(-1)).float().mean(dim=0).min().item(),
            "loss:",
            loss.item(),
            "turnover:",
            os.abs().mean().item(),
        )
