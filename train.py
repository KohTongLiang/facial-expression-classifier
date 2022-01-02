import torch
import argparse

from model import Classifier
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Lambda, Compose


def prepare_dataset(root_dir, batch_size, transform):
    train_dataset = datasets.ImageFolder(root=f'{root_dir}/train/', transform=transform)
    test_dataset = datasets.ImageFolder(root=f'{root_dir}/test/', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True, num_workers=4)
    return train_dataloader, test_dataloader

def train(train_dataloader, model, batch_size, loss_fn, optimizer, writer, t):
    size = len(train_dataloader.dataset)

    model.train()
    i = 0
    for batch, (x, y) in enumerate(train_dataloader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch_size % 100 == 0:
        loss, current = loss.item(), batch * len(x)

        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        i+=1
    loss /= i 
    writer.add_scalar('Training loss', loss, t)

def test(test_dataloader, model, loss_fn, writer, t):
    size = len(test_dataloader.dataset)
    test_batches = len(test_dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= test_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    writer.add_scalar('Testing loss', test_loss, t)


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser(description="Simple facial expression classifier")

    parser.add_argument("--path", type=str, default="./dataset", help="Path to dataset.")
    parser.add_argument("--batch", type=int, default=64, help="Training batch size.")
    parser.add_argument("--epoch", type=int, default=50, help="Total number of training iterations")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate of training.")
    parser.add_argument("--device", type=str, default="cuda", help="GPU Device.")
    parser.add_argument("--ckpt", type=str, default=None, help="Continue training")

    args = parser.parse_args()

    root_dir = args.path
    batch_size = args.batch
    epochs = args.epoch
    lr = args.lr
    device = args.device

    # prepare dataset
    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])
    train_dataloader, test_dataloader = prepare_dataset(root_dir, batch_size, transform)
    model = Classifier().to(device)

    # tensorboard init
    writer = SummaryWriter("Experiment")

    # load checkpoint if provided
    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint)

    # define loss objective
    loss_fn = nn.CrossEntropyLoss()
    
    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, batch_size, loss_fn, optimizer, writer, t)
        test(test_dataloader, model, loss_fn, writer, t)
        torch.save(model.state_dict(), f"./checkpoints/model_epoch_{t}.pth")
    writer.close()
    print("Done!")