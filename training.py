import torch
import intel_extension_for_pytorch as ipex
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from guided_diffusion.bratsloader import BRATSDataset3D_midslice
from torchmetrics.functional.classification import dice


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    
    # checkpoint = torch.load('model0.pth')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # model.train()
    # model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)
    
    for batch, (X, y, _) in enumerate(dataloader):
        X, y = X.to(torch.float32).to(device), y.to(torch.float32).to(device)

        # Compute prediction error
        with torch.cpu.amp.autocast():
            pred = model(X)
            loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 1 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# def train(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     model.train()
#     for batch, (X, y, _) in enumerate(dataloader):
#         X, y = X.to(torch.float32).to(device), y.to(torch.float32).to(device)

#         # Compute prediction error
#         pred = model(X)
#         loss = loss_fn(pred, y)

#         # Backpropagation
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#         if batch % 1 == 0:
#             loss, current = loss.item(), (batch + 1) * len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    model = ipex.optimize(model, dtype=torch.bfloat16)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y, _ in dataloader:
            X, y = X.to(torch.float32).to(device), y.to(torch.float32).to(device)
            with torch.cpu.amp.autocast():
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct = dice(pred,y.int())
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct
    
    
if __name__ == "__main__":
    seed=42
    torch.manual_seed(seed)

    batch_size = 64
    in_ch = 4
    img_size = 256

    tran_list = [transforms.Resize((img_size,img_size))]
    transform_test = transforms.Compose(tran_list)

    training_data = BRATSDataset3D_midslice('data/MICCAI_BraTS2020_TrainingData/',transform_test)

    training_dataset = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True)

    validation_data = BRATSDataset3D_midslice('data/MICCAI_BraTS2020_TestingData/',transform_test)

    validation_dataset = DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=True)

    # Set device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Define model
    from guided_diffusion.unet_parts import BasicUNet
    model = BasicUNet(n_channels=in_ch, n_classes=1).to(device)
    # print(model)


    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    best_accuracy = 0.0
    epochs = 200

    import time
    
    checkpoint = torch.load('model_nonnormalized.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    model.train()
    model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        # Try putting optimize ipex here and see if it does anything during updates. 
        
        train(training_dataset, model, loss_fn, optimizer)
        # corrects = test(validation_dataset, model, loss_fn)

        # if corrects > best_accuracy:
            # best_accuracy = corrects
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, "model_nonnormalized.pth")
        # print("Saved PyTorch Model State to model.pth")

    print("Done!")