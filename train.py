import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import utils
from torch.utils.tensorboard import SummaryWriter
from utils import YoloLoss, load_yaml, metrics
from models import create_model
from datasets import create_dataloaders


torch.manual_seed(32)

def train(model, train_loader, optimizer, epoch, scheduler, device, S, B, train_loss_lst, writer):
    model.train()  # Set the module in training mode
    train_loss = 0
    pbar = tqdm(train_loader, leave=True)
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        # back prop
        criterion = YoloLoss(S, B)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()

        # log batch0 images
        if batch_idx == 0 and epoch == 0:
            inputs = inputs.cpu()  # convert to cpu
            img_grid = utils.make_grid(inputs)
            writer.add_image('image batch0', img_grid, 0)

        # print loss and accuracy
        pbar.set_description(f"[Epoch {epoch+1}] loss = {train_loss/(batch_idx+1):.03f}")

    # record training loss
    train_loss /= len(pbar)
    train_loss_lst.append(train_loss)

    tqdm.write(f"Epoch {epoch} training summary -- loss = {train_loss:.03f}")
    return train_loss_lst


def validate(model, val_loader, device, S, B, valid_loss_list):
    model.eval()  # Sets the module in evaluation mode
    val_loss = 0
    pbar = tqdm(val_loader, leave=True)
    with torch.no_grad(): # without gradient calculation
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            criterion = YoloLoss(S, B)
            val_loss += criterion(output, target).item()

    val_loss /= len(val_loader)

    tqdm.write(f"Epoch {epoch} validation summary -- loss = {val_loss:.03f}")
   
    valid_loss_list.append(val_loss)
    return valid_loss_list

if __name__ == "__main__":
    EPOCHS = 1
    SAVE_FREQUENCY = 2
    LR = 1e-3
    BATCH_SIZE = 32

    img_list_path = "datasets/augmented_data"
    S, B, num_classes, input_size = 7, 2, 3, 483

    # create output file folder
    start = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    output_dir = os.path.join('train', start)
    os.makedirs(output_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  

    # build model
    model = create_model(S, B, num_classes).to(device)

    # get data loader
    train_loader, val_loader = create_dataloaders(img_list_path, 0.8, 0.2, BATCH_SIZE, input_size, S, B, num_classes)
    
    optimizer = SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.0005)
    
    scheduler = OneCycleLR(optimizer, max_lr=0.01, epochs=EPOCHS, steps_per_epoch=len(train_loader), anneal_strategy='cos')
    
    train_loss_lst, valid_loss_list = [], []

    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(output_dir)
  
    # train epoch
    for epoch in range(EPOCHS):
        train_loss_lst = train(model, train_loader, optimizer, epoch, scheduler, device, S, B, train_loss_lst, writer)
        valid_loss_list = validate(model, val_loader, device, S, B, valid_loss_list)
        writer.add_scalar('Loss/train', np.average(train_loss_lst), epoch)
        writer.add_scalar('Loss/validate', np.average(valid_loss_list), epoch)

        # save model weight every save_freq epoch
        if epoch % SAVE_FREQUENCY == 0:
            torch.save(model.state_dict(), os.path.join(output_dir, 'epoch' + str(epoch) + '.pt'))

    # save model
    torch.save(model.state_dict(), os.path.join(output_dir, 'last.pt'))

    # save loss plot
    fig = plt.figure()
    print("FIG CREATED")
    plt.plot(range(EPOCHS), train_loss_lst, 'g', label='train loss')
    plt.plot(range(EPOCHS), valid_loss_list, 'r', label='val loss')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(output_dir, 'loss_plot.jpg'))
    plt.close(fig)
