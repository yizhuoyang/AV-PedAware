import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from dataloader.avped_dataloader import AVpedLoader
from network.avped import FusionNet
from utils.loss import regression_loss

# Set CUDA visible devices
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def train_model(model, train_dataloader, optimizer, loss_fn, cls_loss, cls_loss2, device):
    model.train()
    train_loss = 0
    for data in tqdm(train_dataloader, total=len(train_dataloader), unit='batch'):
        spec, image, gt, detect, segment = [d.to(device) for d in data]
        segment = segment.long()
        optimizer.zero_grad()
        p, d, s = model(spec, image)
        loss_detect = cls_loss(d, detect)
        loss_seg = cls_loss2(s, segment)
        loss_position = loss_fn(p, gt)
        loss = loss_position + loss_detect*0.5 + loss_seg*0.3
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_dataloader)

def validate_model(model, val_dataloader, loss_fn, cls_loss, cls_loss2, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in val_dataloader:
            spec, image, gt, detect, segment = [d.to(device) for d in data]
            segment = segment.long()
            p, d, s = model(spec, image)
            loss_detect = cls_loss(d, detect)
            loss_seg = cls_loss2(s, segment)
            loss_position = loss_fn(p, gt)
            loss = loss_position + loss_detect*0.5+ loss_seg*0.3
            val_loss += loss.item()
    return val_loss / len(val_dataloader)

def main(args):
    # Load the model
    model = FusionNet(args.dropout_rate)
    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path))
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    with open(args.train_anno_path, "r") as f:
        train_anno = f.readlines()

    with open(args.val_anno_path, "r") as f:
        val_anno = f.readlines()

    # Define data loaders
    train_data = AVpedLoader(train_anno, args.audio_path, args.image_path, args.image_detect_path,
                             args.image_semantic_path, args.label_path, args.lidar_path, dark_aug=1, testing=0)
    val_data = AVpedLoader(val_anno, args.audio_path, args.image_path, args.image_detect_path,
                           args.image_semantic_path, args.label_path, args.lidar_path, dark_aug=1, testing=0)
    train_dataloader = DataLoader(train_data, args.batchsize, shuffle=True, num_workers=args.workers, drop_last=True)
    val_dataloader = DataLoader(val_data, args.batchsize, shuffle=True, num_workers=args.workers, drop_last=True)
    # Define optimizer, scheduler, and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    loss_fn  = regression_loss
    weights = torch.tensor([1.0,3.0]).to(device)
    cls_loss  = torch.nn.CrossEntropyLoss(torch.tensor([1.0,1.0]).to(device))
    cls_loss2  = torch.nn.CrossEntropyLoss(weights)
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.train_epoch):
        train_loss = train_model(model, train_dataloader, optimizer, loss_fn, cls_loss, cls_loss2, device)
        val_loss = validate_model(model, val_dataloader, loss_fn, cls_loss, cls_loss2, device)

        print(f"Epoch {epoch + 1}/{args.train_epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")

        # Save the model if it has the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_path, 'model_best.pth'))

        # Save the model at the end of each epoch
        torch.save(model.state_dict(), os.path.join(args.save_path, f'{epoch}.pth'))

if __name__ == "__main__":
    # Define command line arguments and parse them
    parser = argparse.ArgumentParser(description='AVped Training Script')
    parser = argparse.ArgumentParser(description='AVped Training Script')
    parser.add_argument('--audio_path', type=str, default='Data/npy_data', help='Path to the audio data')
    parser.add_argument('--image_path', type=str, default='Data/image', help='Path to the image data')
    parser.add_argument('--image_detect_path', type=str, default='Data/image_detection', help='Path to the image detection')
    parser.add_argument('--image_semantic_path', type=str, default='Data/image_semantic', help='Path to the image segmentation')
    parser.add_argument('--label_path', type=str, default='Data/gt', help='Path to the labels')
    parser.add_argument('--lidar_path', type=str, default='Data/lidar', help='Path to the labels')
    parser.add_argument('--save_path', type=str, default='output/', help='Path to save the model')
    parser.add_argument('--train_anno_path', type=str, default='Data/annotation/anotation_train_all/trainval.txt', help='Path to the training annotation file')
    parser.add_argument('--val_anno_path', type=str, default='Data/annotation/anotation_test_all/trainval.txt', help='Path to the validation annotation file')
    parser.add_argument('--checkpoint_path', type=str, default='', help='Path to the checkpoint file')
    parser.add_argument('--num_mic', type=int, default=8, help='Number of microphones')
    parser.add_argument('--dropout_rate', type=float, default=0.6, help='Dropout rate')
    parser.add_argument('--batchsize', type=int, default=32, help='Batch size')
    parser.add_argument('--train_epoch', type=int, default=80, help='Number of training epochs')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--gpu', type=str, default='cuda:0', help='specify the gpu device')
    args = parser.parse_args()

    # Check if save_path directory exists, if not, create it
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    main(args)
