import os
import torch
import argparse
from network.avped import FusionNet
from utils.evaluate import evaluate


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AVped Training Script')
    parser.add_argument('--audio_path', type=str, default='Data/npy_data', help='Path to the audio data')
    parser.add_argument('--image_path', type=str, default='Data/image', help='Path to the image data')
    parser.add_argument('--image_detect_path', type=str, default='Data/image_detection', help='Path to the image detection')
    parser.add_argument('--image_semantic_path', type=str, default='Data/image_semantic', help='Path to the image segmentation')
    parser.add_argument('--label_path', type=str, default='Data/gt', help='Path to the labels')
    parser.add_argument('--lidar_path', type=str, default='Data/lidar', help='Path to the labels')
    parser.add_argument('--val_anno_path', type=str, default='Data/annotation/anotation_test_all/trainval.txt', help='Path to the validation annotation file')
    parser.add_argument('--checkpoint_path', type=str, default='', help='Path to the checkpoint file')
    parser.add_argument('--gpu', type=str, default='cuda:0', help='specify the gpu device')
    parser.add_argument('--vis_lidar', type=int, default='0', help='vis the lidar with detected bbox')
    parser.add_argument('--save_lidar', type=int, default='0', help='save the lidar with detected bbox')
    parser.add_argument('--dark_aug', type=int, default='0', help='use dark_aug to simulate camera fail condition: 0: camera works, 1:camera fial')
    args = parser.parse_args()



    model = FusionNet()
    if args.checkpoint_path != '':
        model.load_state_dict(torch.load(args.checkpoint_path))
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)
    model.eval()

    with open(args.val_anno_path, "r") as f:
        val_anno = f.readlines()


    Dx,Dy,map = evaluate(model,val_anno,args.audio_path,args.image_path,args.lidar_path,args.label_path,args.gpu,args.dark_aug,args.vis_lidar)
    print(Dx,Dy,map)

