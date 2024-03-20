import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from network.audio_net import AudioNet
from network.visual_net import VisualNet
from network.detection_heads import DetectNet


class FusionNet(nn.Module):
    def __init__(self, dropout_rate=0.6, kerel_num=32, feature_dim=512):
        super().__init__()
        # Visual Def
        self.feature_dim = feature_dim
        self.resnet_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet_feature = nn.Sequential(*list(self.resnet_model.children())[:-2])
        self.convvis = nn.Conv2d(256, kerel_num, (3, 3), padding=1)
        self.convres = nn.Conv2d(2048, 256, (3, 3), padding=1)
        self.fcvis = nn.Linear(2048, feature_dim)

        # Audio Def
        # self.audionet = ASTModel(input_tdim=224,input_fdim=224,label_dim=self.feature_dim, audioset_pretrain=False,imagenet_pretrain=False,model_size='base384')
        self.convt1 = nn.Conv2d(8, 16, (3, 64))
        self.convt2 = nn.Conv2d(8, 16, (5, 64))

        self.convf1 = nn.Conv2d(8, 16, (64, 3))
        self.convf2 = nn.Conv2d(8, 16, (64, 5))

        self.fc1 = nn.Linear(3904, feature_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.bn_a1    = nn.BatchNorm2d(16)
        self.bn_a2    = nn.BatchNorm2d(16)
        self.bn_a3    = nn.BatchNorm2d(16)
        self.bn_a4    = nn.BatchNorm2d(16)
        # Detection heads
        self.fc2 = nn.Linear(feature_dim, 256)
        self.fc3 = nn.Linear(256, 100)
        self.fc4 = nn.Linear(100, 1)


        self.detect = nn.Linear(feature_dim, 256)
        self.detec2 = nn.Linear(256, 128)
        self.detec3 = nn.Linear(128, 2)

        self.position_est1 = nn.Linear(self.feature_dim, 256)
        self.position_est2 = nn.Linear(256, 128)
        self.position_est3 = nn.Linear(128, 7)


        self.dense = nn.Linear(self.feature_dim, 32*32)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(0.4)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(0.4)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout3 = nn.Dropout2d(0.4)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.upsample_semantic1 = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsample_semantic2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.concat = nn.Conv2d(160, 2, kernel_size=1)
        self.dropout_p1 = nn.Dropout(0.4)
        self.dropout_p2 = nn.Dropout(0.4)
        self.dropout_d = nn.Dropout(0.4)
    def forward(self, x, y):
        [b, c, row, col] = x.size()
        # AudioNet

        t1 = F.relu(self.convt1(x))
        # t1 = self.bn_a1(t1)
        t1 = self.dropout1(t1).view(b, -1)

        t2 = F.relu(self.convt2(x))
        # t2 = self.bn_a2(t2)
        t2 = self.dropout2(t2).view(b, -1)

        f1 = F.relu(self.convf1(x))
        # f1 = self.bn_a3(f1)
        f1 = self.dropout3(f1).view(b, -1)

        f2 = F.relu(self.convf2(x))
        f2 = self.bn_a4(f2)
        f2 = self.dropout4(f2).view(b, -1)

        feature_tf = torch.cat([t1, t2, f1, f2], dim=-1)
        feature_tf = feature_tf.view(b, -1)
        feature_tf_audio = F.relu(self.fc1(feature_tf))
        feature_tf = feature_tf_audio.view(-1, 1, self.feature_dim)

        # visual network
        y = self.resnet_feature(y)
        y = F.relu(self.convres(y))
        y = F.relu(self.convvis(y))
        fi = y.view(-1, 2048)
        fi = F.relu(self.fcvis(fi))
        fi_tem = fi.view(-1, 1, self.feature_dim)

        # Feature Fusion
        f_all = torch.cat([feature_tf, fi_tem], dim=1)
        detect = F.relu(self.detect(fi))
        detect = F.relu(self.detec2(detect))
        # detect  = self.dropout_d(detect)
        detect = self.detec3(detect)
        detect_soft = F.softmax(detect)
        detect_expanded = detect_soft.unsqueeze(-1)
        f_all = f_all * detect_expanded
        f_all = torch.sum(f_all, 1)
        f_all = f_all.view(b,-1)



        ##################  Detection module #####################
        # Predict Trajectoy
        position = F.relu(self.position_est1(f_all))
        position  = F.relu(self.position_est2(position))
        position = self.dropout_p2(position)
        position  = self.position_est3(position)


        x = F.relu(self.dense(feature_tf_audio))
        x = x.view(-1, 1, 32, 32)

        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        # x = self.dropout1(x)
        x = self.upsample1(x)

        semantic1 = x.clone()

        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        # x = self.dropout2(x)
        x = self.upsample2(x)

        semantic2 = x.clone()

        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        # x = self.dropout3(x)
        x = self.upsample3(x)

        semantic1_up = self.upsample_semantic1(semantic1)
        semantic2_up = self.upsample_semantic2(semantic2)
        x = torch.cat((x, semantic1_up, semantic2_up), dim=1)

        segmentation = self.concat(x)


        return position,detect,segmentation


