import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectNet(nn.Module):
    def __init__(self, feature_dim=512):
        super(DetectNet, self).__init__()
        self.feature_dim = feature_dim
        #  position head
        self.position_est1 = nn.Linear(self.feature_dim, 256)
        self.position_est2 = nn.Linear(256, 128)
        self.position_est3 = nn.Linear(128, 7)
        self.dropout_p2 = nn.Dropout(0.4)
        #  segmentation head
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


    def forward(self, f_all,f_audio):

        position = F.relu(self.position_est1(f_all))
        position  = F.relu(self.position_est2(position))
        position = self.dropout_p2(position)
        position  = self.position_est3(position)

        x = F.relu(self.dense(f_audio))
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
        return position,segmentation



