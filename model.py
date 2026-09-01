import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from timm.utils.model import freeze_batch_norm_2d
from torch_cluster import grid_cluster
from torch_scatter import scatter


def freeze_bn(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            freeze_bn(module)

        if isinstance(module, torch.nn.BatchNorm2d):
            setattr(model, n, freeze_batch_norm_2d(module))


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def output_head(in_dim, feat_dim, out_dim):
    if feat_dim:
        fc = nn.Sequential(nn.Conv2d(in_dim, feat_dim, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(feat_dim, out_dim, 1))
    else:
        fc = nn.Sequential(nn.Conv2d(in_dim, out_dim, 1))
    return fc


class UpConcat(nn.Module):
    def __init__(self, in_channels, out_channels, size=None, scale_factor=None):
        super().__init__()
        self.upsample = nn.Upsample(size=size, scale_factor=scale_factor,
                                    mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels * 2, out_channels, 
                              kernel_size=3, padding=1, bias=False)

    def forward(self, x_to_upsample, x):
        x_to_upsample = self.upsample(x_to_upsample)
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)


class UpsamplingConcat(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x_to_upsample, x):
        x_to_upsample = self.upsample(x_to_upsample)
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)


class FPN_resnet_da3(nn.Module):
    def __init__(
            self, 
            arch='resnet18', 
            out_channels=256, 
            input_size=[720, 1280], 
            da3_channels=1536,
        ):
        super().__init__()

        if arch == 'resnet18':
            in_channels=[64, 128, 256, 512]
            resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        elif arch == 'resnet50':
            in_channels=[256, 512, 1024, 2048]
            resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        elif arch == 'resnet101':
            in_channels=[256, 512, 1024, 2048]
            resnet = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT)
        else:
            NotImplementedError
        freeze_bn(resnet)

        # resnet
        self.layer0 = nn.Sequential(*list(resnet.children())[:4])
        self.layer1 = resnet.layer1 # 1/4
        self.layer2 = resnet.layer2 # 1/8
        self.layer3 = resnet.layer3 # 1/16
        self.layer4 = resnet.layer4 # 1/32

        # fpn
        self.latelal_conv1 = nn.Conv2d(in_channels[0] + da3_channels, out_channels, 1)
        self.latelal_conv2 = nn.Conv2d(in_channels[1] + da3_channels, out_channels, 1)
        self.latelal_conv3 = nn.Conv2d(in_channels[2] + da3_channels, out_channels, 1)
        self.latelal_conv4 = nn.Conv2d(in_channels[3] + da3_channels, out_channels, 1)

        self.upsampling_layer2 = UpConcat(out_channels, out_channels, size=[input_size[0]//4, input_size[1]//4])
        self.upsampling_layer3 = UpConcat(out_channels, out_channels, size=[input_size[0]//8, input_size[1]//8])
        self.upsampling_layer4 = UpConcat(out_channels, out_channels, size=[input_size[0]//16, input_size[1]//16])


    def forward(self, x, da3_feats):
        """
        forward の Docstring
        
        :param x: [N, 3, H, W]
        :param da3_feature: [feat_layer_19, feat_layer_27, feat_layer_33, feat_layer_39] ([4, N, C', 20, 36])
        """
        # backbone
        # x: [720, 1280], [1080, 1920], [864, 1536]
        x0 = self.layer0(x) # [6, 64, 180, 320], [6, 64, 270, 480], [6, 64, 216, 384] (1/4)
        x1 = self.layer1(x0) # [6, 64, 180, 320], [6, 64, 270, 480], [6, 64, 216, 384] (1/4)
        x2 = self.layer2(x1) # [6, 128, 90, 160], [6, 128, 135, 240], [6, 128, 108, 192] (1/8)
        x3 = self.layer3(x2) # [6, 256, 45, 80], [6, 256, 68, 120], [6, 256, 54, 96] (1/16)
        x4 = self.layer4(x3) # [6, 512, 23, 40], [6, 512, 34, 60], [6, 512, 27, 48] (1/32)

        # upsample da3 feature
        da3_feat1 = F.interpolate(da3_feats[0], x1.shape[-2:], mode='bilinear') # C', 1/4
        da3_feat2 = F.interpolate(da3_feats[1], x2.shape[-2:], mode='bilinear') # C', 1/8
        da3_feat3 = F.interpolate(da3_feats[2], x3.shape[-2:], mode='bilinear') # C', 1/16
        da3_feat4 = F.interpolate(da3_feats[3], x4.shape[-2:], mode='bilinear') # C', 1/32

        # concat resnet and da3 feature
        x1 = torch.cat([x1, da3_feat1], dim=1)
        x2 = torch.cat([x2, da3_feat2], dim=1)
        x3 = torch.cat([x3, da3_feat3], dim=1)
        x4 = torch.cat([x4, da3_feat4], dim=1)

        # lateral conv
        x1 = self.latelal_conv1(x1) # C, 1/4
        x2 = self.latelal_conv2(x2) # C, 1/8
        x3 = self.latelal_conv3(x3) # C, 1/16
        x4 = self.latelal_conv4(x4) # C, 1/32
        
        # top-down
        f_out3 = self.upsampling_layer4(x4, x3) # 1/16
        f_out2 = self.upsampling_layer3(f_out3, x2) # 1/8
        f_out1 = self.upsampling_layer2(f_out2, x1) # 1/4
    
        return [f_out1, f_out2, f_out3] # 1/4, 1/8, 1/16
    

class UNet(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        freeze_bn(backbone)
        self.first_conv = nn.Conv2d(feat_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        self.up3_skip = UpsamplingConcat(256 + 128, 256)
        self.up2_skip = UpsamplingConcat(256 + 64, 256)
        self.up1_skip = UpsamplingConcat(256 + feat_dim, feat_dim)

    def forward(self, x):
        b, c, h, w = x.shape

        # pad input
        m = 16
        ph, pw = math.ceil(h / m) * m - h, math.ceil(w / m) * m - w
        pt, pb = ph // 2, ph - (ph // 2)
        pl, pr = pw // 2, pw - (pw // 2)
        x = torch.nn.functional.pad(x, [pl, pr, pt, pb])

        # (H, W)
        skip_x = {'1': x}
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        # (H/4, W/4)
        x = self.layer1(x)
        skip_x['2'] = x
        x = self.layer2(x)
        skip_x['3'] = x

        # (H/8, W/8)
        x = self.layer3(x)

        # First upsample to (H/4, W/4)
        x = self.up3_skip(x, skip_x['3'])

        # Second upsample to (H/2, W/2)
        x = self.up2_skip(x, skip_x['2'])

        # Third upsample to (H, W)
        x = self.up1_skip(x, skip_x['1'])

        # Unpad
        x = x[..., pt:pt + h, pl:pl + w]

        return x


class Detection(nn.Module):
    def __init__(
            self,
            arch='resnet18',
            pool='mean',
            decoder: str='unet',
            input_size=(720, 1280),
            feat_dim: int=256,
            da3_size: tuple=(280, 504),
            da3_dim: int=1536,
            voxel_grid: tuple=(0.05, 0.05, 0.25),
    ):
        super().__init__()

        self.base = FPN_resnet_da3(
            arch=arch,
            out_channels=feat_dim, 
            input_size=input_size, 
            da3_channels=da3_dim,
        )

        # img heads
        self.head_heatmap = output_head(feat_dim, None, 1)
        self.foot_heatmap = output_head(feat_dim, None, 1)

        # uosample image feature to da3 size
        self.upsample_da3 = nn.Upsample(size=da3_size, mode='bilinear', align_corners=False)

        # voxelize
        self.voxel_grid = torch.tensor(voxel_grid, dtype=torch.float)
        self.start = torch.tensor([0., 0., 0.], dtype=torch.float)

        # pooling
        self.pool = pool
        assert self.pool == 'mean' or self.pool == 'max'

        # height compresser
        z_size = int(2.0 / voxel_grid[-1])
        if self.voxel_grid[0] == 0.05:
            self.world_height = nn.Sequential(
                nn.Conv2d(feat_dim * z_size, feat_dim, 1), nn.ReLU(),
                nn.Conv2d(feat_dim, feat_dim, 3, padding=1), nn.ReLU(),
                nn.Conv2d(feat_dim, feat_dim, 3, padding=2, dilation=2), nn.ReLU(),
                nn.Conv2d(feat_dim, feat_dim, 3, padding=4, dilation=4), nn.ReLU(),
                nn.Conv2d(feat_dim, feat_dim, 3, stride=2, padding=1), nn.ReLU(),
            )
        else:
            NotImplementedError
        
        if decoder == 'unet':
            self.decoder = UNet(
                feat_dim=feat_dim
            )
        else:
            NotImplementedError

        # world heads
        self.world_heatmap = output_head(feat_dim, None, 1)
        self.world_offset = output_head(feat_dim, None, 2)
    
        # init
        self.head_heatmap[-1].bias.data.fill_(-2.19)
        self.foot_heatmap[-1].bias.data.fill_(-2.19)
        self.world_heatmap[-1].bias.data.fill_(-2.19)
        fill_fc_weights(self.world_offset)
    
    def forward(self, imgs, da3_features, da3_pointmaps, voxel_region):
        """
        imgs: [B, N, C, H, W]
        da3_features (List): [B, 4, N, 20, 36, C']
        da3_pointmaps: [B, N, 280, 504, 3]
        voxel_region (List): [min_x, max_x, min_y, max_y, min_z, max_z]
        """
        B, N, C, H, W = imgs.shape
        imgs = imgs.squeeze(0)
        da3_pointmaps = da3_pointmaps.squeeze(0)
        da3_features = da3_features.squeeze(0).permute(0, 1, 4, 2, 3)
        
        # extract image features
        imgs_feat, _, _ = self.base(imgs, da3_features)

        # predict image heatmap
        head_heatmap = self.head_heatmap(imgs_feat)
        foot_heatmap = self.foot_heatmap(imgs_feat)

        # upsample image features
        imgs_feat = self.upsample_da3(imgs_feat)

        # voxelize image features
        _, C, _, _ = imgs_feat.shape
        da3_point = da3_pointmaps.reshape(-1, 3)
        imgs_feat = imgs_feat.permute(0, 2, 3, 1).reshape(-1, C)
        
        voxel_feat = self.voxelize(imgs_feat, da3_point, voxel_region)
        B, X, Y, Z, C = voxel_feat.shape
        voxel_feat = voxel_feat.permute(0, 4, 3, 2, 1).reshape(B, C*Z, Y, X)
        bev_feat = self.world_height(voxel_feat)
        bev_feat = self.decoder(bev_feat)

        world_heatmap = self.world_heatmap(bev_feat)
        world_offset = self.world_offset(bev_feat)

        return (world_heatmap, world_offset), (head_heatmap, foot_heatmap), bev_feat
    
    def voxelize(self, feats, points, region):
        """
        :param feats: [N*280*508, C]
        :param points: [N*280*504, 3]
        :param region (List): [min_x, max_x, min_y, max_y, min_z, max_z]
        """
        _, C = feats.shape
        min_x, max_x, min_y, max_y, min_z, max_z = region
        valid_mask = (points[:, 0] > min_x) & (points[:, 0] < max_x) & \
            (points[:, 1] > min_y) & (points[:, 1] < max_y) & \
                (points[:, 2] > min_z) & (points[:, 2] < max_z)
        valid_points = points[valid_mask]
        valid_feats = feats[valid_mask]
        
        voxel_grid = self.voxel_grid.to(valid_points.device)
        start = self.start.to(valid_points.device)
        cluster = grid_cluster(
            valid_points,
            voxel_grid,
            start=start
        )

        voxel_feats = scatter(valid_feats, cluster, dim=0, reduce=self.pool)
        voxel_pos = scatter(valid_points, cluster, dim=0, reduce='mean')

        cluster_unique = cluster.unique()
        voxel_feats = voxel_feats[cluster_unique]
        voxel_pos = voxel_pos[cluster_unique]
        
        grid_coords = (voxel_pos / voxel_grid).long()
        X = int(max_x / voxel_grid[0])
        Y = int(max_y / voxel_grid[1])
        Z = int(max_z / voxel_grid[2])
        voxels = torch.zeros([X, Y, Z, C], dtype=feats.dtype, device=feats.device)
        voxels[grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]] = voxel_feats
        
        return voxels.unsqueeze(0)