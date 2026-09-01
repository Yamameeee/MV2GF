import torch
import lightning as pl
from model import Detection
from loss import FocalLoss, RegL1Loss


class Trainer(pl.LightningModule):
    def __init__(
            self,
            arch='resnet18',
            pool='mean', 
            decoder='unet',
            input_reduce=1.5,
            feat_dim=256,
            da3_height=280,
            da3_width=504,
            da3_dim=1536,
            voxel_grid_xy=0.05,
            voxel_grid_z=0.25,
            learning_rate=1e-3,
            min_learning_rate=1e-6,
            div_factor=25,
            warmup_iter_rate=0.2,
    ):
        super().__init__()

        # model
        self.arch = arch
        self.pool = pool
        self.decoder = decoder
        self.input_reduce = input_reduce
        self.input_size = (int(1080 / input_reduce), int(1920 / input_reduce))
        self.feat_dim = feat_dim
        self.da3_size = (da3_height, da3_width)
        self.da3_dim = da3_dim
        self.voxel_grid = (voxel_grid_xy, voxel_grid_xy, voxel_grid_z)

        self.model = Detection(
            arch=self.arch,
            pool=self.pool,
            decoder=self.decoder,
            input_size=self.input_size,
            feat_dim=self.feat_dim,
            da3_size=self.da3_size,
            da3_dim=self.da3_dim,
            voxel_grid=self.voxel_grid,
        )

        # loss
        self.focal_loss = FocalLoss()
        self.regress_loss = RegL1Loss()

        # optimizer and learning rate
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.div_factor = div_factor # initial_lr = learning_rate / div_factor
        self.warmup_iter_rate = warmup_iter_rate

        self.save_hyperparameters()

    def forward(self, imgs, da3_feats, da3_points, voxel_region):
        (world_heatmap, world_offset), (head_heatmap, foot_heatmap), bev_feat = self.model(
            imgs, da3_feats, da3_points, voxel_region,
        )
        return world_heatmap, world_offset, head_heatmap, foot_heatmap, bev_feat
    
    def loss(self, world_heatmap, world_offset, head_heatmap, foot_heatmap, world_gt, imgs_gt):
        B, N = imgs_gt['foot_map'].shape[:2]
        for key in imgs_gt.keys():
            imgs_gt[key] = imgs_gt[key].view([B * N] + list(imgs_gt[key].shape)[2:])
        
        # loss bev
        loss_w_hm = self.focal_loss(world_heatmap, world_gt['heatmap'])
        loss_w_off = self.regress_loss(
            world_offset, world_gt['reg_mask'], world_gt['idx'], world_gt['offset']
        )
        
        # loss img
        loss_img_head = self.focal_loss(head_heatmap, imgs_gt['head_map'])
        loss_img_foot = self.focal_loss(foot_heatmap, imgs_gt['foot_map'])
        
        w_loss = loss_w_hm + loss_w_off
        img_loss = loss_img_head + loss_img_foot
        loss = w_loss + img_loss / N

        loss_dict = {
            'loss_world_center': loss_w_hm,
            'loss_world_offset': loss_w_off,
            'loss_image_head': loss_img_head,
            'loss_image_foot': loss_img_foot,
            'loss_image': img_loss,
        }

        return loss, loss_dict
    
    def training_step(self, batch, batch_idx):
        imgs, da3_feats, da3_points, world_gt, imgs_gt, \
            frame, region_size, Rworld_shape = batch
        
        region_size = region_size.squeeze(0).tolist()
        voxel_region = [
            -self.voxel_grid[0], 
            region_size[1],
            -self.voxel_grid[1], 
            region_size[0],
            -self.voxel_grid[2], 
            2.0
        ]
        world_heatmap, world_offset, head_heatmap, foot_heatmap, _ = self(
            imgs, da3_feats, da3_points, voxel_region
        )
        loss, loss_dict = self.loss(
            world_heatmap, world_offset, head_heatmap, foot_heatmap,
            world_gt, imgs_gt
        )

        self.log('train_loss', loss, prog_bar=True, batch_size=1)
        for key, value in loss_dict.items():
            self.log(f'train/{key}', value, batch_size=1)
        
        return loss
    
    def configure_optimizers(self):
        initial_learning_rate = self.learning_rate / self.div_factor
        final_div_factor = initial_learning_rate / self.min_learning_rate

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            div_factor=self.div_factor,
            final_div_factor=final_div_factor,
            pct_start=self.warmup_iter_rate,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }
    

if __name__ == '__main__':
    from lightning.pytorch.cli import LightningCLI
    torch.set_float32_matmul_precision('medium')

    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.link_arguments("trainer.accumulate_grad_batches", "data.init_args.accumulate_grad_batches")

    cli = MyLightningCLI(Trainer)