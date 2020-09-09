import numpy as np
import cv2
import torch
import os
from torch.nn import functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
from albumentations import CLAHE, HorizontalFlip, Compose, RandomBrightnessContrast, RandomGamma, Resize, \
    ChannelShuffle, ShiftScaleRotate, VerticalFlip, Normalize
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_lightning.callbacks import LearningRateLogger, ModelCheckpoint

aug = Compose([
    Resize(480, 640),
    ChannelShuffle(p=0.5),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.7, border_mode=cv2.BORDER_CONSTANT),
    CLAHE(p=0.1),
    RandomBrightnessContrast(p=0.8),
    RandomGamma(p=0.5),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


class REMODEL_dataset(Dataset):
    def __init__(self, dataset_dir, transforms=None):
        self.dataset_dir = dataset_dir
        self.transforms = transforms
        self.img_lst = os.listdir(os.path.join(self.dataset_dir, "img"))

    def __getitem__(self, idx):
        image_name = self.img_lst[idx]
        img = cv2.imread(os.path.join(self.dataset_dir, "img", image_name))
        mask = cv2.imread(os.path.join(self.dataset_dir, "msk", image_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask'] / 255.0
        return img, mask

    def __len__(self):
        return len(self.img_lst)


def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    intersection = (m1 * m2).sum().float()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


class REMODEL_segmenter(pl.LightningModule):

    def __init__(self, data_path: str, batch_size: int, lr: float):
        super(REMODEL_segmenter, self).__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.lr = lr

        self.net = smp.Unet('efficientnet-b3', encoder_weights='imagenet', activation='sigmoid')

    def forward(self, x):
        # return self.net(x)
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float().view(-1, 3, 480, 640)
        mask = mask.float().view(-1, 1, 480, 640)
        out = self(img)
        loss_val = F.binary_cross_entropy_with_logits(out, mask)
        log_dict = {'train_loss': loss_val}
        return {'loss': loss_val, 'log': log_dict, 'progress_bar': log_dict}

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float().view(-1, 3, 480, 640)
        mask = mask.float().view(-1, 1, 480, 640)
        out = self(img)
        loss_val = F.binary_cross_entropy_with_logits(out, mask)
        val_dice = dice_coeff(out, mask)
        return {'val_loss': loss_val, 'dice': val_dice}

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x['val_loss'] for x in outputs]).mean()
        dice_val = torch.stack([x['dice'] for x in outputs]).mean()
        log_dict = {'val_loss': loss_val, 'dice': dice_val}
        return {'log': log_dict, 'val_loss': log_dict['val_loss'], 'dice': log_dict['dice'], 'progress_bar': log_dict}

    def prepare_data(self):
        validation_split = .25
        shuffle_dataset = True
        random_seed = 42

        self.dataset = REMODEL_dataset(dataset_dir=self.data_path, transforms=aug)
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        self.train_sampler = SubsetRandomSampler(train_indices)
        self.valid_sampler = SubsetRandomSampler(val_indices)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.train_sampler)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.valid_sampler)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.2)
        return [opt], [sch]


if __name__ == '__main__':
    checkpoint_callback = ModelCheckpoint(
        filepath='best_weights_{epoch:04d}-{dice:.5f}',
        save_top_k=3,
        verbose=True,
        monitor='dice',
        mode='max'
    )

    model = REMODEL_segmenter(data_path="dataset", batch_size=4, lr=3e-4)
    lr_logger = LearningRateLogger()
    trainer = pl.Trainer(
        callbacks=[lr_logger],
        checkpoint_callback=checkpoint_callback,
        max_epochs=200,
        gpus=1
    )
    trainer.fit(model)
