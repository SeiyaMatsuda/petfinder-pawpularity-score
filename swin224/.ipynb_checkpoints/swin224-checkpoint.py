#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../input/U-2-Net/')
sys.path.append('../')


# In[2]:


# from u2net_test import extract
# extract('../input/petfinder-pawpularity-score/train', '../input/petfinder-pawpularity-score/train_U2NET')
# extract('../input/petfinder-pawpularity-score/test', '../input/petfinder-pawpularity-score/test_U2NET')


# In[3]:


import numpy as np, pandas as pd
from glob import glob
import shutil, os
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import IncrementalPCA
from tqdm.notebook import tqdm
import torch.nn.functional as F
from albumentations.pytorch.transforms import ToTensorV2
import seaborn as sns
import PIL.Image as Image
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import ttach as tta
import time
import pandas_profiling as pdp
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from utils.util import *
from utils.losses import *
import torch.nn as nn
import transformers as T
import albumentations
import pandas as pd
import cv2
import numpy as np
import timm
import torch.nn as nn
from sklearn import metrics
import torch
from tqdm import tqdm
import math
import albumentations
import tez
import torch.optim as optim
import warnings
warnings.simplefilter('ignore')


# In[4]:


class CONFIG:
    DATA_PATH = Path('../input/petfinder-pawpularity-score')
    OUTPUT_DIR = Path('./')
    MODEL_NAME = 'swin_large_patch4_window7_224'
    batch_size = 64
    image_size = 224
    fold = 5
    epoch = 20
    lr = 1e-5
    device='cuda'
    training_step=False
    pretrained=True
    SEED=999
    TTA = True
    MIX_UP = True
    MASK = False
if not os.path.isdir(CONFIG.OUTPUT_DIR):
    os.makedirs(CONFIG.OUTPUT_DIR)
LOGGER = init_logger(OUTPUT_DIR=CONFIG.OUTPUT_DIR)
fix_seed(CONFIG.SEED)


# In[5]:


train_df = pd.read_csv(CONFIG.DATA_PATH / 'train.csv')
train_df['path'] = train_df['Id'].map(lambda x:str(CONFIG.DATA_PATH/'train'/x)+'.jpg')
train_df['mask_path'] = train_df['Id'].map(lambda x:str(CONFIG.DATA_PATH/'train_U2NET'/x)+'.jpg')
train_df['image_size'] = train_df['path'].apply(lambda image_id : Image.open(image_id).size)
train_df['width'] = train_df['image_size'].apply(lambda x: x[0])
train_df['height'] = train_df['image_size'].apply(lambda x: x[1])

test_df = pd.read_csv(CONFIG.DATA_PATH / 'test.csv')
test_df['path'] = test_df['Id'].map(lambda x:str(CONFIG.DATA_PATH/'test'/x)+'.jpg')
test_df['image_size'] = test_df['path'].apply(lambda image_id : Image.open(image_id).size)
test_df['width'] = test_df['image_size'].apply(lambda x: x[0])
test_df['height'] = test_df['image_size'].apply(lambda x: x[1])

if CONFIG.MASK:
    train_df['mask_path'] = train_df['Id'].map(lambda x:str(CONFIG.DATA_PATH/'train_U2NET'/x)+'.jpg')
    test_df['mask_path'] = test_df['Id'].map(lambda x:str(CONFIG.DATA_PATH/'test_U2NET'/x)+'.jpg')
    
train_df.head()


# In[6]:


num_bins = int(np.floor(1+(3.3)*(np.log2(len(train_df)))))
train_df = get_train_data(train_df, train_df['Pawpularity'], n_splits = CONFIG.fold, regression=True, num_bins=num_bins)


# In[7]:


train_aug = albumentations.Compose(
    [
    albumentations.Resize(CONFIG.image_size, CONFIG.image_size, p=1),
    albumentations.HueSaturationValue(
            hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
        ),
    albumentations.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
        ),
    albumentations.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=255.0,
        p=1.0,)],p=1.0,
)
test_aug = albumentations.Compose(
    [
        albumentations.Resize(CONFIG.image_size, CONFIG.image_size, p=1),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    assert alpha > 0, "alpha should be larger than 0"
    assert x.size(0) > 1, "Mixup cannot be applied to a single instance."

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0])
    mixed_x = lam * x + (1 - lam) * x[rand_index, :]
    target_a, target_b = y, y[rand_index]
    return mixed_x, target_a, target_b, lam


# In[8]:


class PawpularDataset:
    def __init__(self, df, targets, augmentations, mask=True):
        self.image_paths = df['path'].tolist()
        self.mask = mask
        if self.mask:
            self.mask_paths = df['mask_path'].tolist()
        self.targets = targets
        if self.targets is None:
            self.targets = torch.ones(len(self.image_paths))
        self.augmentations = augmentations
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, item):
        image = cv2.imread(self.image_paths[item])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.mask:
            mask = cv2.imread(self.mask_paths[item])
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            idx = np.where(mask > 127.5)
            h_max = idx[0].min()
            h_min = idx[0].max()
            w_max = idx[1].min()
            w_min = idx[1].max()
        
            image = image[h_max:h_min, w_max:w_min,:]
            mask = mask[h_max:h_min, w_max:w_min]
            image = image * np.expand_dims(mask > 127.5, 2)
            
            if self.augmentations is not None:
                augmented = self.augmentations(image=image)
                image = augmented["image"]
        else:
            if self.augmentations is not None:
                augmented = self.augmentations(image=image)
                image = augmented["image"]

        targets = self.targets[item]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.float)
        }


# In[9]:


class PawpularModel(nn.Module):
    def __init__(self, pet_classify_model, model_name):
        super().__init__()
        self.pet_classify_model = pet_classify_model
        self.pet_classify_model.requires_grad = False
        self.model = timm.create_model(model_name, pretrained=CONFIG.pretrained, in_chans=3)
#         self.model.patch_embed.proj=nn.Conv2d(4, 96, kernel_size=(4, 4), stride=(4, 4))
        self.model.head = nn.Linear(self.model.head.in_features, 128)
        self.dropout = nn.Dropout(0.1)
        self.dense1 = nn.Linear(128+37, 64)
        self.dense2 = nn.Linear(64, 1)

    def forward(self, image):
        p = self.pet_classify_model(F.adaptive_avg_pool2d(image, (224,224)))
        p = torch.softmax(p, dim=1)
        x = self.model(image)
        x = self.dropout(x)
        x = torch.cat([x, p], dim=1)
        x = self.dense1(x)
        x = self.dense2(x)
        return torch.sigmoid(x.squeeze(1))
    
class pet_categor_extract_model(nn.Module):
    def __init__(self,class_num):
        super().__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=False, in_chans=3)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 128)
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(128
                               , class_num)

    def forward(self, image):
        x = self.model(image)
        x = self.dropout(x)
        x = self.dense(x)
        return x.squeeze(1)


# In[10]:


def train_fn(train_loader, model, criterion, optimizer, scheduler, batch_size, epoch, device):
    start = end = time.time()
    losses = AverageMeter()
    model.train()
    for iter, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()
        img ,target = data['image'], data['targets']
        img = img.to(device)
        target = target.to(device)
        if torch.rand(1)[0] < 0.5 and CONFIG.MIX_UP:
            mix_images, target_a, target_b, lam = mixup(img, target, alpha=0.5)
            y_preds = model(mix_images)
            loss = criterion(y_preds, target_a) * lam + (1 - lam) * criterion(y_preds, target_b)
        else:
            y_preds = model(img)
            loss = criterion(y_preds, target)
        # record loss
        losses.update(loss.item(), batch_size)
        loss.backward()
        optimizer.step()
    scheduler.step()
    return losses.avg

def valid_fn(valid_loader, model, criterion, device):
    start = end = time.time()
    losses = AverageMeter()

    # switch to evaluation mode
    model.eval()
    preds = []

    for iter, data in enumerate(valid_loader):
        img ,target = data['image'], data['targets']
        img = img.to(device)
        target = target.to(device)
        batch_size = target.size(0)

        # compute loss
        with torch.no_grad():
            y_preds = model(img)

        loss = criterion(y_preds, target)
        losses.update(loss.item(), batch_size)

        # record score
        preds.append(y_preds.to("cpu").numpy())

    predictions = np.concatenate(preds)
    return losses.avg, predictions


# In[11]:


def train_loop(train, fold_):
    LOGGER.info(f"========== fold: {fold_} training ==========")

    # ====================================================
    # Data Loader
    # ====================================================
    cl_model = pet_categor_extract_model(class_num=37)
    cl_model.to(CONFIG.device)
    cl_model.load_state_dict(fix_model_state_dict(torch.load('../input/pretrained_models/efficientnet_b0_Oxford_classifier_size_224.pth')["model"]))

    model = PawpularModel(cl_model, model_name=CONFIG.MODEL_NAME)
    model.to(CONFIG.device)
    
    earlystopping= EarlyStopping(patience=3, path=CONFIG.OUTPUT_DIR / f"{CONFIG.MODEL_NAME}_{fold_}_latest.pth")
    
    if torch.cuda.device_count()>1:
        model=nn.DataParallel(model)
    dense_features = [
        'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
        'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
    ]
    train_idx = train[train.fold!=fold_].index
    val_idx = train[train.fold ==fold_].index
    train_folds = train.loc[train_idx].reset_index(drop=True)
    valid_folds = train.loc[val_idx].reset_index(drop=True)
    train_dataset = PawpularDataset(
        train_folds, targets=train_folds['Pawpularity']/100,
        augmentations=train_aug, mask=CONFIG.MASK
    )
    
    val_dataset = PawpularDataset(
        valid_folds, targets=valid_folds['Pawpularity']/100,
        augmentations=test_aug, mask=CONFIG.MASK
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
    )
    valid_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=False,
    )

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-4)
    criterion = nn.BCELoss()
    metric = RMSE()
    best_score = np.inf
    best_loss = np.inf
    
    for epoch in range(CONFIG.epoch):
        start_time = time.time()
        # train
        avg_loss = train_fn(train_loader, model, criterion, optimizer, scheduler, CONFIG.batch_size, epoch, CONFIG.device)
#        # eval
        if CONFIG.TTA:
            eval_model = tta.ClassificationTTAWrapper(model, tta.aliases.hflip_transform())
        else:
            eval_model = model
        avg_val_loss, preds = valid_fn(valid_loader, eval_model, criterion, CONFIG.device)
        valid_labels = torch.tensor(valid_folds["Pawpularity"].values).float()
        score = metric(preds * 100, valid_labels)
        elapsed = time.time() - start_time
        
        LOGGER.info(
            f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  lr: {scheduler.get_lr()[0]:.8f} time: {elapsed:.0f}s"
        )
        LOGGER.info(f"Epoch {epoch+1} - Score: {score}")
        if score < best_score:
            best_score = score
            LOGGER.info(f"Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model")
            torch.save(
                {"model": model.state_dict(), "preds": preds * 100}, CONFIG.OUTPUT_DIR / f"{CONFIG.MODEL_NAME}_{fold_}_best.pth")
            
        earlystopping(avg_val_loss, model) #callメソッド呼び出し
        if earlystopping.early_stop: #ストップフラグがTrueの場合、breakでforループを抜ける
            print("Early Stopping!")
            break
            
    check_point = torch.load(CONFIG.OUTPUT_DIR / f"{CONFIG.MODEL_NAME}_{fold_}_best.pth")

    valid_folds["preds"] = check_point["preds"]

    return valid_folds


# In[12]:


def get_result(result_df):
    metric = RMSE()
    preds = result_df["Pawpularity"].values
    labels = result_df["preds"].values
    score = metric(preds, labels)
    LOGGER.info(f"Score: {score:<.5f}")


# In[13]:


def inference():
    predictions = []
    dense_features = [
        'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
        'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
    ]
    test_dataset = PawpularDataset(
        test_df, targets=None,
        augmentations=test_aug, mask=CONFIG.MASK
    )
    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
    )

    for fold in range(5):
        LOGGER.info(f"========== model: {CONFIG.MODEL_NAME} fold: {fold} inference ==========")
        cl_model = pet_categor_extract_model(class_num=37)
        cl_model.to(CONFIG.device)
        cl_model.load_state_dict(fix_model_state_dict(torch.load('../input/pretrained_models/efficientnet_b0_Oxford_classifier_size_224.pth')["model"]))

        model = PawpularModel(cl_model, model_name=CONFIG.MODEL_NAME)
        model.to(CONFIG.device)
        model.load_state_dict(fix_model_state_dict(torch.load(CONFIG.OUTPUT_DIR / f"{CONFIG.MODEL_NAME}_{fold}_best.pth")["model"]))
        
        if torch.cuda.device_count()>1:
            model=nn.DataParallel(model)
        
        if CONFIG.TTA:
            eval_model = tta.ClassificationTTAWrapper(model, tta.aliases.hflip_transform())
        else:
            eval_model = model
        model.eval()
        preds = []
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            img,target = data['image'], data['targets']
            img = img.to(CONFIG.device)
            target = target.to(CONFIG.device)
            with torch.no_grad():
                y_preds = model(img)
            preds.append(y_preds.to("cpu").numpy())
        preds = np.concatenate(preds)
        predictions.append(preds)
    predictions = np.mean(predictions, axis=0)
    return predictions * 100


# In[14]:


def main():
    # Training
    oof_df = pd.DataFrame()
    if CONFIG.training_step:
        for fold in range(CONFIG.fold):
            _oof_df = train_loop(train_df, fold)
            oof_df = pd.concat([oof_df, _oof_df])
            LOGGER.info(f"========== fold: {fold} result ==========")
            get_result(_oof_df)
        # Save OOF result
        oof_df.to_csv(CONFIG.OUTPUT_DIR / "oof_df.csv", index=False)
    else:
        oof_df = pd.read_csv(CONFIG.OUTPUT_DIR / "oof_df.csv")
    # CV result
    LOGGER.info(f"========== CV ==========")
    get_result(oof_df)
    # Inference
    predictions = inference()
    # submission
    submission = test_df.copy()
    submission["Pawpularity"] = predictions
    submission = submission[["Id", "Pawpularity"]]
    submission.to_csv(CONFIG.OUTPUT_DIR / "submission.csv", index=False)


# In[15]:


if __name__ == "__main__":
    main()