"""
For training and tesing a brain aging prediction model in human/marmoset
based on a densely connected convolutional network  

Writted by Yang Xiao, Lei Zhao, Shuer Ye in Jun 17, 2025

@: xiaoyang9604@gmail.com

"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset, random_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import numpy as np
import nibabel as nib 
import csv
import re
import pandas as pd
import glob
import time

from a_DenseNetAging import DenseNet3DMRI
from b_SaliencyMap import GuidedGradCAM

def load_nii_as_tensor(nii_path):
      img = nib.load(nii_path)
      data = img.get_fdata()
      affine = img.affine
      data = np.nan_to_num(data, nan=0.0)
    #   data = (data - np.mean(data)) / (np.std(data) + 1e-5)
      data = torch.tensor(data, dtype = torch.float32)
      data = data.unsqueeze(0)
      return data, affine

class GMVDataset(Dataset):
    def __init__(self, img_path):
       self.samples = []
       self.labels = []
       self.subj = []  

       for folder in sorted(os.listdir(img_path)):
          folder_path = os.path.join(img_path, folder)
          if not os.path.isdir(folder_path):
              continue
          
          demo_file = os.path.join(folder_path, 'Demography_cn.csv')
          if not os.path.exists(demo_file):
            continue
          
          df = pd.read_csv(demo_file)
          age_dict = {str(k): v for k, v in zip(df["Subjid"], df["Age"])}

          gmv_folder = os.path.join(folder_path, "GMV_Rename")
          if not os.path.isdir(gmv_folder):
              continue
          
          folder_count = 0
          for fname in sorted(os.listdir(gmv_folder)):
                    prefix = folder + "_"
                    if fname.endswith('.nii'): # or fname.endswith('.nii.gz')
                        # match = re.search(r'sub-[^_\.]+', fname)
                        # if not match:
                        #     continue
                        # subjid = match.group(0)
                        subjid = os.path.splitext(fname)[0] 
                        subjid = subjid[len(prefix):]
                        subjid = str(subjid)

                        if subjid not in age_dict:
                            continue

                        nii_path = os.path.join(gmv_folder, fname)  
                        self.samples.append(nii_path)
                        self.labels.append(float(age_dict[subjid]))
                        self.subj.append(subjid)
                        folder_count += 1

          print(f"Folder '{folder}': matched {folder_count} samples")
       print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, affine = load_nii_as_tensor(self.samples[idx])
        y = torch.tensor(self.labels[idx], dtype = torch.float32)
        subjid = self.subj[idx]
        return X, y, affine, subjid

# model training
def train(model, train_loader, val_loader, device, epochs = 30, lr = 1e-4, fold = 1, patience = 20, save_path='./model', resume_training = False):
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr = lr, betas = (0.9, 0.999), weight_decay = 1e-04) # betas from Nature Aging, 2021
    criterion = nn.L1Loss() # mae loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_mae = float('inf')
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, f'best_mdl_fold{fold}.pt')
    
    start_epoch = 0
    if resume_training:
        checkpoint_pattern = os.path.join(save_path, f'checkpoint_fold{fold}_epoch*.pt')
        checkpoint_files = glob.glob(checkpoint_pattern)

        checkpoint_files.sort(key=lambda x: int(x.split('_epoch')[-1].split('.')[0]))
        latest_checkpoint = checkpoint_files[-1]
        checkpoint = torch.load(latest_checkpoint)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_mae = checkpoint.get('best_mae', float('inf'))
        patience_counter = checkpoint.get('patience_counter', 0)
    
        print(f"[Fold {fold}] Resumed training from epoch {start_epoch}, best mae = {best_mae:.4f}")

    # Early stopping tracker
    patience_counter = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch, _, _ in tqdm(train_loader, desc = f"[Fold {fold}] Epoch {epoch+1}/{epochs}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze(dim=1)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        
		# model validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch, _, _ in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch).squeeze(dim=1)
                val_preds.extend(y_pred.cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())
                
        val_mae = mean_absolute_error(val_targets, val_preds)
        pearson_r, _ = pearsonr(val_targets, val_preds)
        spearman_r, _ = spearmanr(val_targets, val_preds)
        print(f'[Fold {fold}] Epoch {epoch+1}: '
              f'Training Loss = {train_loss: .4f}, '
              f'Validate MAE = {val_mae: .4f}, '
              f'Pearson r = {pearson_r:.4f}, '
              f'Spearman r = {spearman_r:.4f}')
              
        scheduler.step(val_mae)

        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(save_path, f'checkpoint_fold{fold}_epoch{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_mae': val_mae,
                'best_mae': best_mae,
                'patience_counter': patience_counter
            }, ckpt_path)
            print(f'[Fold {fold}] Checkpoint saved at epoch {epoch+1}')

		# save model
        if val_mae < best_mae:
            best_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f'[Fold {fold}] model updated. Best validate MAE = {best_mae: .4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'[Fold {fold}] Early stopping at epoch {epoch+1}')
                break
            
    return model_path, best_mae

def test(model, test_loader, device, model_path, map_save_path, target_layer = 'feas.denseblock4'):
    model.load_state_dict(torch.load(model_path, weights_only = True))
    model.to(device)
    model.eval()
    preds, targets = [], []
    # guided_gradcam = GuidedGradCAM(model, target_layer)

    for X_batch, y_batch, affines, subjids in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        with torch.no_grad():
            y_pred = model(X_batch).squeeze(dim=1)
            preds.extend(y_pred.cpu().numpy())
            targets.extend(y_batch.cpu().numpy())

        # for i in range(X_batch.size(0)):
        #     saliency = guided_gradcam(X_batch[i:i+1], device)
        #     affine = affines[i].numpy()
        #     subj_id = subjids[i]
        #     save_path = os.path.join(map_save_path, f'Guided_Grad-CAM_sample_{subj_id}.nii')
        #     nib.save(nib.Nifti1Image(saliency[0,0], affine = affine), save_path)
            
    mae = mean_absolute_error(targets, preds)
    pearson_r, _ = pearsonr(targets, preds)
    spearman_r, _ = spearmanr(targets, preds)

    return preds, targets, mae, pearson_r, spearman_r

def main():

    config = {
        'epoch_size': 50,
        'batch_sizes': [5],
        'lrs': [1e-4],
        'k_folds': 5,
        'img_path': 'D:/_workings/projects/DenseNetAging/0_datasets/',
        'save_path': 'D:/_workings/projects/DenseNetAging/results/1_model/',
        'saliency_path': 'D:/_workings/projects/DenseNetAging/results/2_saliency/',
        'resume_training': True
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    torch.backends.cudnn.benchmark = True
    start_time = time.time()
            
    dataset = GMVDataset(config['img_path'])
    subj = dataset.subj

    ages = np.array(dataset.labels)
    age_bins, bin_edges = pd.qcut(ages, q = 5, labels=False, retbins=True)
    print(age_bins, bin_edges)
    bin_counts = pd.Series(age_bins).value_counts().sort_index()
    bin_ranges = [f"({bin_edges[i]: .1f}, {bin_edges[i+1]: .1f})" for i in range(len(bin_edges) - 1)]
    bin_summary = pd.DataFrame({
        'Bin': range(5),
        'Age Range': bin_ranges,
        'Sample Count': bin_counts
    })
    print(bin_summary.to_string(index=False))

	# split dataset
    # kf = KFold(n_splits = k_folds, shuffle = True, random_state = 1)
    skf = StratifiedKFold(n_splits = config['k_folds'], shuffle = True, random_state = 1)

    all_test_mae = []
    all_test_pearsonr = []
    all_test_spearmanr = []
    
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(np.zeros(len(age_bins)), age_bins)): #enumerate(kf.split(dataset)):
        test_set = Subset(dataset, test_idx)
        test_subj = [subj[i] for i in test_idx]

        # remaining_set = Subset(dataset, train_val_idx)
        # train_len = int(0.75 * len(remaining_set)) # 60% = 0.75 * 0.8; is there some good ideas?
        # val_len = len(remaining_set) - train_len
        # train_set, val_set = random_split(remaining_set, [train_len, val_len], 
        #                                   generator = torch.Generator().manual_seed(1))
        
        remaining_bins = age_bins[train_val_idx]
        skf_inter = StratifiedShuffleSplit(n_splits = 1, train_size = 0.8, test_size = 0.2, random_state = 1)
        train_idx_rel, val_idx_rel = next(skf_inter.split(train_val_idx, remaining_bins))
        train_idx = train_val_idx[train_idx_rel]
        val_idx = train_val_idx[val_idx_rel]
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)

        # Grid search for best lr and batch size based on val_mae
        best_val_mae = float('inf')
        best_lr = None
        best_bs = None
        best_model_path = None

        for lr in config['lrs']:
            for bs in config['batch_sizes']:

                print(f"[Fold {fold+1}] trying lr = {lr}, batch size = {bs}")

                train_loader = DataLoader(train_set, batch_size = bs, shuffle = True, num_workers = 4, pin_memory=True)
                val_loader = DataLoader(val_set, batch_size = bs, num_workers = 4, pin_memory=True)

                model = DenseNet3DMRI()
                temp_save_path = os.path.join(config['save_path'], f'temp_lr{lr}_bs{bs}_fold{fold+1}')
                model_path, val_mae = train(model, train_loader, val_loader, device, 
                                            epochs = config['epoch_size'], lr=lr, fold = fold + 1, 
                                            save_path = temp_save_path, resume_training=False)

                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_lr = lr
                    best_bs = bs
                    best_model_path = model_path

        print(f'[Fold {fold+1}], '
              f'Best lr = {best_lr: .4f}, '
              f'Best batch size = {best_bs: .1f}, '
              f'Best validated mae = {best_val_mae:.4f}')

        test_loader = DataLoader(test_set, batch_size = best_bs, num_workers = 4, pin_memory=True)
        
        model = DenseNet3DMRI()
        model.to(device)
        preds, targets, mae, pearsonr, spearmanr = test(model, test_loader, device, best_model_path, config['saliency_path'])
        
        all_test_mae.append(mae)
        all_test_pearsonr.append(pearsonr)
        all_test_spearmanr.append(spearmanr)
        print(f'Fold {fold+1} Testing MAE = {mae: .4f}, Pearson r = {pearsonr: .4f}, Spearman r = {spearmanr: .4f}')
        df = pd.DataFrame({'subjid': test_subj, 'pred': preds, 'target': targets})
        df.to_csv(os.path.join(config['save_path'], f'results_fold{fold+1}.csv'), index = False)
        
    for i, (mae, p_r, s_r) in enumerate(zip(all_test_mae, all_test_pearsonr, all_test_spearmanr)):
        print(f"Fold {i+1}: MAE = {mae: .4f}, Pearson r = {p_r: .4f}, Spearman r = {s_r: .4f}")

    print(f'average MAE: {np.mean(all_test_mae): .4f} ({np.std(all_test_mae):.4f})')   
    print(f'Average Pearson r: {np.mean(all_test_pearsonr):.4f} ({np.std(all_test_pearsonr):.4f})')
    print(f'Average Spearman r: {np.mean(all_test_spearmanr):.4f} ({np.std(all_test_spearmanr):.4f})')
    end_time = time.time()
    print(f'Total time: {end_time - start_time: .2f} seconds')     

if __name__ == "__main__":
    main()

