"""
For training and tesing a brain aging prediction model in human/marmoset
based on a traditional machine learning [Lasso, Ridge, SVR, LightGBM]

Writted by Yang Xiao, Lei Zhao, Shuer Ye in Sep 25, 2025

@: xiaoyang9604@gmail.com

"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
import time
from collections import defaultdict

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from joblib import Memory
import joblib

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, spearmanr

from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import SVR
import lightgbm as lgb

CACHE_DIR = './cachedir'
memory = Memory(CACHE_DIR, verbose=0)

def load_and_flatten_data(img_path, mmap_filename='data_flat.mmap', y_filename='ages.npy', subj_filename='subjects.npy'):
        os.makedirs(CACHE_DIR, exist_ok=True)

        mmap_path = os.path.join(CACHE_DIR, mmap_filename)
        y_path = os.path.join(CACHE_DIR, y_filename)
        subj_path = os.path.join(CACHE_DIR, subj_filename)

        if os.path.exists(mmap_path) and os.path.exists(y_path) and os.path.exists(subj_path):
            print("Loading X, y, subjects from cached .npy/.mmap ...")
            X_memmap = np.memmap(mmap_path, dtype='float32', mode='r')
            n_samples = np.load(y_path).shape[0]
            n_features = X_memmap.size // n_samples
            X_memmap = X_memmap.reshape((n_samples, n_features))
            y = np.load(y_path)
            subjects = np.load(subj_path, allow_pickle=True).tolist()
            return X_memmap, y, subjects


        samples_info = []
        
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
                    samples_info.append({'path': nii_path, 
                                            'age': float(age_dict[subjid]), 
                                            'subjid': subjid})
                    folder_count += 1

            print(f"Folder '{folder}': matched {folder_count} samples")
        print(f"Total samples: {len(samples_info)}")

        if len(samples_info) == 0:
            raise RuntimeError(f"No samples found under {img_path}")
            
        first_img_data = nib.load(samples_info[0]['path']).get_fdata()
        n_features = first_img_data.flatten().shape[0]
        
        n_features=100000

        mmap_path = os.path.join(CACHE_DIR, mmap_filename)
        X_memmap = np.memmap(mmap_path, dtype='float32', mode='w+', shape=(len(samples_info), n_features))
        y = np.zeros(len(samples_info), dtype='float32')
        # X_flat = []
        # y = []
        subjects = []

        for i, sample in enumerate(tqdm(samples_info, desc="Loading imaging data...", total=len(samples_info))):
            try:
                img = nib.load(sample['path'])
                data = img.get_fdata()
                data = np.nan_to_num(data, nan=0.0)
                X_memmap[i, :] = data.flatten()[:n_features]
                y[i] = sample['age']
                # X_flat.append(data.flatten())
                # y.append(sample['age'])
                subjects.append(sample['subjid'])
            except Exception as e:
                print(f"Warning：reading {sample['path']} yeild error: {e}")
        X_memmap.flush()  
        np.save(y_path, y)
        np.save(subj_path, np.array(subjects, dtype=object))  
        # return np.array(X_flat), np.array(y), subjects
        return X_memmap, y, subjects

class PCASubsetSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.n_components is None:
            return X
        return X[:, :self.n_components]
    
@memory.cache
def prepare_and_cache_pca_fold(train_idx, test_idx, data_hash, max_pca_components, img_path):
    X_memmap, ages, _ = load_and_flatten_data(img_path)
    
    train_idx = list(train_idx)
    test_idx = list(test_idx)
    
    X_train_high_dim, X_test_high_dim = X_memmap[train_idx], X_memmap[test_idx]
    y_train, y_test = ages[train_idx], ages[test_idx]
    
    scaler = StandardScaler() #非正态RobustScaler(), StandardScaler()
    batch_size = 1000
    for i in range(0, X_train_high_dim.shape[0], batch_size):
        batch = X_train_high_dim[i:i+batch_size]
        scaler.partial_fit(batch)
    X_train_scaled_path = os.path.join(CACHE_DIR, 'X_train_scaled.mmap')
    X_test_scaled_path = os.path.join(CACHE_DIR, 'X_test_scaled.mmap')
    X_train_scaled = np.memmap(X_train_scaled_path, dtype='float32', mode='w+', shape=X_train_high_dim.shape)
    X_test_scaled  = np.memmap(X_test_scaled_path,  dtype='float32', mode='w+', shape=X_test_high_dim.shape)

    for i in range(0, X_train_high_dim.shape[0], batch_size):
        X_train_scaled[i:i+batch_size] = scaler.transform(X_train_high_dim[i:i+batch_size])
    for i in range(0, X_test_high_dim.shape[0], batch_size):
        X_test_scaled[i:i+batch_size] = scaler.transform(X_test_high_dim[i:i+batch_size])

    X_train_scaled.flush()
    X_test_scaled.flush()

    # pca = PCA(n_components=max_pca_components, random_state=1)
    pca = IncrementalPCA(n_components=max_pca_components, batch_size=batch_size)
    for i in range(0, X_train_scaled.shape[0], batch_size):
        batch = X_train_scaled[i:i+batch_size]
        batch = batch.astype(np.float32)
        pca.partial_fit(batch)
    X_train_pca_path = os.path.join(CACHE_DIR, 'X_train_pca.mmap')
    X_test_pca_path  = os.path.join(CACHE_DIR, 'X_test_pca.mmap')
    X_train_pca = np.memmap(X_train_pca_path, dtype='float32', mode='w+', shape=(X_train_scaled.shape[0], max_pca_components))
    X_test_pca  = np.memmap(X_test_pca_path,  dtype='float32', mode='w+', shape=(X_test_scaled.shape[0], max_pca_components))

    for i in range(0, X_train_scaled.shape[0], batch_size):
        X_train_pca[i:i+batch_size] = pca.transform(X_train_scaled[i:i+batch_size].astype(np.float32))
    for i in range(0, X_test_scaled.shape[0], batch_size):
        X_test_pca[i:i+batch_size] = pca.transform(X_test_scaled[i:i+batch_size].astype(np.float32))

    X_train_pca.flush()
    X_test_pca.flush()
    
    return X_train_pca, X_test_pca, y_train, y_test


def main():
    IMG_PATH = 'D:/_workings/projects/DenseNetAging/0_datasets/'
    SAVE_PATH = 'D:/_workings/projects/DenseNetAging/results/2_TraditionalModel/'
    os.makedirs(SAVE_PATH, exist_ok=True)
    K_FOLDS = 5
    MAX_PCA_COMPONENTS = 500

    param_grids = {
        'Lasso': {
            'selector__n_components': Integer(100, MAX_PCA_COMPONENTS),
            'model__alpha': Real(1e-7, 1e0, prior='log-uniform')
        },

        'Ridge': {
            'selector__n_components': Integer(100, MAX_PCA_COMPONENTS),
            'model__alpha': Real(1e-6, 1e3, prior='log-uniform'),
            'model__fit_intercept': Categorical([True, False]),
            'model__solver': Categorical(['auto', 'svd', 'cholesky', 'lsqr', 'saga']),
            'model__max_iter': Categorical([100, 500, 1000, 5000])
        },

        'SVR': {
            'selector__n_components': Integer(100, MAX_PCA_COMPONENTS),
            'model__C': Real(1e-4, 1e4, prior='log-uniform'),
            'model__gamma': Real(1e-7, 1e2, prior='log-uniform')
        },

        'LightGBM': {
            'selector__n_components': Integer(100, MAX_PCA_COMPONENTS),
            'model__num_leaves': Integer(2, 19),
            "model__max_depth": Categorical([-1, 2, 3, 4, 5, 10, 20]),
            'model__learning_rate': Real(0.0001, 0.1, prior='log-uniform'),
            # "model__min_data_in_leaf": Categorical([i for i in range(1, 45, 2)]),
            # "model__bagging_fraction": Categorical([i / 10. for i in range(2, 11)]),
            # "model__bagging_freq": Categorical([0, 1, 2, 3, 4, 5]),
            # 'model__lambda_l1': Categorical([0, 0.001, 0.005, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3]),
            'model__n_estimators': Categorical([i for i in range(100, 1000, 50)]),
            # 'model__feature_fraction': Categorical([i / 10. for i in range(2, 11)]), # when using dummy variables
        }
    }

    pipelines = {
        'Lasso': Pipeline([
            ('selector', PCASubsetSelector()),
            ('model', Lasso(random_state=1, 
                            tol=1e-4, 
                            max_iter=5000))
        ]),
        'Ridge': Pipeline([
            ('selector', PCASubsetSelector()),
            ('model', Ridge(random_state=1))
        ]),
        'SVR': Pipeline([
            ('selector', PCASubsetSelector()),
            ('model', SVR())
        ]),
        'LightGBM': Pipeline([
            ('selector', PCASubsetSelector()),
            ('model', lgb.LGBMRegressor(metric = 'l2', num_threads=1,
                                        random_state=1, verbose = -1, n_jobs=1))
        ])
    }

    start_time = time.time()
    X, ages, subjid = load_and_flatten_data(IMG_PATH)
    data_hash = joblib.hash(X)
    print(f"Feature's dimension: {X.shape}, Label's dimension: {ages.shape}")

    age_bins, bin_edges = pd.qcut(ages, q = 5, labels=False, retbins=True)
    bin_counts = pd.Series(age_bins).value_counts().sort_index()
    bin_ranges = [f"({bin_edges[i]: .1f}, {bin_edges[i+1]: .1f})" for i in range(len(bin_edges) - 1)]
    bin_summary = pd.DataFrame({
        'Bin': range(5),
        'Age Range': bin_ranges,
        'Sample Count': bin_counts
    })
    print(bin_summary.to_string(index=False))

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=1)
    for i, (train_idx, test_idx) in enumerate(skf.split(X, age_bins)):
        prepare_and_cache_pca_fold(tuple(train_idx), tuple(test_idx), data_hash, MAX_PCA_COMPONENTS, IMG_PATH)

    results_summary = []
    for name in pipelines:
        print(f"\n--- Model traing: {name} ---")

        y_pred_df = pd.DataFrame({'subject': subjid, 'true': ages, f'pred_{name}': np.nan})
        fold_results = defaultdict(list)

        skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=1)
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, age_bins)):
            X_train, X_test, y_train, y_test = prepare_and_cache_pca_fold(tuple(train_idx), 
                                                                                  tuple(test_idx), 
                                                                                  data_hash, 
                                                                                  MAX_PCA_COMPONENTS, 
                                                                                  IMG_PATH)

            print('CV iteration: %d' % (fold))
            print('Training set size: %d' % len(y_train))
            print('Test set size: %d' % len(y_test))
            print('Number of features = %d' % X_train.shape[1])
            
            # print('GridSearch begins...')
            # grid = GridSearchCV(pipeline, 
                                # param_grid=params,  
                                # cv=3, 
                                # scoring="neg_mean_absolute_error",
                                # n_jobs=-1, 
                                # verbose=1)
            # grid.fit(X_train, y_train)
            # print('Best: %f using %s' % (grid.best_score_, grid.best_params_))
            # best_pipeline = grid.best_estimator_

            # print('RandomizedSearch begins...')
            # random_search = RandomizedSearchCV(model, 
                                                # param_distributions = param_dist,
                                                # n_iter = rscv_kwargs.get('n_random', 10),
                                                # random_state = rscv_kwargs.get('random_seed', 42),
                                                # cv = 5, 
                                                # n_jobs = -1, 
                                                # verbose = 1, 
                                                # scoring = 'r2')
            
            # random_results = random_search.fit(X_train_norm, y_train)
            # print('Best: %f using %s' % (random_results.best_score_, random_results.best_params_))
            # best_model = random_search.best_estimator_

            print('Bayessearch begins...')
            grid = BayesSearchCV(pipelines[name], 
                                        search_spaces = param_grids[name],
                                        n_iter = 32,
                                        random_state = 1,
                                        cv = 3, 
                                        n_jobs = 1, 
                                        verbose = 1, 
                                        scoring = 'neg_mean_absolute_error')
            
            grid.fit(X_train, y_train)
            print('Best: %f using %s' % (grid.best_score_, grid.best_params_))
            best_pipeline = grid.best_estimator_

            preds = best_pipeline.predict(X_test)
            y_pred_df.loc[test_idx, f'pred_{name}'] = preds
            
            mae = mean_absolute_error(y_test, preds)
            pearson_r, _ = pearsonr(y_test, preds)
            spearman_r, _ = spearmanr(y_test, preds)
            
            fold_results['mae'].append(mae)
            fold_results['pearsonr'].append(pearson_r)
            fold_results['spearmanr'].append(spearman_r)
            print(f'Fold {fold+1} Testing MAE = {mae: .4f}, Pearson r = {pearson_r: .4f}, Spearman r = {spearman_r: .4f}')
            
            joblib.dump(best_pipeline, os.path.join(SAVE_PATH, f'{name}_best_model_fold{fold+1}.joblib'))
            results_path = os.path.join(SAVE_PATH, f'{name}_cv_results_fold{fold+1}.csv')
            pd.DataFrame(grid.cv_results_).to_csv(results_path, index=False)
    
        y_pred_df.to_csv(os.path.join(SAVE_PATH, f'y_pred_results_{name}.csv'), index=False)

        mean_mae = np.mean(fold_results['mae'])
        std_mae = np.std(fold_results['mae'])
        mean_p = np.mean(fold_results['pearsonr'])
        std_p = np.std(fold_results['pearsonr'])
        mean_s = np.mean(fold_results['spearmanr'])
        std_s = np.std(fold_results['spearmanr'])

        print(f'average MAE: {mean_mae: .4f} ({std_mae:.4f})')   
        print(f'Average Pearson r: {mean_p:.4f} ({std_p:.4f})')
        print(f'Average Spearman r: {mean_s:.4f} ({std_s:.4f})')
        
        results_summary.append({
            "Model": name,
            "MAE": f"{mean_mae:.4f} ± {std_mae:.4f}",
            "Pearson r": f"{mean_p:.4f} ± {std_p:.4f}",
            "Spearman r": f"{mean_s:.4f} ± {std_s:.4f}"
        })

    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))
    end_time = time.time()
    print(f'Total time: {(end_time - start_time) / 60:.2f} min')

if __name__ == "__main__":
    main()
