import os
import random
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
import scipy
from tqdm import tqdm
import pickle
from argparse import ArgumentParser

from embeddings import blosum_embedding, ablang_embedding

def setup_args():
    """Parse command-line arguments for Gaussian Process training"""
    parser = ArgumentParser()
    parser.add_argument('--dms_input', type=str, default='')
    parser.add_argument('--model_type', type=str, default='ablang_gp')
    parser.add_argument('--output_dir', type=str, default='tmp')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_input', type=str, default=None)
    parser.add_argument('--label_col', type=str, default='DMS_score')
    parser.add_argument('--kernel', type=str, default='rbf')
    parser.add_argument('--add_kernel', action='store_true', default=True)
    return parser.parse_args()

def seed_everything(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def get_embeddings(dataset, embedding_type = "ablang2", label_col = "DMS_score"):
    """Extract embeddings from dataset"""
    if embedding_type == "ablang2":
        paired_sequences = dataset[["heavy", "light"]].values
        embeddings = ablang_embedding(paired_sequences)
    elif embedding_type == "blosum62":
        embeddings = np.array(list(map(blosum_embedding, tqdm(dataset["mutseq"]))))
    else:
        raise ValueError(f"Invalid embedding type: {embedding_type}")
    if label_col in dataset.columns:
        y = dataset[label_col].values
    else:
        y = None
    return embeddings, y

def train_gp(X_train, y_train, kernel = "rbf", add_kernel = True, random_state = 42):
    """Train Gaussian Process model"""
    if kernel == "rbf":
        kernel = RBF()
    elif kernel == "matern":
        kernel = Matern(nu=2.5)
    else:
        raise ValueError(f"Invalid kernel: {kernel}")

    if add_kernel:
        kernel = WhiteKernel() + ConstantKernel() + kernel

    gp = GaussianProcessRegressor(kernel=kernel, random_state=random_state)
    gp.fit(X_train, y_train)
    return gp

def main(args):
    """Main function for training and evaluation"""
    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(f"{args.output_dir}/gp_model.pkl"):
        print(f"Model {args.output_dir}/gp_model.pkl already exists")
        exit()

    seed_everything(args.seed)

    embedding_type = "ablang2" if args.model_type == "ablang_gp" else "blosum62" if args.model_type == "blosum_gp" else None
    if embedding_type is None:
        raise ValueError(f"Invalid model type: {args.model_type}")
    # Load and preprocess data
    df_all = pd.read_csv(args.dms_input)

    # Split data
    df_train = df_all


    # Create datasets and extract embeddings
    print("Preparing training data...")
    X_train, y_train = get_embeddings(df_train, label_col=args.label_col, embedding_type=embedding_type)

    # Train GP model
    print("Training Gaussian Process model...")
    gp_model = train_gp(X_train, y_train, kernel=args.kernel, add_kernel=args.add_kernel, random_state=args.seed)

    # Save model and predictions
    print(f"Saving model to {args.output_dir}")
    with open(f'{args.output_dir}/gp_model.pkl', 'wb') as f:
        pickle.dump(gp_model, f)

    # Test predictions if test data provided
    if args.test_input:
        print("Making predictions on test data...")
        df_test = pd.read_csv(args.test_input)
        X_test, y_test = get_embeddings(df_test, label_col=args.label_col, embedding_type=embedding_type)
        test_preds, test_std = gp_model.predict(X_test, return_std=True)
        np.save(f'{args.output_dir}/test_preds.npy', test_preds)
        np.save(f'{args.output_dir}/test_uncertainties.npy', test_std)

        test_spearman = scipy.stats.spearmanr(test_preds, y_test)[0]
        print(f"Test Spearman correlation: {test_spearman:.4f}")

if __name__ == '__main__':
    args = setup_args()
    main(args)
