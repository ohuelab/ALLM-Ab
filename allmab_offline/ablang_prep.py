import os
import gc
import random
import copy
import yaml
import json
import pickle
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
import scipy
import ablang2

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel


from strategy_gp import get_selector


DMS_COL = 'DMS_score'


def parse_config(config_path):
    """
    Reads YAML configuration for active learning and model settings.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def seed_everything(seed=42):
    """
    Sets random seeds for reproducibility across libraries and environments.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


def generate_unique_id(string):
    """
    Generates a unique SHA-1 hash from a given string.
    """
    import hashlib
    hash_object = hashlib.sha1()
    hash_object.update(string.encode('utf-8'))
    return hash_object.hexdigest()


def write_log(log_file, text, is_print=True):
    """
    Writes text to a log file and optionally prints it.
    """
    if is_print:
        print(text)
    datetime = time.strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"[{datetime}] {text}\n")
    log_file.flush()


def metric(preds, labels, bottom_preds=None, bottom_labels=None,
           top_preds=None, top_labels=None, eps=1e-6):
    """
    Calculates performance metrics (Pearson, Spearman, RMSE).
    If subsets (top/bottom) are provided, also calculates precision/recall/F1.
    """
    pearson = scipy.stats.pearsonr(preds, labels)[0]
    spearman = scipy.stats.spearmanr(preds, labels)[0]
    rmse = np.mean((preds - labels) ** 2) ** 0.5
    ms = {
        'pearson': pearson,
        'spearman': spearman,
        'rmse': rmse
    }

    if top_preds is not None and bottom_preds is not None:
        all_preds = np.concatenate([bottom_preds, preds, top_preds])
        n_top = len(top_preds)
        n_bottom = len(bottom_preds)
        top_pred_idxs = np.argsort(all_preds)[-n_top:]
        top_metrics = []
        bottom_metrics = []
        unbias_metrics = []

        for k in [10, 20, 50, 100]:
            precision = (top_pred_idxs[-min(k, n_top):] >= (len(all_preds) - n_top)).mean()
            recall = (top_pred_idxs[-min(k, n_top):] >= (len(all_preds) - n_top)).sum() / n_top
            f1 = 2 * precision * recall / (precision + recall + eps)
            top_metrics.append([precision, recall, f1])

            precision = (top_pred_idxs[:min(k, n_bottom)] < n_bottom).mean()
            recall = (top_pred_idxs[:min(k, n_bottom)] < n_bottom).sum() / n_top
            f1 = 2 * precision * recall / (precision + recall + eps)
            bottom_metrics.append([precision, recall, f1])

        precision = (top_pred_idxs >= (len(all_preds) - n_top)).mean()
        recall = (top_pred_idxs >= (len(all_preds) - n_top)).sum() / n_top
        f1 = 2 * precision * recall / (precision + recall + eps)
        top_metrics.append([precision, recall, f1])

        precision = (top_pred_idxs < n_bottom).mean()
        recall = (top_pred_idxs < n_bottom).sum() / n_top
        f1 = 2 * precision * recall / (precision + recall + eps)
        bottom_metrics.append([precision, recall, f1])

        for i, _ in enumerate([10, 20, 50, 100, 'top_frac']):
            unbias_metrics.append(
                list(np.array(top_metrics[i]) - np.array(bottom_metrics[i]))
            )

        ms.update({
            'top10_precision_recall_f1': top_metrics[0],
            'top20_precision_recall_f1': top_metrics[1],
            'top50_precision_recall_f1': top_metrics[2],
            'top100_precision_recall_f1': top_metrics[3],
            'top_frac_precision_recall_f1': top_metrics[4],

            'bottom10_precision_recall_f1': bottom_metrics[0],
            'bottom20_precision_recall_f1': bottom_metrics[1],
            'bottom50_precision_recall_f1': bottom_metrics[2],
            'bottom100_precision_recall_f1': bottom_metrics[3],
            'bottom_frac_precision_recall_f1': bottom_metrics[4],

            'unbias10_precision_recall_f1': unbias_metrics[0],
            'unbias20_precision_recall_f1': unbias_metrics[1],
            'unbias50_precision_recall_f1': unbias_metrics[2],
            'unbias100_precision_recall_f1': unbias_metrics[3],
            'unbias_frac_precision_recall_f1': unbias_metrics[4],
        })

    return ms

def check_config(config):
    """
    Checks the configuration settings for consistency.
    """
    if config["active_learning"]["strategy"] in ["ei", "pi", "ucb", "thompson"]:
        assert config.get("use_dropout", False) or config.get("ensemble", False), \
            "Ensemble-based selection strategies require dropout sampling or cv models."


def get_predictions(train_df, X_all, X_train, config):
    dms_column=DMS_COL if config["noise_level"] == 0 else DMS_COL + "_noise"
    y_train = train_df[dms_column].tolist()

    kernel = RBF() + ConstantKernel() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
    gpr.fit(X_train, y_train)
    return gpr.predict(X_all, return_std=True)


def main(config):
    """
    Main function for the active learning experiment.
    Separates out training logic and passes model predictions
    (rather than the model) to the selection strategy.
    """
    seed_everything(config["random_seed"])

    # Log file
    os.makedirs(config["tmp_path"], exist_ok=True)
    logfile_path = os.path.join(config["tmp_path"], 'active_learning_log.txt')
    logfile = open(logfile_path, 'a')
    if os.path.exists(logfile_path):
        logfile.write('\n\n')
        write_log(logfile, f"Resuming active learning experiment with config: {config}")

    # Load mapping and select DMS data
    train_mapping_path = config["train_dms_mapping"]
    train_df_mapping = pd.read_csv(train_mapping_path)
    name = train_df_mapping.loc[config["dms_index"], 'DMS_id']

    # Active learning parameters
    check_config(config)
    N_init = config["active_learning"]["N_init"]
    N_per_cycle = config["active_learning"]["N_per_cycle"]
    M_cycles = config["active_learning"]["M_cycles"]
    strategy_params = config["active_learning"].get("strategy_params", {})
    strategy_name = config["active_learning"]["strategy"]
    if strategy_name in ["random", "thompson"]:
        strategy_params["seed"] = config["random_seed"]

    initial_selection_mode = config.get("initial_selection_mode", "random")

    # Load data and initialize cycle
    initial_cycle = 0
    for cycle in range(M_cycles):
        cycle_df_path = os.path.join(config["tmp_path"], f"predictions_cycle_{cycle+1}.csv")
        if os.path.exists(cycle_df_path):
            df_all = pd.read_csv(cycle_df_path)
            initial_cycle = cycle + 1
    write_log(logfile, f"Starting from cycle {initial_cycle}")
    ablang = ablang2.pretrained(model_to_use='ablang2-paired', random_init=False, device='cuda')


    if initial_cycle == 0:
        write_log(logfile, "Starting new active learning experiment.")
        dms_data_path = os.path.join(config["dms_input"], f"{name}.csv")
        df_all = pd.read_csv(dms_data_path)
        df_all["selected_cycle"] = -1
        df_all["is_test"] = False

        all_seqs = list(zip(df_all["heavy"].tolist(), df_all["light"].tolist()))
        D=1000
        Xs = []
        for i in tqdm(range(0, len(all_seqs), D)):
            seqs = all_seqs[i:i+D]
            X = ablang(seqs, mode='seqcoding')
            Xs.append(X)
        X_all = np.concatenate(Xs, axis=0)
        np.save(os.path.join(config["tmp_path"], "X_all.npy"), X_all)


    logfile.close()


if __name__ == '__main__':
    """
    Example usage:
    python active_learning_script.py --config config.yaml
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    args_cmd = parser.parse_args()
    config_dict = parse_config(args_cmd.config)
    config_dict["batch_size"] = args_cmd.batch_size
    main(config_dict)
