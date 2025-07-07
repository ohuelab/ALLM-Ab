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
import torch
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.nn.data_parallel import DataParallel
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
import scipy

import DEMEmodel
from loss import listMLE
from dataset import StructureDataset, SequenceDataset, AbLang2Dataset
from protein_mpnn_utils import ProteinMPNN
from utils import DMS_file_for_LLM

from esm import ESM2, Alphabet, pretrained
from ablang2.load_model import load_model

from strategy import get_selector


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
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


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


def make_dataset(df_or_list, index_list, evaluation, args, noise=False):
    """
    Creates a dataset object for structure or sequence models.
    Uses a noise column if required (e.g. for validation).
    """
    dms_eval = evaluation and not noise
    if args["model_type"] == 'structure':
        return StructureDataset(
            df_or_list,
            index_list,
            structure_path=args["structure_path"],
            batch_size=args["batch_size"],
            esm_alphabet=None,
            evaluation=evaluation,
            dms_column=DMS_COL if dms_eval or args["noise_level"] == 0 else DMS_COL + "_noise"
        )
    elif args["model_type"] == 'sequence':
        if args["use_weight"] == 'pretrained':
            _, esm_alphabet_ = pretrained.load_model_and_alphabet('esm2_t33_650M_UR50D')
        else:
            esm_alphabet_ = Alphabet.from_architecture("ESM-1b")
        return SequenceDataset(
            df_or_list,
            index_list,
            structure_path=args["structure_path"],
            batch_size=args["batch_size"],
            esm_alphabet=esm_alphabet_,
            evaluation=evaluation,
            dms_column=DMS_COL if dms_eval or args["noise_level"] == 0 else DMS_COL + "_noise"
        )
    elif args["model_type"] == 'ablang2':
        _, tokenizer, _ = load_model("ablang2-paired")
        return AbLang2Dataset(
            df_or_list,
            index_list,
            batch_size=args["batch_size"],
            tokenizer=tokenizer,
            evaluation=evaluation
        )
    else:
        raise ValueError(f"Invalid model type: {args['model_type']}")

def forward_with_loss_func(model, batch_data, parallel_running):
    """
    Performs a forward pass and computes the listMLE loss.
    """
    outputs = model(batch_data)
    if parallel_running:
        y = torch.cat([d.reg_labels for d in batch_data], dim=0).to(outputs.device)
    else:
        y = batch_data.reg_labels
    loss_val = listMLE(-outputs, -y)
    return outputs, y, loss_val


def create_model_func(args):
    """
    Creates and returns a new model based on configuration arguments.
    """
    parallel = (torch.cuda.device_count() > 1)
    hidden_dim = 128
    num_layers = 3

    if args["model_type"] == 'structure':
        model_local = ProteinMPNN(
            ca_only=False,
            num_letters=21,
            node_features=hidden_dim,
            edge_features=hidden_dim,
            hidden_dim=hidden_dim,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            augment_eps=0.2,
            k_neighbors=48
        )
        if args["use_weight"] == 'pretrained':
            state_dict = torch.load("./cache/v_48_020.pt", torch.device('cpu'))
            model_local.load_state_dict(state_dict['model_state_dict'])
        if args["lora"]:
            lora_config = {
                "target_modules": [
                    "W1", "W2", "W3", "W11", "W12", "W13",
                    "W_in", "W_out"
                ],
                "modules_to_save": [],
                "inference_mode": False,
                "lora_dropout": 0.1,
                "lora_alpha": 8
            }
            peft_config = LoraConfig(**lora_config)
            model_local = get_peft_model(model_local, peft_config)
    elif args["model_type"] == 'sequence':
        if args["use_weight"] == 'pretrained':
            esm_pretrain_model, _ = pretrained.load_model_and_alphabet('esm2_t33_650M_UR50D')
        else:
            esm_pretrain_model = ESM2()
        if args["lora"]:
            lora_config = {
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "out_proj",
                    "lm_head.dense"
                ],
                "modules_to_save": [],
                "inference_mode": False,
                "lora_dropout": 0.1,
                "lora_alpha": 8
            }
            peft_config = LoraConfig(**lora_config)
            esm_pretrain_model = get_peft_model(esm_pretrain_model, peft_config)
        model_local = DEMEmodel.DEME(esm_pretrain_model, None)
    elif args["model_type"] == 'ablang2':
        AbLang, tokenizer, _ = load_model("ablang2-paired")
        if args["lora"]:
            config = LoraConfig(
                    lora_alpha=8,
                    lora_dropout=0.1,
                    target_modules=[
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "out_proj",
                        "intermediate_layer.0",
                        "intermediate_layer.2",
                    ],
            )
            AbLang = get_peft_model(AbLang, config)
        model_local = DEMEmodel.DEMEAbLang2(AbLang)
    else:
        raise ValueError(f"Invalid model type: {args['model_type']}")

    if parallel:
        model_local = DataParallel(model_local)
    model_local.cuda()
    return model_local


def training_ensemble(
    train_data,
    model_init_func,
    mc_splits=5,
    max_steps=100,
    eval_interval=5,
    seed=42,
    ratio_small=0.5,
    ratio_large=0.75,
    early_stopping=False,
    early_stop_patience=10,
):
    """
    Conducts Monte Carlo cross-validation to find the best training step.
    """
    if len(train_data) < 50:
        val_ratio = ratio_small
    else:
        val_ratio = ratio_large

    rng = np.random.RandomState(seed)

    ensemble_models = []
    for mc_i in range(mc_splits):
        # Train/Val split
        train_idx, val_idx = train_test_split(
            train_data.index,
            test_size=val_ratio,
            random_state=rng.randint(0, 1_000_000)
        )
        df_train_split = train_data.loc[train_idx].reset_index(drop=True)
        df_val_split = train_data.loc[val_idx].reset_index(drop=True)

        # Build datasets
        train_dataset = model_init_func["make_dataset_func"](
            df_train_split, df_train_split.index, evaluation=False, args=model_init_func["args"]
        )
        val_dataset = model_init_func["make_dataset_func"](
            df_val_split, df_val_split.index, evaluation=True, noise=True, args=model_init_func["args"]
        )

        # Create model
        model = model_init_func["create_model_func"]()

        # Train model and evaluate
        model, _, _, _ = training(
            model,
            train_dataset,
            val_dataset=val_dataset,
            test_dataset=None,
            n_steps=max_steps,
            eval_interval=eval_interval,
            config=model_init_func["args"],
            early_stopping=early_stopping,
            early_stop_patience=early_stop_patience
        )
        model.eval()
        model = model.cpu()
        ensemble_models.append(model)

        del model
        gc.collect()
        torch.cuda.empty_cache()
    return ensemble_models


def monte_carlo_early_stopping(
    train_data,
    model_init_func,
    mc_splits=5,
    max_steps=100,
    eval_interval=5,
    seed=42,
    ratio_small=0.5,
    ratio_large=0.75,
):
    """
    Conducts Monte Carlo cross-validation to find the best training step.
    """
    if len(train_data) < 50:
        val_ratio = ratio_small
    else:
        val_ratio = ratio_large

    spearman_records = defaultdict(list)
    rng = np.random.RandomState(seed)

    for mc_i in range(mc_splits):
        # Train/Val split
        train_idx, val_idx = train_test_split(
            train_data.index,
            test_size=val_ratio,
            random_state=rng.randint(0, 1_000_000)
        )
        df_train_split = train_data.loc[train_idx].reset_index(drop=True)
        df_val_split = train_data.loc[val_idx].reset_index(drop=True)

        # Build datasets
        train_dataset = model_init_func["make_dataset_func"](
            df_train_split, df_train_split.index, evaluation=False, args=model_init_func["args"]
        )
        val_dataset = model_init_func["make_dataset_func"](
            df_val_split, df_val_split.index, evaluation=True, noise=True, args=model_init_func["args"]
        )

        # Create model
        model = model_init_func["create_model_func"]()

        # Train model and evaluate
        model, val_scores, _, _ = training(
            model,
            train_dataset,
            val_dataset=val_dataset,
            test_dataset=None,
            n_steps=max_steps,
            eval_interval=eval_interval,
            config=model_init_func["args"]
        )

        # Record validation Spearman
        for step_val, sp_val in val_scores:
            spearman_records[step_val].append(sp_val)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Determine the best step
    best_step = None
    best_mean_spearman = -9999
    for step_i, sp_list in spearman_records.items():
        mean_sp = np.mean(sp_list)
        if mean_sp > best_mean_spearman:
            best_mean_spearman = mean_sp
            best_step = step_i

    # Return optional ensemble predictions
    return best_step


def get_predictions(model, df_data, config):
    """
    Generates prediction values for a given DataFrame.
    Allows toggling of noise for validation-like usage.
    """
    dataset = make_dataset(
        df_data,
        df_data.index,
        evaluation=True,
        args=config)
    parallel_running = isinstance(model, DataParallel)
    loader_class = DataListLoader if parallel_running else DataLoader
    data_loader = loader_class(
        dataset=dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=2
    )

    model.eval()
    all_preds = []
    for batch in tqdm(data_loader):
        if not parallel_running:
            batch = batch.cuda()
        with torch.no_grad():
            pout, _, _ = forward_with_loss_func(model, batch, parallel_running)
            all_preds.append(pout.detach().cpu().numpy())
    return np.concatenate(all_preds).reshape(-1)

def get_predictions_dropout(model, df_data, config, n_samples=5):
    """
    Generates predictions and uncertainty estimates using Monte Carlo Dropout.
    """
    dataset = make_dataset(
        df_data,
        df_data.index,
        evaluation=True,
        args=config)
    parallel_running = isinstance(model, DataParallel)
    loader_class = DataListLoader if parallel_running else DataLoader
    data_loader = loader_class(
        dataset=dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=2
    )

    # モデルを訓練モードにしてDropoutを有効化
    model.train()
    all_predictions = []

    # n_samples回予測を実行
    for _ in range(n_samples):
        batch_predictions = []
        for batch in tqdm(data_loader):
            if not parallel_running:
                batch = batch.cuda()
            with torch.no_grad():
                pout, _, _ = forward_with_loss_func(model, batch, parallel_running)
                batch_predictions.append(pout.detach().cpu().numpy())
        # 1回のサンプリング結果を保存
        all_predictions.append(np.concatenate(batch_predictions).reshape(-1))

    # 予測値の統計量を計算
    all_predictions = np.array(all_predictions)

    return all_predictions

def check_config(config):
    """
    Checks the configuration settings for consistency.
    """
    if config["active_learning"]["strategy"] in ["ei", "pi", "ucb", "thompson"]:
        assert config.get("use_dropout", False) or config.get("ensemble", False), \
            "Ensemble-based selection strategies require dropout sampling or cv models."


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

    if initial_cycle == 0:
        write_log(logfile, "Starting new active learning experiment.")
        dms_data_path = os.path.join(config["dms_input"], f"{name}.csv")
        df_all = pd.read_csv(dms_data_path)
        df_all = DMS_file_for_LLM(df_all, focus=False if config["model_type"] == 'structure' else True)
        df_all["selected_cycle"] = -1
        df_all["is_test"] = False

        # Split test data
        test_df = df_all.sample(n=config["test_size"], random_state=config["random_seed"])
        test_indices = test_df.index.tolist()
        pool_df = df_all.drop(test_df.index)
        pool_indices = pool_df.index.tolist()
        all_indices = {"pool": pool_indices, "test": test_indices}
        df_all.loc[test_indices, "is_test"] = True
        with open(os.path.join(config["tmp_path"], "selected_indices.json"), "w") as f:
            json.dump(all_indices, f)

        # Optionally add noise column
        if config["noise_level"] > 0 and DMS_COL+"_noise" not in pool_df.columns:
            write_log(logfile, f"  Adding noise to DMS scores with std = {config['noise_level']}")
            rng = np.random.RandomState(config["random_seed"])
            dstd = pool_df[DMS_COL].std() * config["noise_level"]
            pool_df[DMS_COL + "_noise"] = pool_df[DMS_COL] + rng.normal(0, dstd, size=len(pool_df))

        # 初期サンプリングの分岐  # CHANGED
        if initial_selection_mode == "random":
            write_log(logfile, f"  Initializing with {N_init} random samples.")
            initial_selector = get_selector("random", {"seed": config["random_seed"]})
            init_indices = initial_selector.select_samples(pool_df, None, N_init)
        else:
            write_log(logfile, f"  Initializing with {N_init} samples from baseline prediction.")
            model_init_dict = {
                "parallel": (torch.cuda.device_count() > 1),
                "args": config,
                "create_model_func": lambda: create_model_func(config),
                "make_dataset_func": make_dataset
            }

            model = create_model_func(config)
            model.cuda()
            model.eval()
            if config.get("use_dropout", False):
                write_log(logfile, f"  Use dropout.")
                all_preds_list = get_predictions_dropout(model, df_all, config, config["mc_splits"])
                pd.DataFrame(all_preds_list).to_csv(os.path.join(config["tmp_path"], "predictions_cycle_0_dropout.csv"), index=False)
                df_all[f"cycle_0_preds"] = all_preds_list.mean(axis=0)
                df_all[f"cycle_0_uncertainties"] = all_preds_list.std(axis=0)
            else:
                if config.get("model_type", "structure") != "sequence":
                    all_preds = get_predictions(model, df_all, config)
                else:
                    df_all_esm = pd.read_csv(os.path.join(config["dms_input_esm"], f"{name}.csv"))
                    assert len(df_all_esm) == len(df_all), "Mismatch in the number of samples between df_all and df_all_esm"
                    all_preds = df_all_esm["esm2_t33_650M_UR50D"].values
                df_all[f"cycle_0_preds"] = all_preds
            model.cpu()
            del model
            gc.collect()
            torch.cuda.empty_cache()

            write_log(logfile, f"  Initializing with {N_init} samples from baseline prediction.")
            if config.get("ensemble", False):
                # 初期はモデルが1つなので、greedy selection のみ
                baseline_selector = get_selector("greedy", strategy_params)
            else:
                baseline_selector = get_selector(strategy_name, strategy_params)
                # baseline予測値をプール部分に対応付け
            if config.get("use_dropout", False):
                pool_preds_list = all_preds_list[:, pool_df.index]
                init_indices = baseline_selector.select_samples(pool_df, pool_preds_list, N_init)
            else:
                pool_preds = all_preds[pool_df.index]
                init_indices = baseline_selector.select_samples(pool_df, pool_preds, N_init)
        df_all.to_csv(os.path.join(config["tmp_path"], "predictions_cycle_0.csv"), index=False)

        train_selected_df = pool_df.loc[init_indices].copy()
        df_all.loc[init_indices, "selected_cycle"] = 0
        pool_df.drop(init_indices, inplace=True)
        all_indices[str(0)] = init_indices
        with open(os.path.join(config["tmp_path"], "selected_indices.json"), "w") as f:
            json.dump(all_indices, f)

    else:
        with open(os.path.join(config["tmp_path"], "selected_indices.json"), "r") as f:
            all_indices = json.load(f)
        initial_pool_indices = all_indices["pool"]
        test_indices = all_indices["test"]
        test_df = df_all.loc[test_indices]
        train_indices = []
        for cycle in range(initial_cycle+1):
            if str(cycle) in all_indices:
                train_indices.extend(all_indices[str(cycle)])
        pool_df = df_all.loc[initial_pool_indices].copy()
        if config["noise_level"] > 0 and DMS_COL+"_noise" not in pool_df.columns:
            write_log(logfile, f"  Adding noise to DMS scores with std = {config['noise_level']}")
            rng = np.random.RandomState(config["random_seed"])
            dstd = pool_df[DMS_COL].std() * config["noise_level"]
            pool_df[DMS_COL + "_noise"] = pool_df[DMS_COL] + rng.normal(0, dstd, size=len(pool_df))

        train_selected_df = pool_df.loc[train_indices].copy()
        pool_df.drop(train_indices, inplace=True)
        assert (df_all["selected_cycle"]>=0).sum() == len(train_indices), \
            "Mismatch in selected cycle indices: {} vs {}".format((df_all["selected_cycle"]>=0).sum(), len(train_indices))

    for cycle in range(initial_cycle, M_cycles):
        # Re-initialize selector
        selector = get_selector(strategy_name, strategy_params)
        best_score = train_selected_df[DMS_COL+"_noise"].max() if config["noise_level"] > 0 else train_selected_df[DMS_COL].max()
        if strategy_name in ["ei", "pi"]:
            selector.best_score = best_score

        write_log(logfile, f"\n=== Active Learning Cycle {cycle+1} / {M_cycles} ===")

        # Prepare MC early stopping inputs
        model_init_dict = {
            "parallel": (torch.cuda.device_count() > 1),
            "args": config,
            "create_model_func": lambda: create_model_func(config),
            "make_dataset_func": make_dataset
        }
        write_log(logfile, f"  [MC-EarlyStopping] training data size = {len(train_selected_df)}")

        # Build test dataset if any test data
        test_dataset = None
        if not config.get("ensemble", False):
            best_step = 500
            if len(train_selected_df) < 100:
                best_step = monte_carlo_early_stopping(
                    train_data=train_selected_df,
                    model_init_func=model_init_dict,
                    mc_splits=config["mc_splits"],
                    max_steps=config["mc_train_steps"],
                    eval_interval=config["mc_eval_interval"],
                    seed=config["random_seed"],
                    ratio_small=0.5,
                    ratio_large=0.75
                )
                write_log(logfile, f"  [MC-EarlyStopping] best step: {best_step}")

            if best_step >= 100:
                train_selected_df_train, train_selected_df_val = train_test_split(
                    train_selected_df,
                    test_size=0.2 if len(train_selected_df) > 100 else 20,
                    random_state=config["random_seed"]
                )
                final_train_dataset = make_dataset(
                    train_selected_df_train, train_selected_df_train.index, evaluation=False, args=config)
                val_dataset = make_dataset(
                    train_selected_df_val, train_selected_df_val.index, evaluation=True, noise=True, args=config)
            else:
                final_train_dataset = make_dataset(
                    train_selected_df, train_selected_df.index, evaluation=False, args=config)
                val_dataset = None

            # Final training
            final_model = create_model_func(config)

            final_model, val_spearman_scores, test_spearman, best_epoch = training(
                final_model,
                final_train_dataset,
                val_dataset=val_dataset,
                test_dataset=None,
                n_steps=best_step,
                eval_interval=1,
                config=config,
                early_stopping=True if val_dataset is not None else False,
                early_stop_patience=config["early_stop_patience"]
            )
            if val_spearman_scores is not None:
                write_log(logfile, "  Validation Spearman: {} {}".format(len(val_spearman_scores), best_epoch))
                write_log(logfile, f"  Validation Spearman: {val_spearman_scores[(best_epoch-1)][1]:.4f}")
            if test_spearman is not None:
                write_log(logfile, f"  Test Spearman: {test_spearman:.4f}")

            # Save final model
            if config.get("save_model", False):
                save_path = os.path.join(config["tmp_path"], f"model_cycle_{cycle+1}.pt")
                if isinstance(final_model, DataParallel):
                    torch.save(final_model.module.state_dict(), save_path)
                else:
                    torch.save(final_model.state_dict(), save_path)
                write_log(logfile, f"  [Model Saved]: {save_path}")

            # Predictions on entire dataset
            if config.get("use_dropout", False):
                all_preds_list = get_predictions_dropout(final_model, df_all, config,config["mc_splits"])
                df_all[f"cycle_{cycle+1}_preds"] = all_preds_list.mean(axis=0)
                df_all[f"cycle_{cycle+1}_uncertainties"] = all_preds_list.std(axis=0)
            else:
                all_preds = get_predictions(final_model, df_all, config)
                df_all[f"cycle_{cycle+1}_preds"] = all_preds

            # Active selection from pool
            if len(pool_df) > 0:
                if config.get("use_dropout", False):
                    pool_preds_list = all_preds_list[:, pool_df.index]
                    add_indices = selector.select_samples(pool_df, pool_preds_list, N_per_cycle)
                else:
                    pool_preds = all_preds[pool_df.index]
                    add_indices = selector.select_samples(pool_df, pool_preds, N_per_cycle)
                new_data = pool_df.loc[add_indices]
                train_selected_df = pd.concat([train_selected_df, new_data]).reset_index(drop=True)
                df_all.loc[add_indices, "selected_cycle"] = cycle + 1
                pool_df.drop(add_indices, inplace=True)
            else:
                write_log(logfile, "  No more samples to select from pool.")

            del final_model
            gc.collect()
            torch.cuda.empty_cache()
        else:
            # ENSEMBLE training
            ensemble_models = training_ensemble(
                train_data=train_selected_df,
                model_init_func=model_init_dict,
                mc_splits=config["mc_splits"],
                max_steps=config["mc_train_steps"],
                eval_interval=1,
                seed=config["random_seed"],
                ratio_small=0.5,
                ratio_large=0.75,
                early_stopping=True,
                early_stop_patience=config["early_stop_patience"]
            )
            for e_i, model_e in enumerate(ensemble_models):
                if config.get("save_model", False):
                    save_path_e = os.path.join(config["tmp_path"], f"model_cycle_{cycle+1}_ensemble_{e_i}.pt")
                    if isinstance(model_e, DataParallel):
                        torch.save(model_e.module.state_dict(), save_path_e)
                    else:
                        torch.save(model_e.state_dict(), save_path_e)
                    write_log(logfile, f"  [Model {e_i} Saved]: {save_path_e}")
            all_preds_list = []
            for model_e in ensemble_models:
                model_e = model_e.cuda()
                p = get_predictions(model_e, df_all, config)
                model_e.cpu()
                all_preds_list.append(p)
                gc.collect()
                torch.cuda.empty_cache()

            all_preds_list = np.array(all_preds_list)
            all_preds = all_preds_list.mean(axis=0)
            all_uncertainties = all_preds_list.std(axis=0)
            df_all[f"cycle_{cycle+1}_preds"] = all_preds
            df_all[f"cycle_{cycle+1}_uncertainties"] = all_uncertainties
            pool_preds_list = all_preds_list[:, pool_df.index]
            # For selection, gather predictions from each ensemble model on the pool
            if len(pool_df) > 0:
                add_indices = selector.select_samples(pool_df, pool_preds_list, N_per_cycle)
                new_data = pool_df.loc[add_indices]
                train_selected_df = pd.concat([train_selected_df, new_data]).reset_index(drop=True)
                df_all.loc[add_indices, "selected_cycle"] = cycle + 1
                pool_df.drop(add_indices, inplace=True)
            else:
                add_indices = []
                write_log(logfile, "  No more samples to select from pool.")

            # Clean up
            for m_e in ensemble_models:
                del m_e
            gc.collect()
            torch.cuda.empty_cache()

        # Save CSV with selection info
        pred_csv_path = os.path.join(config["tmp_path"], f"predictions_cycle_{cycle+1}.csv")
        df_all.to_csv(pred_csv_path, index=False)

        # Save indices
        all_indices[str(cycle + 1)] = add_indices
        with open(os.path.join(config["tmp_path"], "selected_indices.json"), "w") as f:
            json.dump(all_indices, f)
        if len(pool_df) == 0:
            break

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
