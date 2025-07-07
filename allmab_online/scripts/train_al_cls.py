import os
import gc
import random
import copy
import hashlib
import json

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.nn.data_parallel import DataParallel

from sklearn.model_selection import KFold, train_test_split
import scipy
from tqdm import tqdm

from argparse import ArgumentParser

from peft import get_peft_model, LoraConfig
from loss import listMLE
import DEMEmodel
from dataset import SequenceDataset, AbLang2Dataset
from utils import DMS_file_for_LLM
from esm import ESM2, Alphabet, pretrained
from ablang2.load_model import load_model


def setup_args():
    """
    Parse command-line arguments for the few-shot experiment with
    Monte Carlo cross-validation–based early stopping.
    """
    parser = ArgumentParser()
    parser.add_argument('--dms_input', type=str, default='')
    parser.add_argument('--model_type', type=str, default='sequence')
    parser.add_argument('--lora', action='store_true', default=True, help='Enable LoRA if using sequence model.')
    parser.add_argument('--use_weight', type=str, default='pretrained')
    parser.add_argument('--output_dir', type=str, default='tmp')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--early_stop_patience', type=int, default=20)
    parser.add_argument('--n_steps', type=int, default=1000)
    parser.add_argument('--cluster', type=int, default=0)
    parser.add_argument('--test_input', type=str, default=None)
    return parser.parse_args()


def seed_everything(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def generate_unique_id(string):
    """Generate a unique SHA-1 hash from a given string."""
    hash_object = hashlib.sha1()
    hash_object.update(string.encode('utf-8'))
    return hash_object.hexdigest()


def write_log(log_file, text, is_print=True):
    """Write text to a log file and optionally print it."""
    if is_print:
        print(text)
    log_file.write(text + '\n')


def metric(preds, labels, bottom_preds=None, bottom_labels=None,
          top_preds=None, top_labels=None, eps=1e-6):
    """
    Calculate performance metrics (Pearson, Spearman, RMSE).
    If 'top_test' split is used, it also calculates precision/recall/F1
    for top/bottom subsets of data.
    """
    metrics = {
        'pearson': scipy.stats.pearsonr(preds, labels)[0],
        'spearman': scipy.stats.spearmanr(preds, labels)[0],
        'rmse': np.mean((preds - labels) ** 2) ** 0.5
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
            # Top side metrics
            precision = (top_pred_idxs[-min(k, n_top):] >= (len(all_preds) - n_top)).mean()
            recall = (top_pred_idxs[-min(k, n_top):] >= (len(all_preds) - n_top)).sum() / n_top
            f1 = 2 * precision * recall / (precision + recall + eps)
            top_metrics.append([precision, recall, f1])

            # Bottom side metrics
            precision = (top_pred_idxs[:min(k, n_bottom)] < n_bottom).mean()
            recall = (top_pred_idxs[:min(k, n_bottom)] < n_bottom).sum() / n_top
            f1 = 2 * precision * recall / (precision + recall + eps)
            bottom_metrics.append([precision, recall, f1])

        # Additional set for "top_frac"
        precision = (top_pred_idxs >= (len(all_preds) - n_top)).mean()
        recall = (top_pred_idxs >= (len(all_preds) - n_top)).sum() / n_top
        f1 = 2 * precision * recall / (precision + recall + eps)
        top_metrics.append([precision, recall, f1])

        precision = (top_pred_idxs < n_bottom).mean()
        recall = (top_pred_idxs < n_bottom).sum() / n_top
        f1 = 2 * precision * recall / (precision + recall + eps)
        bottom_metrics.append([precision, recall, f1])

        for i, _ in enumerate([10, 20, 50, 100, 'top_frac']):
            unbias_metrics.append(list(np.array(top_metrics[i]) - np.array(bottom_metrics[i])))

        metrics.update({
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

    return metrics


def make_dataset(df_or_list, index_list, evaluation, args):
    """Create a dataset according to model_type (structure or sequence)."""
    if args.model_type == 'sequence':


        if args.use_weight == 'pretrained':
            _, esm_alphabet_ = pretrained.load_model_and_alphabet('esm2_t33_650M_UR50D')
        else:
            esm_alphabet_ = Alphabet.from_architecture("ESM-1b")

        return SequenceDataset(
            df_or_list,
            index_list,
            batch_size=args.batch_size,
            esm_alphabet=esm_alphabet_,
            evaluation=evaluation
        )
    elif args.model_type == 'ablang2':
        _, tokenizer, _ = load_model("ablang2-paired")
        return AbLang2Dataset(
            df_or_list,
            index_list,
            batch_size=args.batch_size,
            tokenizer=tokenizer,
            evaluation=evaluation
        )
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")


def forward_with_loss_func(model, batch_data, parallel_running):
    """Performs a forward pass and computes the listMLE loss."""
    outputs = model(batch_data)
    y = torch.cat([d.reg_labels for d in batch_data], dim=0).to(outputs.device) if parallel_running else batch_data.reg_labels
    loss_val = listMLE(-outputs, -y)
    return outputs, y, loss_val


def get_predictions(model, df_data, args):
    """Generates prediction values for a given DataFrame."""
    dataset = make_dataset(df_data, df_data.index, evaluation=True, args=args)
    parallel_running = isinstance(model, DataParallel)
    loader_class = DataListLoader if parallel_running else DataLoader
    data_loader = loader_class(
        dataset=dataset,
        batch_size=args.batch_size,
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


def training(model, train_dataset, val_dataset=None, test_dataset=None,
            n_steps=100, eval_interval=5, args=None, early_stopping=False,
            early_stop_patience=10):
    """Train the model with optional validation and early stopping."""
    early_stop_patience_ = early_stop_patience // eval_interval
    parallel_running = isinstance(model, DataParallel)
    loader_class = DataListLoader if parallel_running else DataLoader

    train_loader = loader_class(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=2
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    early_stop_counter = 0
    best_spearman = float('-inf')
    val_spearman_scores = [] if val_dataset else None
    best_epoch = -1 if val_dataset else None
    best_model = None

    for step in range(1, n_steps + 1):
        print("epoch", step)
        model.train()
        for batch_data in tqdm(train_loader):
            optimizer.zero_grad()
            if not parallel_running:
                batch_data = batch_data.cuda()
            outputs, y, loss_val = forward_with_loss_func(model, batch_data, parallel_running)
            loss_val.backward()
            optimizer.step()

        if val_dataset and (step % eval_interval == 0 or step == n_steps):
            val_metrics = evaluate_validation(model, val_dataset, parallel_running, loader_class, args)
            val_spearman = val_metrics['spearman']
            val_spearman_scores.append((step, val_spearman))

            if early_stopping:
                if val_spearman > best_spearman:
                    best_spearman = val_spearman
                    early_stop_counter = 0
                    best_model = copy.deepcopy(model).cpu()
                    best_epoch = step
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= early_stop_patience_:
                        print("Early stopping triggered!")
                        break

        gc.collect()
        torch.cuda.empty_cache()

    test_spearman = None
    if test_dataset:
        test_metrics = evaluate_test(model, test_dataset, parallel_running, loader_class, args)
        test_spearman = test_metrics['spearman']

    if early_stopping and best_model is not None:
        model = best_model.cuda()

    return model, val_spearman_scores, test_spearman, best_epoch


def evaluate_validation(model, val_dataset, parallel_running, loader_class, args):
    """Evaluate model on validation dataset."""
    val_loader = loader_class(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2
    )

    model.eval()
    val_preds, val_labels = [], []

    for vbatch in val_loader:
        if not parallel_running:
            vbatch = vbatch.cuda()
        with torch.no_grad():
            out_v, y_v, _ = forward_with_loss_func(model, vbatch, parallel_running)
            val_preds.append(out_v.detach().cpu().numpy())
            val_labels.append(y_v.detach().cpu().numpy())

    val_preds = np.concatenate(val_preds).reshape(-1)
    val_labels = np.concatenate(val_labels).reshape(-1)

    return {'spearman': scipy.stats.spearmanr(val_preds, val_labels)[0]}


def evaluate_test(model, test_dataset, parallel_running, loader_class, args):
    """Evaluate model on test dataset."""
    test_loader = loader_class(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2
    )

    model.eval()
    test_preds, test_labels = [], []

    for tbatch in test_loader:
        if not parallel_running:
            tbatch = tbatch.cuda()
        with torch.no_grad():
            pout, py, _ = forward_with_loss_func(model, tbatch, parallel_running)
            test_preds.append(pout.detach().cpu().numpy())
            test_labels.append(py.detach().cpu().numpy())

    test_preds = np.concatenate(test_preds).reshape(-1)
    test_labels = np.concatenate(test_labels).reshape(-1)

    return metric(test_preds, test_labels)


def create_model_func(args):
    """Create and return a new model based on configuration arguments."""
    parallel = (torch.cuda.device_count() > 1)

    if args.model_type == 'sequence':
        esm_pretrain_model = get_esm_model(args)
        model_local = DEMEmodel.DEME(esm_pretrain_model)
    elif args.model_type == 'ablang2':
        ablang2_pretrain_model = get_ablang2_model(args)
        model_local = DEMEmodel.DEMEAbLang2(ablang2_pretrain_model)
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")

    if parallel:
        model_local = DataParallel(model_local)
    model_local.cuda()
    return model_local


def get_esm_model(args):
    """Get ESM model based on configuration."""
    if args.use_weight == 'pretrained':
        esm_pretrain_model, _ = pretrained.load_model_and_alphabet('esm2_t33_650M_UR50D')
    else:
        esm_pretrain_model = ESM2()

    if args.lora:
        lora_config = {
            "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj", "lm_head.dense"],
            "modules_to_save": [],
            "inference_mode": False,
            "lora_dropout": 0.1,
            "lora_alpha": 8
        }
        peft_config = LoraConfig(**lora_config)
        esm_pretrain_model = get_peft_model(esm_pretrain_model, peft_config)

    return esm_pretrain_model

def get_ablang2_model(args):
    """Get ablang2 model based on configuration."""
    AbLang, tokenizer, _ = load_model("ablang2-paired")
    if args.lora:
        lora_config = {
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "out_proj",
                "intermediate_layer.0",
                "intermediate_layer.2",
            ],
            "modules_to_save": [],
            "inference_mode": False,
            "lora_dropout": 0.1,
            "lora_alpha": 8
        }
        peft_config = LoraConfig(**lora_config)
        AbLang = get_peft_model(AbLang, peft_config)
    return AbLang

def main(args):
    """Main function for training and evaluation."""
    save_args(args)
    seed_everything(args.seed)

    df_all = pd.read_csv(args.dms_input)
    df_all = DMS_file_for_LLM(df_all, focus=True)

    df_test = None
    if args.test_input is None:
        args.test_input = args.dms_input
    if args.test_input:
        df_test = pd.read_csv(args.test_input)
        df_test = DMS_file_for_LLM(df_test, focus=True)
    df_all = df_all[df_all["cluster"]!=args.cluster].copy().reset_index()
    train_index, vali_index = train_test_split(df_all.index, test_size=0.2, random_state=args.seed)
    df_train = df_all.iloc[train_index].reset_index(drop=True)
    df_val = df_all.iloc[vali_index].reset_index(drop=True)

    model_final = create_model_func(args)

    print("Train Dataset")
    final_train_dataset = make_dataset(df_train, df_train.index, evaluation=False, args=args)
    print("Val Dataset")
    val_dataset = make_dataset(df_val, df_val.index, evaluation=True, args=args)

    print("Training")
    model_final, val_spearman_scores, test_spearman, best_epoch = training(
        model_final,
        final_train_dataset,
        val_dataset,
        test_dataset=None,
        n_steps=args.n_steps,
        eval_interval=1,
        args=args,
        early_stopping=True,
        early_stop_patience=args.early_stop_patience,
    )

    print_results(val_spearman_scores, test_spearman, best_epoch)
    save_results(model_final, df_test, args)

    cleanup(model_final)


def save_args(args):
    """Save arguments to file."""
    with open(f'{args.output_dir}/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def print_results(val_spearman_scores, test_spearman, best_epoch):
    """Print training results."""
    if val_spearman_scores:
        print(f"  Validation Spearman: {len(val_spearman_scores)} {best_epoch}")
        print(f"  Validation Spearman: {val_spearman_scores}")
    if test_spearman:
        print(f"  Test Spearman: {test_spearman:.4f}")


def save_results(model, df_test, args):
    """Save model and predictions."""
    if args.model_type == 'sequence':
        torch.save(model.esm_pretrain_model.state_dict(), f'{args.output_dir}/model.pt')
    elif args.model_type == 'ablang2':
        torch.save(model.ablang2_pretrain_model.state_dict(), f'{args.output_dir}/model.pt')
    if df_test is not None:
        test_preds = get_predictions(model, df_test, args)
        np.save(f'{args.output_dir}/test_preds.npy', test_preds)


def cleanup(model):
    """Clean up resources."""
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    args = setup_args()
    main(args)
