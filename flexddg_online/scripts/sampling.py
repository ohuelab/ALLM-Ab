import torch
import torch.nn.functional as F
from esm import ESM2, Alphabet, pretrained
from ablang2.load_model import load_model
from ablang2.pretrained import format_seq_input
from peft import get_peft_model, LoraConfig
import numpy as np
import argparse
import gc
import os
import pandas as pd
from utils import load_config
from logits import get_prediction_score

ABLANG2_TOKEN_OFFSET=1
ABLANG2_MASK_TOKEN = 23
ESM_TOKEN_OFFSET = 4
TOK_NUM = 20

ablang_standard_aa_tokens = list(range(ABLANG2_TOKEN_OFFSET, ABLANG2_TOKEN_OFFSET+TOK_NUM))
esm_standard_aa_tokens = list(range(ESM_TOKEN_OFFSET, ESM_TOKEN_OFFSET+TOK_NUM))

def filtering_seqs(seq):
    # avoid free C
    return seq not in ["C"]


def esm_sample_tokens(mask_logits, alphabet, temperature=1.0, sample_num=10, bias_probs=None):
    """Sample tokens from logits with optional temperature and bias"""
    mask_logits_temp = mask_logits / temperature

    probs = F.softmax(mask_logits_temp, dim=-1)
    if bias_probs is not None:
        probs = probs + bias_probs
        probs = probs / probs.sum(dim=-1, keepdim=True)

    predicted_ids = torch.multinomial(probs, num_samples=sample_num, replacement=True)
    predicted_ids = predicted_ids.transpose(0, 1)

    predicted_seqs = []
    for row in predicted_ids:
        tokens = [alphabet.get_tok(idx.item()+ESM_TOKEN_OFFSET) for idx in row]
        seq = "".join(tokens)
        predicted_seqs.append(seq)

    return predicted_seqs, predicted_ids

def ablang_sample_tokens(mask_logits, tokenizer, temperature=1.0, sample_num=10, bias_probs=None):
    """Sample tokens from logits with optional temperature and bias"""
    tokenizer_dict = tokenizer.aa_to_token
    token2aa = {v:k for k,v in tokenizer_dict.items()}
    mask_logits_temp = mask_logits / temperature

    probs = F.softmax(mask_logits_temp, dim=-1)
    if bias_probs is not None:
        probs = probs + bias_probs
        probs = probs / probs.sum(dim=-1, keepdim=True)

    predicted_ids = torch.multinomial(probs, num_samples=sample_num, replacement=True)
    predicted_ids = predicted_ids.transpose(0, 1)

    predicted_seqs = []
    for row in predicted_ids:
        aa_seq = [token2aa[idx.item()+ABLANG2_TOKEN_OFFSET] for idx in row]
        seq = "".join(aa_seq)
        predicted_seqs.append(seq)

    return predicted_seqs, predicted_ids


def mutseq_to_mut(mutseq, wt_seq, chain=None, offset=0, indel_indices=None):
    """Convert mutation sequence to list of mutations in standard format"""
    chain = chain or ""
    if indel_indices is None:
        indices2indel = {i:i+offset for i in range(len(wt_seq))}
    else:
        indices2indel = {i:v for i,v in enumerate(indel_indices)}

    mutations = []
    assert len(mutseq) == len(wt_seq)
    for i, (wt, mut) in enumerate(zip(wt_seq, mutseq)):
        if wt != mut:
            pos = indices2indel[i]
            mutations.append(f"{wt}{chain}{pos}{mut}")

    return mutations


def normalize_scores(scores, num_mutations, mode=None):
    """Normalize scores based on number of mutations"""
    scores = scores.copy()
    scores[num_mutations == 0] = -np.inf
    num_mutations = np.maximum(num_mutations, 1)
    if mode == "sqrt":
        return scores / (np.sqrt(num_mutations))
    elif mode == "log":
        return scores / (np.log(num_mutations))
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--target_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--generation_num", type=int, default=None)
    args = parser.parse_args()

    print("Loading config and model...")
    config = load_config(args.config)

    if os.path.exists(os.path.join(args.target_dir, "generation_result.csv")):
        print("Generation result already exists. Exiting...")
        exit()

    generation_config = config["generation"]
    model_type = config.get("model_type", "sequence")
    assert model_type in ["ablang2", "sequence"], f"Invalid model type: {model_type}"
    print(f"Loading model {model_type}...")
    if model_type == "ablang2":
        model, tokenizer, _ = load_model("ablang2-paired")
        if args.model_path is not None:
            print(f"Loading model from {args.model_path}")
            if config.get("lora_config") is not None:
                lora_config = config["lora_config"]
                peft_config = LoraConfig(**lora_config)
                model = get_peft_model(model, peft_config)
            state_dict = torch.load(args.model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
    elif model_type == "sequence":
        model, alphabet = pretrained.load_model_and_alphabet('esm2_t33_650M_UR50D')
        if args.model_path is not None:
            print(f"Loading model from {args.model_path}")
            if config.get("lora_config") is not None:
                lora_config = config["lora_config"]
                peft_config = LoraConfig(**lora_config)
            model = get_peft_model(model, peft_config)
            state_dict = torch.load(args.model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)

    ensemble_mode = generation_config.get("ensemble_mode", False)
    model.eval()
    model.cuda()
    if model_type == "ablang2":
        print("Preparing sequence and masking...")
        tokenizer_dict = tokenizer.aa_to_token
        wt_seq = config["wildtype_sequence"][config["mutable_chain"]]
        light_wt_seq = config["light_wt_sequence"]
        mask_indices = np.array(config["mask_indices"])
        masked_seq = "".join(["*" if i in mask_indices else wt_seq[i] for i in range(len(wt_seq))])
        wt_token_indices = [tokenizer_dict[wt_seq[i]] for i in mask_indices]
        wt_token_indices = torch.tensor([wt_token_indices])-ABLANG2_TOKEN_OFFSET

        print("Getting model predictions...")
        masked_wt_seq = [masked_seq, light_wt_seq]
        seqs, _ = format_seq_input([masked_wt_seq], fragmented = False)
        tokens = tokenizer(seqs, pad=True, w_extra_tkns=False, device="cuda")
        logits = model(tokens).cpu()
        mask_logits = logits[0, torch.where(tokens==ABLANG2_MASK_TOKEN)[1].cpu()]
        mask_logits = mask_logits[:, ablang_standard_aa_tokens]

        if ensemble_mode:
            print(f"Running ensemble predictions ({generation_config['ensemble_num']} iterations)...")
            ens_logits_list = []
            for i in range(generation_config["ensemble_num"]):
                print(f"Ensemble iteration {i+1}/{generation_config['ensemble_num']}")
                model.train()
                ens_logits = model(tokens).cpu()
                ens_logits = ens_logits[0, torch.where(tokens==ABLANG2_MASK_TOKEN)[1].cpu()]
                ens_logits = ens_logits[:, ablang_standard_aa_tokens]
                ens_logits_list.append(ens_logits)
    elif model_type == "sequence":
        print("Preparing sequence and masking...")
        wt_seq = config["wildtype_sequence"][config["mutable_chain"]]
        mask_indices = np.array(config["mask_indices"])
        masked_seq = "".join(["<mask>" if i in mask_indices else wt_seq[i] for i in range(len(wt_seq))])
        wt_token_indices = [alphabet.tok_to_idx[wt_seq[i]]-ESM_TOKEN_OFFSET for i in mask_indices]
        wt_token_indices = torch.tensor([wt_token_indices]).cpu()
        print("Getting model predictions...")
        masked_input = alphabet.get_batch_converter()([('',masked_seq)])[2].cuda()
        logits_masked = model(masked_input)["logits"]
        logits_masked = logits_masked.cpu()
        mask_logits = logits_masked[0, mask_indices+1, :]
        mask_logits = mask_logits[:, esm_standard_aa_tokens]

        if ensemble_mode:
            print(f"Running ensemble predictions ({generation_config['ensemble_num']} iterations)...")
            ens_logits_list = []
            for i in range(generation_config["ensemble_num"]):
                print(f"Ensemble iteration {i+1}/{generation_config['ensemble_num']}")
                model.train()
                ens_logits = model(masked_input)["logits"].detach().cpu()
                ens_logits = ens_logits[0, mask_indices+1, :]
                ens_logits = ens_logits[:, esm_standard_aa_tokens]
                ens_logits_list.append(ens_logits)

    print("Cleaning up GPU memory...")
    del model
    torch.cuda.empty_cache()
    gc.collect()

    print("Processing logits and sampling sequences...")
    # one-hot encoding for wild type
    bias_probs = torch.zeros_like(mask_logits)
    bias_probs = bias_probs.scatter_(1, wt_token_indices.T, 1) * config["generation"].get("bias_weight", 1.0)
    GENERATION_NUM = generation_config["generation_num"] if args.generation_num is None else args.generation_num
    if model_type == "ablang2":
        predicted_seqs, predicted_ids = ablang_sample_tokens(
            mask_logits,
            tokenizer,
        temperature=generation_config["temperature"],
        sample_num=GENERATION_NUM,
            bias_probs=bias_probs if args.bias else None
        )
    elif model_type == "sequence":
        predicted_seqs, predicted_ids = esm_sample_tokens(
            mask_logits,
            alphabet,
            temperature=generation_config["temperature"],
            sample_num=GENERATION_NUM,
            bias_probs=bias_probs if args.bias else None
        )

    prediction_scores = get_prediction_score(mask_logits, wt_token_indices, predicted_ids)

    print("Processing mutations and scores...")
    predicted_full_seqs = []
    for mutseq in predicted_seqs:
        seq = list(wt_seq)
        for i, mut_aa in zip(mask_indices, mutseq):
            seq[i] = mut_aa
        predicted_full_seqs.append(''.join(seq))
    mutations = [mutseq_to_mut(seq, wt_seq, offset=1) for seq in predicted_full_seqs]
    mutations_str = [",".join(muts) for muts in mutations]
    num_mutations = np.array([len(muts) for muts in mutations])
    normalized_prediction_scores = normalize_scores(
        prediction_scores,
        num_mutations,
        mode=generation_config.get("normalize_mode")
    )

    print("Creating results dataframe...")
    df = pd.DataFrame({
        "mutseq": predicted_seqs,
        "mutations": mutations_str,
        "score": normalized_prediction_scores,
    })
    if ensemble_mode:
        print("Processing ensemble predictions...")
        ens_prediction_scores = [get_prediction_score(ens_logits, wt_token_indices, predicted_ids) for ens_logits in ens_logits_list]
        for i, ens_preds in enumerate(ens_prediction_scores):
            normalized_ens_preds = normalize_scores(
                ens_preds,
                num_mutations,
                mode=generation_config.get("normalize_mode")
            )
            df[f"score_ens_{i}"] = normalized_ens_preds

    print("Sorting and filtering results...")
    sorted_df = df.sort_values(by="score", ascending=False).reset_index(drop=True)
    sorted_df = sorted_df.drop_duplicates(subset="mutseq").reset_index(drop=True)
    sorted_df = sorted_df[sorted_df["mutseq"].apply(filtering_seqs)].reset_index(drop=True)

    print("Saving results...")
    sorted_df.to_csv(os.path.join(args.target_dir, "generation_result.csv"), index=False)
    print(f"Generation result saved to {os.path.join(args.target_dir, 'generation_result.csv')}")
    mask_logits_np = mask_logits.cpu().detach().numpy()
    bias_probs_np = bias_probs.cpu().detach().numpy()
    np.save(os.path.join(args.target_dir, "mask_logits.npy"), {"mask_logits": mask_logits_np, "bias_probs": bias_probs_np})
