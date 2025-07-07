import torch
from ablang2.pretrained import format_seq_input

TOK_NUM = 20
ABLANG2_TOKEN_OFFSET=1
ABLANG2_MASK_TOKEN=23
ESM2_TOKEN_OFFSET = 4

ablang2_standard_aa_token_indices = list(range(ABLANG2_TOKEN_OFFSET,ABLANG2_TOKEN_OFFSET+TOK_NUM))
esm2_standard_aa_tokens = list(range(ESM2_TOKEN_OFFSET, ESM2_TOKEN_OFFSET+TOK_NUM))


def get_prediction_score(logits, wt_token_ids, mutation_ids):
    """Calculate prediction scores for mutations compared to wild type"""
    logits_expanded = logits.expand(len(mutation_ids), len(mutation_ids[0]), TOK_NUM)
    wt_logits = torch.gather(logits, 1, wt_token_ids)
    mutation_logits = torch.gather(logits_expanded, 2, mutation_ids.unsqueeze(2))
    wt_logits_sum = torch.sum(wt_logits, dim=1)[0]
    mutation_logits_sum = torch.sum(mutation_logits, dim=1).reshape(-1)
    return (mutation_logits_sum - wt_logits_sum).detach().numpy()


def esm2_logit_inference(df, model, alphabet, wt_seq, masked_wt_seq, mask_indices):
    wt_seq = wt_seq[0]+wt_seq[1]
    masked_wt_seq = masked_wt_seq[0]+masked_wt_seq[1]
    mask_indices = mask_indices[0]+[len(mask_indices[0])+i for i in mask_indices[1]]

    # Get model logits for masked sequence
    masked_input = alphabet.get_batch_converter()([('',masked_wt_seq)])[2].cuda()
    logits = model.esm_pretrain_model(masked_input)["logits"].cpu()

    # Get wild type token indices
    wt_token_indices = [alphabet.tok_to_idx[wt_seq[i]]-ESM2_TOKEN_OFFSET for i in mask_indices]
    wt_token_indices = torch.tensor([wt_token_indices]).cpu()

    # Get logits at mask positions
    mask_logits = logits[0, torch.tensor(mask_indices)+1, :]
    mask_logits = mask_logits[:, esm2_standard_aa_tokens]

    # Get mutation token indices
    mt_token_indices_list = []
    for i, d in df.iterrows():
        mt_hseq = d["heavy"]
        mt_lseq = d["light"]
        mt_seq = mt_hseq+mt_lseq
        mt_token_indices = [alphabet.tok_to_idx[mt_seq[i]]-ESM2_TOKEN_OFFSET for i in mask_indices]
        mt_token_indices = torch.tensor(mt_token_indices).cpu()
        mt_token_indices_list.append(mt_token_indices)
    mt_token_indices_list = torch.stack(mt_token_indices_list)
    return get_prediction_score(mask_logits, wt_token_indices, mt_token_indices_list)

def ablang2_logit_inference(df, model, tokenizer, wt_seq, masked_wt_seq, mask_indices):
    tokenizer_dict = tokenizer.aa_to_token

    # Split sequences and indices for heavy and light chains
    hseq, lseq = wt_seq
    h_mask_indices, l_mask_indices = mask_indices

    # Get wild type token indices for both chains
    h_wt_token_indices = [tokenizer_dict[hseq[i]] for i in h_mask_indices]
    l_wt_token_indices = [tokenizer_dict[lseq[i]] for i in l_mask_indices]

    # Tokenize masked sequence
    seqs, _ = format_seq_input([masked_wt_seq], fragmented = False)
    tokens = tokenizer(seqs, pad=True, w_extra_tkns=False, device="cpu").cuda()
    mask_indices = torch.where(tokens==ABLANG2_MASK_TOKEN)[1].cpu()

    # Get model logits at mask positions
    logits = model.ablang2_pretrain_model(tokens).cpu()
    mask_logits = logits[0, mask_indices]
    mask_logits = mask_logits[:, ablang2_standard_aa_token_indices]

    # Get mutation token indices
    wt_token_indices = torch.tensor([h_wt_token_indices+l_wt_token_indices])-ABLANG2_TOKEN_OFFSET
    mt_token_indices_list = []
    for i, d in df.iterrows():
        mt_hseq = d["heavy"]
        mt_lseq = d["light"]
        h_mt_token_indices = [tokenizer_dict[mt_hseq[i]] for i in h_mask_indices]
        l_mt_token_indices = [tokenizer_dict[mt_lseq[i]] for i in l_mask_indices]
        mt_token_indices = torch.tensor(h_mt_token_indices+l_mt_token_indices)-ABLANG2_TOKEN_OFFSET
        mt_token_indices_list.append(mt_token_indices)
    mt_token_indices_list = torch.stack(mt_token_indices_list)
    return get_prediction_score(mask_logits, wt_token_indices, mt_token_indices_list)
