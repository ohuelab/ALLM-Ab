import torch
from ablang2.pretrained import format_seq_input

TOK_NUM = 20
ABLANG2_TOKEN_OFFSET=1
ABLANG2_MASK_TOKEN=23
ESM2_TOKEN_OFFSET = 4

ablang2_standard_aa_token_indices = list(range(ABLANG2_TOKEN_OFFSET,ABLANG2_TOKEN_OFFSET+TOK_NUM))
esm2_standard_aa_tokens = list(range(ESM2_TOKEN_OFFSET, ESM2_TOKEN_OFFSET+TOK_NUM))


def get_perplexity(logits, token_ids):
    """Calculate perplexity for masked tokens"""

    logits_expanded = logits.expand(len(token_ids), len(token_ids[0]), TOK_NUM)
    token_logits = torch.gather(logits_expanded, 2, token_ids.unsqueeze(2))
    # Apply log softmax and gather log probabilities for the true tokens
    log_probs = torch.nn.functional.log_softmax(logits_expanded, dim=2)
    token_log_probs = torch.gather(log_probs, 2, token_ids.unsqueeze(2))
    # Calculate mean negative log likelihood across positions
    nll = -torch.mean(token_log_probs.squeeze(), dim=1)
    # Convert to perplexity
    perplexity = torch.exp(nll)
    return perplexity.detach().numpy()


def esm2_perplexity(sequences, model, alphabet, masked_wt_seq, mask_indices, device="cuda", mask_logits = None):
    assert isinstance(sequences[0], str), "sequences must be a list of strings"
    assert isinstance(masked_wt_seq, str), "masked_wt_seq must be a string"
    # Get model logits for masked sequence
    if mask_logits is None:
        masked_input = alphabet.get_batch_converter()([('',masked_wt_seq)])[2].to(device)
        logits = model(masked_input)["logits"].cpu()
        mask_logits = logits[0, mask_indices+1, :]
        mask_logits = mask_logits[:, esm2_standard_aa_tokens]

    # Get token indices for sequences
    token_indices_list = []
    for mt_seq in sequences:
        token_indices = [alphabet.tok_to_idx[mt_seq[i]]-ESM2_TOKEN_OFFSET for i in mask_indices]
        token_indices = torch.tensor(token_indices).cpu()
        token_indices_list.append(token_indices)
    token_indices_list = torch.stack(token_indices_list)

    return get_perplexity(mask_logits, token_indices_list)

def ablang2_perplexity(paired_sequences, model, tokenizer, masked_wt_seq, mask_indices, device="cuda", mask_logits = None):
    assert isinstance(paired_sequences[0], tuple) or isinstance(paired_sequences[0], list), "sequences must be a list of tuples or lists"
    assert isinstance(masked_wt_seq, tuple) or isinstance(masked_wt_seq, list), "masked_wt_seq must be a tuple or list"
    tokenizer_dict = tokenizer.aa_to_token

    # Get model logits at mask positions
    if mask_logits is None:
        seqs, _ = format_seq_input([masked_wt_seq], fragmented = False)
        tokens = tokenizer(seqs, pad=True, w_extra_tkns=False, device=device)

        logits = model(tokens).cpu()
        mask_logits = logits[0, torch.where(tokens==ABLANG2_MASK_TOKEN)[1].cpu()]
        mask_logits = mask_logits[:, ablang2_standard_aa_token_indices]

    # Get token indices for sequences
    token_indices_list = []
    for mt_hseq, _ in paired_sequences:
        token_indices = [tokenizer_dict[mt_hseq[i]] for i in mask_indices]
        token_indices = torch.tensor(token_indices) - ABLANG2_TOKEN_OFFSET
        token_indices_list.append(token_indices)
    token_indices_list = torch.stack(token_indices_list)

    return get_perplexity(mask_logits, token_indices_list)


def get_prediction_score(logits, wt_token_ids, mutation_ids):
    """Calculate prediction scores for mutations compared to wild type"""
    logits_expanded = logits.expand(len(mutation_ids), len(mutation_ids[0]), TOK_NUM)
    wt_logits = torch.gather(logits, 1, wt_token_ids)
    mutation_logits = torch.gather(logits_expanded, 2, mutation_ids.unsqueeze(2))
    wt_logits_sum = torch.sum(wt_logits, dim=1)[0]
    mutation_logits_sum = torch.sum(mutation_logits, dim=1).reshape(-1)
    return (mutation_logits_sum - wt_logits_sum).detach().numpy()


def esm2_logit_inference(sequences, model, alphabet, wt_seq, masked_wt_seq, mask_indices, device="cuda", mask_logits = None):
    assert isinstance(sequences[0], str), "sequences must be a list of strings"
    assert isinstance(wt_seq, str), "wt_seq must be a string"
    assert isinstance(masked_wt_seq, str), "masked_wt_seq must be a string"
    # Get model logits for masked sequence

    wt_token_indices = [alphabet.tok_to_idx[wt_seq[i]]-ESM2_TOKEN_OFFSET for i in mask_indices]
    wt_token_indices = torch.tensor([wt_token_indices]).cpu()
    if mask_logits is None:
        masked_input = alphabet.get_batch_converter()([('',masked_wt_seq)])[2].to(device)
        logits = model(masked_input)["logits"].cpu()

        # Get logits at mask positions
        mask_logits = logits[0, mask_indices+1, :]
        mask_logits = mask_logits[:, esm2_standard_aa_tokens]

    # Get mutation token indices
    mt_token_indices_list = []
    for mt_seq in sequences:
        mt_token_indices = [alphabet.tok_to_idx[mt_seq[i]]-ESM2_TOKEN_OFFSET for i in mask_indices]
        mt_token_indices = torch.tensor(mt_token_indices).cpu()
        mt_token_indices_list.append(mt_token_indices)
    mt_token_indices_list = torch.stack(mt_token_indices_list)

    return get_prediction_score(mask_logits, wt_token_indices, mt_token_indices_list)

def ablang2_logit_inference(paired_sequences, model, tokenizer, wt_seq, masked_wt_seq, mask_indices, device="cuda", mask_logits = None):
    assert isinstance(paired_sequences[0], tuple) or isinstance(paired_sequences[0], list), "sequences must be a list of tuples or lists"
    assert isinstance(wt_seq, tuple) or isinstance(wt_seq, list), "wt_seq must be a tuple or list"
    assert isinstance(masked_wt_seq, tuple) or isinstance(masked_wt_seq, list), "masked_wt_seq must be a tuple or list"
    tokenizer_dict = tokenizer.aa_to_token

    # Split sequences and indices for heavy and light chains
    hseq, lseq = wt_seq
    h_mask_indices, l_mask_indices = mask_indices

    # Get wild type token indices for both chains
    h_wt_token_indices = [tokenizer_dict[hseq[i]] for i in h_mask_indices]
    l_wt_token_indices = [tokenizer_dict[lseq[i]] for i in l_mask_indices]

    if mask_logits is None:

        seqs, _ = format_seq_input([masked_wt_seq], fragmented = False)
        tokens = tokenizer(seqs, pad=True, w_extra_tkns=False, device=device)
        mask_indices = torch.where(tokens==ABLANG2_MASK_TOKEN)[1].cpu()

        logits = model(tokens).cpu()
        mask_logits = logits[0, mask_indices]
        mask_logits = mask_logits[:, ablang2_standard_aa_token_indices]

    # Get mutation token indices
    wt_token_indices = torch.tensor([h_wt_token_indices+l_wt_token_indices])-ABLANG2_TOKEN_OFFSET
    mt_token_indices_list = []
    for mt_hseq, mt_lseq in paired_sequences:
        h_mt_token_indices = [tokenizer_dict[mt_hseq[i]] for i in h_mask_indices]
        l_mt_token_indices = [tokenizer_dict[mt_lseq[i]] for i in l_mask_indices]
        mt_token_indices = torch.tensor(h_mt_token_indices+l_mt_token_indices)-ABLANG2_TOKEN_OFFSET
        mt_token_indices_list.append(mt_token_indices)
    mt_token_indices_list = torch.stack(mt_token_indices_list)
    return get_prediction_score(mask_logits, wt_token_indices, mt_token_indices_list)
