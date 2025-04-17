import os
import yaml
from easydict import EasyDict

class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream):
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.load(f, Loader))
    return config


def DMS_file_for_LLM(df,focus=False):
    df['chain_id'] = df['chain_id'].fillna('')
    df['wildtype_sequence'] = df['wildtype_sequence'].apply(eval)
    df['mutant'] = df['mutant'].apply(eval)
    df['mutated_sequence'] = df['mutated_sequence'].apply(eval)
    input_wt_seqs = []
    input_mt_seqs = []
    input_focus_wt_seqs = []
    input_focus_mt_seqs = []
    input_mutants = []
    input_focus_mutants = []
    focus_chains = []
    for i in df.index:
        mutants = df.loc[i,'mutant']
        for c in mutants:
            if c not in focus_chains:
                if mutants[c] != '':
                    focus_chains.append(c)
    for i in df.index:
        chain_ids = df.loc[i,'chain_id']
        wt_seqs = ''
        mt_seqs = ''
        focus_wt_seqs = ''
        focus_mt_seqs = ''
        wt_seq_dic = df.loc[i,'wildtype_sequence']
        mt_seq_dic = df.loc[i,'mutated_sequence']
        mutants = df.loc[i,'mutant']
        revise_mutants = []
        focus_revise_mutants = []
        start_idx = 0
        focus_start_idx = 0
        for i,chain_id in enumerate(chain_ids):
            ms = mutants.get(chain_id,"")
            if ms != '':
                for m in ms.split(':'):
                    pos = int(m[1:-1]) + start_idx
                    revise_mutants.append(m[:1]+str(pos)+m[-1:])
            wt_seqs += wt_seq_dic[chain_id]
            mt_seqs += mt_seq_dic[chain_id]
            start_idx += len(wt_seq_dic[chain_id])
            if chain_id in focus_chains:
                if ms != '':
                    for m in ms.split(':'):
                        pos = int(m[1:-1]) + focus_start_idx
                        focus_revise_mutants.append(m[:1]+str(pos)+m[-1:])
                focus_wt_seqs += wt_seq_dic[chain_id]
                focus_mt_seqs += mt_seq_dic[chain_id]
                focus_start_idx += len(wt_seq_dic[chain_id])


        input_wt_seqs.append(wt_seqs)
        input_mt_seqs.append(mt_seqs)
        input_mutants.append(':'.join(revise_mutants))

        input_focus_wt_seqs.append(focus_wt_seqs)
        input_focus_mt_seqs.append(focus_mt_seqs)
        input_focus_mutants.append(':'.join(focus_revise_mutants))
    if not focus:
        df['wildtype_sequence'] = input_wt_seqs
        df['mutated_sequence'] = input_mt_seqs
        df['mutant'] = input_mutants
    else:
        df['wildtype_sequence'] = input_focus_wt_seqs
        df['mutated_sequence'] = input_focus_mt_seqs
        df['mutant'] = input_focus_mutants

    return df
