import copy
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import HeteroData, Dataset
from tqdm import tqdm
from ablang2.pretrained import format_seq_input

class SequenceDataset(Dataset):
    def __init__(self, df, idxs, batch_size, esm_alphabet, evaluation=True, predict=False, seed_bias=0, dms_column='DMS_score'):
        super().__init__()
        self.stage = 'train' if not predict else 'test'
        self.df = df
        self.idxs = idxs
        self.batch_size = batch_size
        self.esm_alphabet = esm_alphabet
        self.esm_alphabet_dic = esm_alphabet.to_dict()
        self.evaluation = evaluation
        self.predict = predict
        self.seed_bias = seed_bias
        self.dms_column = dms_column

        # 総サンプル数を取得
        if not evaluation:
            self.n = 0
            for idx in idxs:
                sub_df = df.loc[[idx]] if isinstance(df.loc[idx], (pd.Series)) else df.loc[idx]
                if sub_df.ndim == 1:
                    sub_df = sub_df.to_frame().T
                self.n += len(sub_df)
        else:
            self.n = len(idxs)

        # アミノ酸を整数に変換する辞書
        alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
        self.alphabet_dict = dict(zip(alphabet, range(len(alphabet))))

    def __len__(self):
        if self.evaluation:
            return len(self.idxs)
        else:
            return min(self.batch_size * 256, self.n)

    def __getitem__(self, index):
        if self.evaluation:
            idx = self.idxs[index]
            seq = self.df.loc[idx, 'mutated_sequence']
            mseq = list(seq)
            mutant = self.df.loc[idx, 'mutant']
            # reg_label = -self.df.loc[idx, 'ddg'] if self.dms_column not in self.df.columns else self.df.loc[idx, self.dms_column]
            reg_label = self.df.loc[idx, self.dms_column]
        else:
            seed = index + self.seed_bias * 1000000
            np.random.seed(seed)
            chosen_idx = np.random.choice(self.idxs)
            sub_df = self.df.loc[[chosen_idx]] if isinstance(self.df.loc[chosen_idx], (pd.Series)) else self.df.loc[chosen_idx]
            if sub_df.ndim == 1:
                sub_df = sub_df.to_frame().T
            selected_row = sub_df.sample(n=1, random_state=seed).iloc[0]

            seq = selected_row['mutated_sequence']
            mseq = list(seq)
            mutant = selected_row['mutant']
            reg_label = selected_row[self.dms_column]

        # 変異文字をベース配列に反映
        if mutant is not None and mutant == mutant and len(mutant) > 0:
            for m in mutant.split(':'):
                try:
                    pos = int(m[1:-1]) - 1
                    mseq[pos] = m[0]
                except:
                    print("Mutant Error", m, mutant)
                    continue

        base_seq = ''.join(mseq)

        # 差分箇所を <mask> トークンとして埋め込み
        esm_token_idxs = []
        esm_base_token_idxs = []
        esm_mask_seq = ''
        for i, aa in enumerate(seq):
            # 差分がある場合、または全く差分がないまま最後に達した場合にマスクを入れる
            if aa != base_seq[i] or (i == len(seq) - 1 and len(esm_token_idxs) == 0):
                esm_mask_seq += '<mask>'
                esm_token_idxs.append(self.esm_alphabet_dic[aa])
                esm_base_token_idxs.append(self.esm_alphabet_dic[base_seq[i]])
            else:
                esm_mask_seq += aa

        # ESM のトークン変換
        graph = HeteroData()
        # このコンバータは (labels, seq) の形式で渡す必要がある
        esm_x = self.esm_alphabet.get_batch_converter()([('', esm_mask_seq)])
        graph.esm_token_idx = torch.tensor([esm_base_token_idxs, esm_token_idxs]).long().T
        graph['protein'].x = esm_x[2][0]

        if not self.predict:
            graph.reg_labels = torch.tensor([[reg_label]]).float()

        return graph

class AbLang2Dataset(Dataset):
    def __init__(self, df, idxs, batch_size, tokenizer, evaluation=True, predict=False, seed_bias=0, dms_column='DMS_score'):
        super().__init__()
        self.stage = 'train' if not predict else 'test'
        self.df = df
        self.idxs = idxs
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.tokenizer_dict = tokenizer.aa_to_token
        self.evaluation = evaluation
        self.predict = predict
        self.seed_bias = seed_bias
        self.dms_column = dms_column

        # 総サンプル数を取得
        if not evaluation:
            self.n = 0
            for idx in idxs:
                sub_df = df.loc[[idx]] if isinstance(df.loc[idx], (pd.Series)) else df.loc[idx]
                if sub_df.ndim == 1:
                    sub_df = sub_df.to_frame().T
                self.n += len(sub_df)
        else:
            self.n = len(idxs)

        # アミノ酸を整数に変換する辞書
        alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
        self.alphabet_dict = dict(zip(alphabet, range(len(alphabet))))

    def __len__(self):
        if self.evaluation:
            return len(self.idxs)
        else:
            return min(self.batch_size * 256, self.n)

    def __getitem__(self, index):
        if self.evaluation:
            idx = self.idxs[index]
            base_hseq = self.df.loc[idx, 'wildtype_heavy']
            base_lseq = self.df.loc[idx, 'wildtype_light']
            hseq = self.df.loc[idx, 'heavy']
            lseq = self.df.loc[idx, 'light']
            # reg_label = -self.df.loc[idx, 'ddg'] if self.dms_column not in self.df.columns else self.df.loc[idx, self.dms_column]
            reg_label = self.df.loc[idx, self.dms_column]
        else:
            seed = index + self.seed_bias * 1000000
            np.random.seed(seed)
            chosen_idx = np.random.choice(self.idxs)
            sub_df = self.df.loc[[chosen_idx]] if isinstance(self.df.loc[chosen_idx], (pd.Series)) else self.df.loc[chosen_idx]
            if sub_df.ndim == 1:
                sub_df = sub_df.to_frame().T
            selected_row = sub_df.sample(n=1, random_state=seed).iloc[0]

            base_hseq = selected_row['wildtype_heavy']
            base_lseq = selected_row['wildtype_light']
            hseq = selected_row['heavy']
            lseq = selected_row['light']
            assert len(base_hseq) == len(hseq)
            assert len(base_lseq) == len(lseq)
            reg_label = selected_row[self.dms_column]

        # 差分箇所を*トークンとして埋め込み
        token_idxs = []
        base_token_idxs = []
        mask_hseq = ''
        for i, aa in enumerate(hseq):
            # 差分がある場合、または全く差分がないまま最後に達した場合にマスクを入れる
            if aa != base_hseq[i] or (i == len(hseq) - 1 and len(token_idxs) == 0):
                mask_hseq += '*'
                token_idxs.append(self.tokenizer_dict[aa])
                base_token_idxs.append(self.tokenizer_dict[base_hseq[i]])
            else:
                mask_hseq += aa
        mask_lseq = ''
        for i, aa in enumerate(lseq):
            if aa != base_lseq[i] or (i == len(lseq) - 1 and len(token_idxs) == 0):
                mask_lseq += '*'
                token_idxs.append(self.tokenizer_dict[aa])
                base_token_idxs.append(self.tokenizer_dict[base_lseq[i]])
            else:
                mask_lseq += aa

        masked_seq = (mask_hseq, mask_lseq)
        seqs, _ = format_seq_input([masked_seq], fragmented = False)
        tokens = self.tokenizer(seqs, pad=True, w_extra_tkns=False, device="cpu")

        # ESM のトークン変換
        graph = HeteroData()
        graph.token_idx = torch.tensor([base_token_idxs, token_idxs]).long().T
        graph['protein'].x = tokens.squeeze(0)

        if not self.predict:
            graph.reg_labels = torch.tensor([[reg_label]]).float()

        return graph
