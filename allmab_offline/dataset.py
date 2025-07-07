import copy
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import HeteroData, Dataset
from protein_mpnn_utils import parse_PDB, tied_featurize
from tqdm import tqdm
from ablang2.pretrained import format_seq_input

class StructureDataset(Dataset):
    def __init__(self, df, idxs, structure_path, batch_size, esm_alphabet, evaluation=True, predict=False, seed_bias=0, dms_column='DMS_score'):
        super().__init__()
        self.stage = 'train' if not predict else 'test'
        self.df = df
        self.idxs = idxs
        self.batch_size = batch_size
        self.evaluation = evaluation
        self.predict = predict
        self.seed_bias = seed_bias
        self.dms_column = dms_column

        # Collects information needed for training or evaluation.
        # In training mode, multiple indices can map to multiple rows.
        # In evaluation mode, we store a single combined feature set.
        if not evaluation:
            # Summation of the number of rows (when idx corresponds to multiple rows in df).
            self.n = 0
            self.batch = {}
            for idx in tqdm(idxs):
                # もし df に同じ idx が複数行あれば、その分だけ足す
                sub_df = df.loc[[idx]] if isinstance(df.loc[idx], (pd.Series)) else df.loc[idx]
                if sub_df.ndim == 1:
                    sub_df = sub_df.to_frame().T
                self.n += len(sub_df)

                # 構造を読み込み、各 idx に対して featurize した結果を辞書に格納
                poi = sub_df.iloc[0]['POI']
                pdb_dict_list = parse_PDB(f'{structure_path}/{poi}.pdb', ca_only=False)
                all_chain_list = [c[-1:] for c in pdb_dict_list[0] if c.startswith('seq_chain')]
                designed_chain_list = all_chain_list
                fixed_chain_list = [ch for ch in all_chain_list if ch not in designed_chain_list]
                chain_id_dict = {pdb_dict_list[0]['name']: (designed_chain_list, fixed_chain_list)}
                self.batch[idx] = tied_featurize(
                    pdb_dict_list,
                    'cpu',
                    chain_id_dict,
                    None, None, None, None, None, False
                )
        else:
            self.n = len(idxs)
            pdb_dict_list = []
            chain_id_dict = {}
            self.poi_dic = {}
            unique_pois = df['POI'].unique()
            for i, poi in enumerate(unique_pois):
                self.poi_dic[poi] = i
                parsed = parse_PDB(f'{structure_path}/{poi}.pdb', ca_only=False)
                pdb_dict_list += parsed
                all_chain_list = sorted([c[-1:] for c in parsed[-1] if c.startswith('seq_chain')])
                designed_chain_list = all_chain_list
                fixed_chain_list = [ch for ch in all_chain_list if ch not in designed_chain_list]
                chain_id_dict[parsed[-1]['name']] = (designed_chain_list, fixed_chain_list)
            self.batch = tied_featurize(
                pdb_dict_list,
                'cpu',
                chain_id_dict,
                None, None, None, None, None, False
            )

        # このアルファベットを使ってアミノ酸を整数に変換
        alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
        self.alphabet_dict = dict(zip(alphabet, range(len(alphabet))))

    def __len__(self):
        # 評価時は単純に idxs の長さ、学習時は登録データ総数を返す
        if self.evaluation:
            return len(self.idxs)
        else:
            return min(self.batch_size * 256, self.n)

    def __getitem__(self, index):
        # 評価用のデータ取得
        if self.evaluation:
            idx = self.idxs[index]
            seq = self.df.loc[idx, 'mutated_sequence']
            mseq = list(seq)
            mutant = self.df.loc[idx, 'mutant']
            poi = self.df.loc[idx, 'POI']

            # reg_label = -self.df.loc[idx, 'ddg'] if self.dms_column not in self.df.columns else self.df.loc[idx, self.dms_column]
            reg_label = self.df.loc[idx, self.dms_column]
            # tied_featurize でまとめられた出力を該当 POI だけ抽出
            X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, \
            visible_list_list, masked_list_list, masked_chain_length_list_list, \
            chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, \
            tied_pos_list_of_lists_list, pssm_coef, pssm_bias, \
            pssm_log_odds_all, bias_by_res_all, tied_beta = self.batch

            # POI 毎の番号に応じて切り出し
            poi_idx = self.poi_dic[poi]
            X = X[[poi_idx]]
            S = S[[poi_idx]]
            mask = mask[[poi_idx]]
            chain_M = chain_M[[poi_idx]]
            residue_idx = residue_idx[[poi_idx]]
            chain_encoding_all = chain_encoding_all[[poi_idx]]

        # 学習用のデータ取得
        else:
            # インデックスごとに乱数シードを設定し、一意のサンプリングを行う
            seed = index + self.seed_bias * 1000000
            np.random.seed(seed)

            # idxs からランダムに 1つ選び、その idx に紐づく df の行からさらに 1つサンプリング
            chosen_idx = np.random.choice(self.idxs)
            sub_df = self.df.loc[[chosen_idx]] if isinstance(self.df.loc[chosen_idx], (pd.Series)) else self.df.loc[chosen_idx]
            if sub_df.ndim == 1:
                sub_df = sub_df.to_frame().T
            selected_row = sub_df.sample(n=1, random_state=seed).iloc[0]

            seq = selected_row['mutated_sequence']
            mseq = list(seq)
            mutant = selected_row['mutant']
            reg_label = selected_row[self.dms_column]  # DMS_score が存在する想定

            X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, \
            visible_list_list, masked_list_list, masked_chain_length_list_list, \
            chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, \
            tied_pos_list_of_lists_list, pssm_coef, pssm_bias, \
            pssm_log_odds_all, bias_by_res_all, tied_beta = self.batch[chosen_idx]

        # mutant がある場合は、変異部位を記録やマスクの操作などを行う
        mpos = []
        if mutant is not None and mutant == mutant and len(mutant) > 0:
            for m in mutant.split(':'):
                try:
                    pos = int(m[1:-1])
                    mpos.append(pos)
                    if not self.evaluation:
                        mseq[pos - 1] = 'X'
                except:
                    print("Mutant Error", m, mutant)
                    continue
        mseq = ''.join(mseq)

        # グラフとして格納
        graph = HeteroData()
        graph['protein'].X = X
        graph['protein'].wt_S = S
        graph['protein'].S = copy.deepcopy(S)
        S_input = torch.tensor([self.alphabet_dict.get(AA, 20) for AA in mseq])
        graph['protein'].S[0, :len(seq)] = S_input
        graph['protein'].mask = mask
        graph['protein'].chain_M = chain_M
        graph['protein'].residue_idx = residue_idx
        graph['protein'].chain_encoding_all = chain_encoding_all
        graph['batch'].x = torch.ones(len(seq), 1)

        target_S = copy.deepcopy(S)
        target_S[0, :len(seq)] = torch.tensor([self.alphabet_dict.get(AA, 20) for AA in seq])
        graph.token_idx = torch.cat([S, target_S], dim=0).long()[None, ..., None]

        if not self.predict:
            graph.reg_labels = torch.tensor([[reg_label]]).float()

        return graph


class SequenceDataset(Dataset):
    def __init__(self, df, idxs, structure_path, batch_size, esm_alphabet, evaluation=True, predict=False, seed_bias=0, dms_column='DMS_score'):
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

class SequenceDataset2(Dataset):
    def __init__(self, df, idxs, structure_path, batch_size, esm_alphabet, evaluation=True, predict=False, seed_bias=0, dms_column='DMS_score'):
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
            base_seq = self.df.loc[idx, 'wildtype_sequence']
            seq = self.df.loc[idx, 'mutated_sequence']
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

            base_seq = selected_row['wildtype_sequence']
            seq = selected_row['mutated_sequence']
            reg_label = selected_row[self.dms_column]

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

class SequenceDataset_AB(Dataset):
    def __init__(self, df, idxs, structure_path, batch_size, esm_alphabet, evaluation=True, predict=False, seed_bias=0, dms_column='DMS_score'):
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
            base_seq = self.df.loc[idx, 'wildtype_heavy']+self.df.loc[idx, 'wildtype_light']
            seq = self.df.loc[idx, 'heavy']+self.df.loc[idx, 'light']
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
            base_seq = base_hseq+base_lseq
            seq = hseq+lseq
            reg_label = selected_row[self.dms_column]

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

