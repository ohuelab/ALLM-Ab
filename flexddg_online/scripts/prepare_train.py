#!/usr/bin/python3

import os
import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import multiprocessing
from utils import load_config

# Constants
ROSETTA_OUTPUT_FILE = 'rosetta.out'
OUTPUT_DB_FILE = 'ddG.db3'
TRAJECTORY_STRIDE = 5
DEFAULT_PROCESSES = 12

# GAM parameters for score terms
ZEMU_GAM_PARAMS = {
    'fa_sol':      (6.940, -6.722),
    'hbond_sc':    (1.902, -1.999),
    'hbond_bb_sc': (0.063,  0.452),
    'fa_rep':      (1.659, -0.836),
    'fa_elec':     (0.697, -0.122),
    'hbond_lr_bb': (2.738, -1.179),
    'fa_atr':      (2.313, -1.649),
}

def mutations_to_seq(mutations, wt_seq, offset=0, indel_indices=None):
    if indel_indices is None:
        indel2indices = {i+offset:i for i in range(len(wt_seq))}
    else:
        indel2indices = {v:i for i,v in enumerate(indel_indices)}
    mutseq = list(wt_seq)

    for mutation in mutations:
        # wt, pos, mut = mutation[0], int(mutation[2:-1]) - offset, mutation[-1]
        wt, pos, mut = mutation[0], int(indel2indices[int(mutation[1:-1])]), mutation[-1]
        assert wt == mutseq[pos], f"{wt}!={mutseq[pos]}, {mutation}, {mutseq}"
        mutseq[pos] = mut
    return ''.join(mutseq)

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze ddG output')
    parser.add_argument('--analyze', type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--mutations_file', type=str)
    parser.add_argument('--processes', type=int, help='Number of processes to use', default=DEFAULT_PROCESSES)
    parser.add_argument('--output_dir', type=str, help='output directory')
    parser.add_argument('--previous_data', type=str, help='previous data to use')
    parser.add_argument('--cycle', type=int, help='cycle number', default=0)
    parser.add_argument('--output_dir_name', type=str, help='name of output directory', default="outputs")
    return parser.parse_args()

def gam_function(x, score_term):
    """Apply GAM transformation to a score term"""
    a, b = ZEMU_GAM_PARAMS[score_term]
    return -1.0 * np.exp(a) + 2.0 * np.exp(a) / (1.0 + np.exp(-1.0 * x * np.exp(b)))

def apply_zemu_gam(scores):
    """Apply GAM transformations to all relevant score terms"""
    scores = scores.copy()
    scores.drop('total_score', axis=1, inplace=True)

    for score_term in ZEMU_GAM_PARAMS:
        assert score_term in scores.columns
        scores[score_term] = scores[score_term].apply(gam_function, score_term=score_term)

    scores['total_score'] = scores[list(ZEMU_GAM_PARAMS.keys())].sum(axis=1)
    scores['score_function_name'] += '-gam'
    return scores


class RosettaOutputAnalyzer:
    def __init__(self, output_folder, process_count=-1):
        self.output_folder = output_folder
        self.process_count = multiprocessing.cpu_count() if process_count == -1 else process_count

    def check_output_success(self, struct_dir):
        """Check if Rosetta output files exist and are valid"""
        rosetta_out = os.path.join(struct_dir, ROSETTA_OUTPUT_FILE)
        db3_file = os.path.join(struct_dir, OUTPUT_DB_FILE)
        return os.path.isfile(rosetta_out) and os.path.isfile(db3_file)

    def find_finished_jobs(self):
        """Find all completed job directories"""
        jobs = {}
        for job_dir in [d for d in os.listdir(self.output_folder)
                       if os.path.isdir(os.path.join(self.output_folder, d))]:
            job_path = os.path.abspath(os.path.join(self.output_folder, job_dir))
            completed_structs = []

            for struct_dir in sorted([d for d in os.listdir(job_path)
                                    if os.path.isdir(os.path.join(job_path, d))]):
                struct_path = os.path.abspath(os.path.join(job_path, struct_dir))
                if self.check_output_success(struct_path):
                    completed_structs.append(struct_path)

            jobs[job_path] = completed_structs
        return jobs

    def process_db3_file(self, db3_path, struct_num, case_name):
        """Extract scores from SQLite database"""
        with sqlite3.connect(db3_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            num_batches = cursor.execute('SELECT max(batch_id) from batches').fetchone()[0]

            scores = pd.read_sql_query('''
                SELECT
                    batches.name,
                    structure_scores.struct_id,
                    score_types.score_type_name,
                    structure_scores.score_value,
                    score_function_method_options.score_function_name
                FROM structure_scores
                INNER JOIN batches ON batches.batch_id=structure_scores.batch_id
                INNER JOIN score_function_method_options ON score_function_method_options.batch_id=batches.batch_id
                INNER JOIN score_types ON score_types.batch_id=structure_scores.batch_id
                   AND score_types.score_type_id=structure_scores.score_type_id
            ''', conn)

        # Process scores DataFrame
        scores['struct_id'] = scores['struct_id'].apply(
            lambda x: TRAJECTORY_STRIDE * (1 + (int(x - 1) // num_batches)))
        scores['name'] = scores['name'].apply(
            lambda x: x[:-9] if x.endswith('_dbreport') else x)

        # Pivot and format
        scores = scores.pivot_table(
            index=['name', 'struct_id', 'score_function_name'],
            columns='score_type_name',
            values='score_value'
        ).reset_index()

        scores.rename(columns={
            'name': 'state',
            'struct_id': 'backrub_steps'
        }, inplace=True)

        scores['struct_num'] = struct_num
        scores['case_name'] = case_name

        return scores

    def calc_ddg(self, scores):
        """
        Calculates the ddG (delta-delta-G) by combining bound/unbound states
        and aggregating them over a range of structure counts.
        Returns a tuple containing the combined ddG DataFrame and
        a detailed structure-level score DataFrame.
        """
        total_structs = np.max(scores['struct_num'])
        nstructs_to_analyze = set([total_structs])
        for x in range(10, total_structs):
            if x % 10 == 0:
                nstructs_to_analyze.add(x)
        nstructs_to_analyze = sorted(nstructs_to_analyze)

        all_ddg_scores = []
        struct_scores = None

        for nstructs in nstructs_to_analyze:
            ddg_scores = scores.loc[
                ((scores['state'] == 'unbound_mut') | (scores['state'] == 'bound_wt')) &
                (scores['struct_num'] <= nstructs)
            ].copy()

            for column in ddg_scores.columns:
                if column not in ['state', 'case_name', 'backrub_steps', 'struct_num', 'score_function_name']:
                    ddg_scores.loc[:, column] *= -1.0

            ddg_scores = pd.concat(
                [
                    ddg_scores,
                    scores.loc[
                        ((scores['state'] == 'unbound_wt') | (scores['state'] == 'bound_mut')) &
                        (scores['struct_num'] <= nstructs)
                    ].copy()
                ]
            )

            ddg_scores = ddg_scores.groupby(
                ['case_name', 'backrub_steps', 'struct_num', 'score_function_name']
            ).sum().reset_index()

            if nstructs == total_structs:
                struct_scores = ddg_scores.copy()

            ddg_scores = ddg_scores.drop(['struct_num', 'state'], axis=1, errors='ignore').groupby(
                ['case_name', 'backrub_steps', 'score_function_name']
            ).mean().round(decimals=5).reset_index()

            new_columns = list(ddg_scores.columns.values)
            ddg_scores = ddg_scores[new_columns]
            ddg_scores['scored_state'] = 'ddG'
            ddg_scores['nstruct'] = nstructs
            all_ddg_scores.append(ddg_scores)

        return pd.concat(all_ddg_scores), struct_scores

    def calc_dgs(self, scores):
        """
        Calculates the dG for mutant and wild-type states by combining
        bound and unbound states, then averaging over a range of structure counts.
        Returns a list of DataFrames, each corresponding to mutant or wild-type dG.
        """
        l = []
        total_structs = np.max(scores['struct_num'])
        nstructs_to_analyze = set([total_structs])
        for x in range(10, total_structs):
            if x % 10 == 0:
                nstructs_to_analyze.add(x)
        nstructs_to_analyze = sorted(nstructs_to_analyze)

        for state in ['mut', 'wt']:
            for nstructs in nstructs_to_analyze:
                dg_scores = scores.loc[
                    (scores['state'].str.endswith(state)) &
                    (scores['state'].str.startswith('unbound')) &
                    (scores['struct_num'] <= nstructs)
                ].copy()

                for column in dg_scores.columns:
                    if column not in ['state', 'case_name', 'backrub_steps', 'struct_num', 'score_function_name']:
                        dg_scores.loc[:, column] *= -1.0

                dg_scores = pd.concat(
                    [
                        dg_scores,
                        scores.loc[
                            (scores['state'].str.endswith(state)) &
                            (scores['state'].str.startswith('bound')) &
                            (scores['struct_num'] <= nstructs)
                        ].copy()
                    ]
                )

                dg_scores = dg_scores.drop('state', axis=1).groupby(
                    ['case_name', 'backrub_steps', 'struct_num', 'score_function_name']
                ).sum().reset_index()

                dg_scores = dg_scores.groupby(
                    ['case_name', 'backrub_steps', 'score_function_name']
                ).mean().round(decimals=5).reset_index()

                new_columns = list(dg_scores.columns.values)
                new_columns.remove('struct_num')
                dg_scores = dg_scores[new_columns]
                dg_scores['scored_state'] = state + '_dG'
                dg_scores['nstruct'] = nstructs
                l.append(dg_scores)

        return l
    def analyze_job(self, job_info):
        """Process a single mutation case"""
        job_path, struct_paths = job_info
        case_name = os.path.basename(job_path)

        # Process all structure files
        struct_scores = []
        for struct_path in struct_paths:
            try:
                struct_num = int(os.path.basename(struct_path))
                db3_path = os.path.join(struct_path, OUTPUT_DB_FILE)
                scores = self.process_db3_file(db3_path, struct_num, case_name)
                struct_scores.append(scores)
            except Exception as e:
                print(f"Error processing {struct_path}: {e}")
                continue

        if not struct_scores:
            return None, None, job_path

        # Combine scores and calculate ddG/dG
        combined_scores = pd.concat(struct_scores)
        ddg_scores, struct_level = self.calc_ddg(combined_scores)
        ddg_with_gam = apply_zemu_gam(ddg_scores)
        dg_scores = self.calc_dgs(combined_scores)

        return [struct_level], [ddg_scores, ddg_with_gam] + dg_scores, job_path

    def analyze(self):
        """Main analysis method"""
        jobs = self.find_finished_jobs()
        if not jobs:
            print('No finished jobs found')
            return

        # Process jobs in parallel
        with multiprocessing.Pool(self.process_count) as pool:
            results = list(tqdm(
                pool.imap(self.analyze_job, jobs.items()),
                total=len(jobs)
            ))

        # Combine results
        struct_scores = []
        ddg_scores = []

        for struct_data, ddg_data, job_path in results:
            if struct_data is None:
                print(f"No valid scores for {job_path}")
                continue
            struct_scores.extend(struct_data)
            ddg_scores.extend(ddg_data)

        # Save results
        basename = os.path.basename(self.output_folder)
        output_dir = os.path.dirname(self.output_folder)
        if struct_scores:
            pd.concat(struct_scores).to_csv(
                os.path.join(output_dir, f'{basename}-struct_scores_results.csv'),
                index=False
            )
        if ddg_scores:
            results_df = pd.concat(ddg_scores)
            results_df.to_csv(
                os.path.join(output_dir, f'{basename}-results.csv'),
                index=False
            )
            self.display_results(results_df)

    def display_results(self, df):
        """Display summary of results"""
        columns = [
            'backrub_steps', 'case_name', 'nstruct',
            'score_function_name', 'scored_state', 'total_score'
        ]
        for score_type in ['mut_dG', 'wt_dG', 'ddG']:
            print(f"\n{score_type}")
            print(df[df['scored_state'] == score_type][columns].head(20))

def main():
    args = parse_args()
    print("Loading config...")
    config = load_config(args.config)

    # Create output directories
    print("Creating output directories...")
    if not os.path.exists(args.analyze):
        os.makedirs(args.analyze)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Analyze Rosetta output if needed
    results_file = os.path.join(args.analyze, f"{args.output_dir_name}-results.csv")
    # if not os.path.exists(results_file):
    print("Analyzing Rosetta output...")
    analyzer = RosettaOutputAnalyzer(os.path.join(args.analyze, args.output_dir_name), args.processes)
    analyzer.analyze()

    # Process results
    print("Processing Rosetta results...")
    results_df = pd.read_csv(results_file, index_col=0)
    ddg_scores = (results_df[results_df["scored_state"]=="ddG"]
                 .groupby("case_name")["total_score"]
                 .min()
                 .sort_index())

    wildtype_sequence = config["wildtype_sequence"]
    wildtype_sequence = {k: v for k, v in wildtype_sequence.items() if k in config["mutable_chain"]}
    # Prepare training data if needed
    training_file = os.path.join(args.output_dir, "training_data_subset.csv")
    if not os.path.exists(training_file):
        print("Preparing training data...")
        mutations_df = pd.read_csv(args.mutations_file)
        assert len(ddg_scores) == len(mutations_df), "Number of ddG scores does not match mutations"

        mutations_df["ddG"] = ddg_scores
        chains = list(config["mutable_chain"])
        mutable_chain = config["mutable_chain"]
        assert len(chains) == 1, "Only one chain is supported"
        # Process mutations
        print("Processing mutations...")
        mutant_dict_list = []
        mutated_sequence_list = []
        mutant_str_dict_list = []
        for idx, row in tqdm(mutations_df.iterrows(), total=len(mutations_df), desc="Processing mutations"):
            mutant_dict = {chain: [] for chain in chains}
            mutations = row["mutations"]
            if pd.isna(mutations):
                mutations = []
            else:
                mutations = mutations.split(",")
            for mutant in mutations:
                mutant_dict[mutable_chain].append(f"{mutant[0]}{mutant[1:-1]}{mutant[-1]}")
            mutant_str_dict = {}
            mutated_sequence = {}
            for chain in chains:
                seq = wildtype_sequence[chain]
                mutated_sequence[chain] = mutations_to_seq(mutant_dict[chain], seq, offset=1)
                mutant_str_dict[chain] = ":".join(mutant_dict[chain])
            mutant_dict_list.append(mutant_dict)
            mutated_sequence_list.append(mutated_sequence)
            mutant_str_dict_list.append(mutant_str_dict)

        # Add metadata
        print("Adding metadata...")
        mutations_df = mutations_df.assign(
            POI=config["POI"],
            DMS_score=-mutations_df["ddG"],
            mutant=mutant_str_dict_list,
            wildtype_sequence=[wildtype_sequence.copy() for _ in range(len(mutations_df))],
            mutated_sequence=mutated_sequence_list,
            chain_id=config["mutable_chain"],
            pdb_file=config["POI"],
            cycle=args.cycle
        )
        mutations_df = mutations_df[["POI", "DMS_score", "mutant", "wildtype_sequence", "mutated_sequence", "chain_id", "pdb_file", "cycle", "mutseq", "mutations"]]
        if config.get("model_type", "sequence") in ["ablang2", "ablang_gp", "blosum_gp"]:
            mutations_df["wildtype_heavy"] = mutations_df["wildtype_sequence"].apply(lambda x: x[config["mutable_chain"]])
            mutations_df["wildtype_light"] = config["light_wt_sequence"]
            mutations_df["heavy"] = mutations_df["mutated_sequence"].apply(lambda x: x[config["mutable_chain"]])
            mutations_df["light"] = config["light_wt_sequence"]

        # Save training data
        print(f"Saving training data to {training_file}")
        mutations_df.to_csv(training_file, index=False)

    # Combine with previous data if available
    print("Combining training data...")
    all_data = pd.read_csv(training_file)
    if args.previous_data:
        print(f"Loading and merging previous data from {args.previous_data}")
        previous_data = pd.read_csv(args.previous_data)
        all_data = pd.concat([previous_data, all_data])

    output_file = os.path.join(args.output_dir, f"training_data.csv")
    print(f"Saving combined training data to {output_file}")
    all_data.to_csv(output_file, index=False)
    print("Done!")

if __name__ == '__main__':
    main()
