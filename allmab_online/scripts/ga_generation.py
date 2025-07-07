from deap import algorithms
from deap import base
from deap import tools
import random
import numpy as np
import multiprocessing
import argparse
from multiprocessing import Pool
import torch
import os
import pandas as pd
from tqdm import tqdm

from logits import esm2_logit_inference
from peft import get_peft_model, LoraConfig
from utils import load_config
from esm import ESM2, Alphabet, pretrained

TOK_NUM = 20
ESM2_TOKEN_OFFSET = 4

esm2_standard_aa_tokens = list(range(ESM2_TOKEN_OFFSET, ESM2_TOKEN_OFFSET+TOK_NUM))

# アミノ酸の一文字コードリスト
AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
              'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S',
              'T', 'V', 'W', 'Y']

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


def custom_population(n, toolbox):
    """カスタム集団生成: 最初の個体に初期配列を使用"""
    pop = toolbox.population(n=n-1)
    pop.insert(0, toolbox.initial_individual())
    return pop

def init_initial(icls, initial_seq):
    return icls(list(initial_seq))

def setup_toolbox(target_length, initial_seq = None, batch_mapper = None):
    from deap import creator
    if 'Fitness' in creator.__dict__:
        del creator.Fitness
    if 'Individual' in creator.__dict__:
        del creator.Individual

    creator.create("Fitness", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox = base.Toolbox()

    toolbox.register("aa_gene", random.choice, AMINO_ACIDS)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                    toolbox.aa_gene, n=target_length)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


    toolbox.register("map", batch_mapper)
    toolbox.register("evaluate", lambda x: None)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate_aa, indpb=0.05)
    toolbox.register("select", tools.selNSGA2)

    if initial_seq is not None:
        toolbox.register("initial_individual", init_initial, creator.Individual, initial_seq)

    return toolbox

def mutate_aa(individual, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = random.choice(AMINO_ACIDS)
    return individual,

def predict_score_esm(sequences):
    scores = esm2_logit_inference(sequences, None, alphabet, wt_seq, masked_wt_seq, mask_indices, device="cuda", mask_logits=mask_logits)
    return scores

def evaluate_batch(individuals_batch):
    batch_sequences = [ "".join(individual) for individual in individuals_batch]
    batch_full_sequences = []
    for seq in batch_sequences:
        mtseq = "".join([wt_seq[i] if i not in mask_indices else seq[mask_indices.index(i)] for i in range(len(wt_seq))])
        batch_full_sequences.append(mtseq)
    scores = predict_score_esm(batch_full_sequences)
    scores = [score if score not in exisiting_mutseqs else -1e9 for score in scores]
    return [(score,) for score in scores]

def batch_mapper(toolbox, population, batch_size=1000):
    batches = [population[i:i+batch_size] for i in range(0, len(population), batch_size)]

    all_fitness = []
    for batch in batches:
        batch_fitness = evaluate_batch(batch)
        all_fitness.extend(batch_fitness)

    return all_fitness

def run_single_ga(random_seed: int,
                  wild_type_seq: str,
                  population_size: int,
                  lambda_: int,
                  generations: int,
                  cxpb: float,
                  mutpb: float):

    random.seed(random_seed)

    # Set a default sequence length if wild_type_seq is None
    target_length = len(wild_type_seq) if wild_type_seq is not None else 10

    toolbox = setup_toolbox(target_length, wild_type_seq, batch_mapper)


    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    if wild_type_seq is not None:
        pop = custom_population(population_size, toolbox)
    else:
        pop = toolbox.population(n=population_size)

    hof = tools.HallOfFame(1)

    pop, log = algorithms.eaMuPlusLambda(
        pop, toolbox,
        mu=population_size,
        lambda_=lambda_,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=generations,
        stats=stats,
        halloffame=hof,
        verbose=False
    )

    return hof

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--training_data", type=str, default=None)
    parser.add_argument("--parallel_num", type=int, default=4)
    args = parser.parse_args()
    config = load_config(args.config)

    if os.path.exists(args.output_file):
        print(f"Output file {args.output_file} already exists. Skipping generation.")
        exit()

    model_type = config.get("model_type", "sequence")
    assert model_type =="sequence", "model_type must be sequence"

    wt_seq = config["wildtype_sequence"][config["mutable_chain"]]
    mask_indices = config["mask_indices"]
    masked_wt_seq = "".join([wt_seq[i] if i not in mask_indices else "<mask>" for i in range(len(wt_seq))])

    wild_type_seq = "".join([wt_seq[i] for i in mask_indices])
    exisiting_mutseqs = []
    if args.training_data is not None:
        print(f"Loading training data from {args.training_data}")
        train_df = pd.read_csv(args.training_data)
        exisiting_mutseqs += list(train_df["mutseq"].unique())
    else:
        train_df = None


    n_runs = config.acquisition["acquisition_num"]
    ga_generation = config.ga_generation if config.get("ga_generation") is not None else {}
    population_size = ga_generation.get("population_size", 30)
    lambda_ = ga_generation.get("lambda_", 10)
    generations = ga_generation.get("generations", 100)
    cxpb = ga_generation.get("cxpb", 0.7)
    mutpb = ga_generation.get("mutpb", 0.1)

    model, alphabet = pretrained.load_model_and_alphabet('esm2_t33_650M_UR50D')

    if args.model_path is not None:
        print(f"Loading model from {args.model_path}")
        if config.get("lora_config") is not None:
            lora_config = config["lora_config"]
            peft_config = LoraConfig(**lora_config)
            model = get_peft_model(model, peft_config)
        state_dict = torch.load(args.model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    model.eval()
    model.cuda()

    masked_input = alphabet.get_batch_converter()([('',masked_wt_seq)])[2].to("cuda")
    logits = model(masked_input)["logits"].cpu()

    # Get logits at mask positions
    mask_logits = logits[0, np.array(mask_indices)+1, :]
    mask_logits = mask_logits[:, esm2_standard_aa_tokens]
    print("mask_logits.shape", mask_logits.shape)
    with Pool(processes=min(multiprocessing.cpu_count(), args.parallel_num)) as pool:
        results = pool.starmap(run_single_ga, [(random_seed, wild_type_seq, population_size, lambda_, generations, cxpb, mutpb) for random_seed in range(n_runs)])

    mutseqs = ["".join(result[0]) for result in results]
    predicted_full_seqs = []
    for mutseq in mutseqs:
        seq = list(wt_seq)
        for i, mut_aa in zip(mask_indices, mutseq):
            seq[i] = mut_aa
        predicted_full_seqs.append(''.join(seq))
    mutations = [mutseq_to_mut(seq, wt_seq, offset=1) for seq in predicted_full_seqs]
    mutations_str = [",".join(muts) for muts in mutations]
    mutations_df = pd.DataFrame({"mutseq": mutseqs, "mutations": mutations_str})
    mutations_df.to_csv(args.output_file, index=False)
