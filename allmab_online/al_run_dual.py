import os
import sys
import json
import subprocess
import multiprocessing
import argparse
from glob import glob
import time
import logging
import shutil
import pandas as pd
import threading
import numpy as np
from scripts.utils import load_config

SLEEP_TIME = 180

class ActiveLearningPipeline:
    def __init__(self, config_path, script_dir="scripts",
                 flex_job_name="flex_ddG", parallel_num=4, priority="-5",
                 batch_size=16, use_shell=False, max_cycles=10, flx_cpus=80):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.max_cycles = min(self.config.max_cycles, max_cycles)
        self.script_dir = script_dir
        self.flex_job_name = flex_job_name
        self.parallel_num = parallel_num
        self.priority = priority
        self.max_wait_time = 60*60*24
        self.batch_size = batch_size
        self.use_shell = use_shell
        self.flx_node = "cpu_40=1"
        self.flx_runtime = "4:30:00"
        self.flx_cpus = flx_cpus
        self.acquisition_num = self.config.acquisition.acquisition_num
        self.flx_num = self.config.flx_num
        self.online_generation = self.config.generation.get("online_generation", True)
        if not self.online_generation:
            if not os.path.exists(self.config.generation.get("generation_file", None)):
                raise Exception("Generation file does not exist")
            self.generation_file = self.config.generation.get("generation_file", None)
        else:
            self.generation_file = None
        self._setup_logging()
        self._setup_scripts()

    def _setup_logging(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

    def _setup_scripts(self):
        offline_inference_script = "offline_inference_gp.py" if self.config.get("model_type", "sequence") in ["ablang_gp", "blosum_gp"] else "offline_inference.py"
        train_script = "train_gp.py" if self.config.get("model_type", "sequence") in ["ablang_gp", "blosum_gp"] else "train_al.py"
        self.scripts = {
            "generation": os.path.join(self.script_dir, f"sampling_dual.py"),
            "offline_inference": os.path.join(self.script_dir, offline_inference_script),
            "acquisition": os.path.join(self.script_dir, "acquisition_multi_hv_dual.py"),
            "prep_flex_ddG": os.path.join(self.script_dir, "prepare_flexddg.py"),
            "flex_ddG": os.path.join(self.script_dir, "run_flex.sh"),
            "prepare_train": os.path.join(self.script_dir, "prepare_train.py"),
            "train": os.path.join(self.script_dir, train_script),
        }

    def execute_in_shell(self, command):
        command = [str(c) for c in command]
        logging.info("Executing command: {}".format(" ".join(command)))
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

        def print_output(pipe, prefix=''):
            for line in iter(pipe.readline, ''):
                logging.info(f"{prefix}{line.rstrip()}")
                sys.stdout.flush()

        stdout_thread = threading.Thread(target=print_output, args=(process.stdout,))
        stderr_thread = threading.Thread(target=print_output, args=(process.stderr, "stderr: "))

        stdout_thread.daemon = True
        stderr_thread.daemon = True

        stdout_thread.start()
        stderr_thread.start()

        process.wait()
        stdout_thread.join()
        stderr_thread.join()

        if process.returncode != 0:
            logging.error(f"Command failed with return code {process.returncode}: {' '.join(command)}")
            return process.returncode

        return 0

    def get_generation_file(self, data_dir):
        generation_dir = os.path.join(data_dir, "generation")
        return os.path.join(generation_dir, "generation_result.csv")

    def get_acquisition_file(self, data_dir):
        acquisition_dir = os.path.join(data_dir, "acquisition")
        return os.path.join(acquisition_dir, "acquisition_result.csv")

    def get_flex_ddG_dir(self, data_dir):
        return os.path.join(data_dir, "flex_ddG")

    def get_train_data_dir(self, data_dir):
        return os.path.join(data_dir, "train_data")

    def get_train_data_file(self, data_dir):
        return os.path.join(self.get_train_data_dir(data_dir), "training_data.csv")

    def get_model_file(self, data_dir):
        if self.config.get("model_type", "sequence") in ["ablang_gp", "blosum_gp"]:
            return os.path.join(self.get_train_data_dir(data_dir), "gp_model.pkl")
        elif self.config.get("model_type", "sequence") in ["ablang2", "sequence"]:
            return os.path.join(self.get_train_data_dir(data_dir), "model.pt")
        else:
            raise ValueError(f"Invalid model type: {self.config.get('model_type', 'sequence')}")

    def count_flex_ddG_scores(self, data_dir):
        return len(glob(f"{data_dir}/flex_ddG/outputs/*/*/score.sc"))

    def run_generation(self, cycle, data_dir, prev_data_dir_targets = None):
        logging.info(f"Running generation for cycle {cycle}")
        generation_file = self.get_generation_file(data_dir)
        if os.path.exists(generation_file):
            logging.info(f"generation file {generation_file} already exists")
            return
        generation_dir = os.path.join(data_dir, "generation")
        if not os.path.exists(generation_dir):
            os.makedirs(generation_dir, exist_ok=True)

        if cycle == 0 and not self.online_generation:
            logging.info(f"Copying generation file {self.generation_file} to {generation_file} for initial generation")
            shutil.copy(self.generation_file, generation_file)
            return

        model_paths = [self.get_model_file(prev_data_dir_target) for prev_data_dir_target in prev_data_dir_targets if prev_data_dir_target is not None]
        if self.online_generation:
            command_args = ["python3", self.scripts["generation"], "--target_dir", generation_dir,
                       "--config", self.config_path]
        else:
            command_args = ["python3", self.scripts["offline_inference"], "--target_dir", generation_dir,
                       "--config", self.config_path, "--pool_data", self.generation_file]
        if len(model_paths) > 0:
            command_args.append("--model_paths")
            for model_path in model_paths:
                command_args.append(model_path)
        if self.config.generation.get("bias", False):
            command_args.append("--bias")
        self.execute_in_shell(command_args)

        if not os.path.exists(generation_file):
            raise Exception(f"Failed to generation for cycle {cycle}")
        logging.info(f"Generation file {generation_file} created")

    def run_acquisition(self, cycle, data_dir, prev_data_dir_targets):
        logging.info(f"Running acquisition for cycle {cycle}")
        mutations_file = self.get_acquisition_file(data_dir)
        if os.path.exists(mutations_file):
            logging.info(f"mutation file {mutations_file} already exists")
            return

        prev_data_files = [self.get_train_data_file(prev_data_dir_target) for prev_data_dir_target in prev_data_dir_targets if prev_data_dir_target is not None]
        acquisition_dir = os.path.join(data_dir, "acquisition")
        if not os.path.exists(acquisition_dir):
            os.makedirs(acquisition_dir, exist_ok=True)
        command_args = [
            "python3", self.scripts["acquisition"],
            "--config", self.config_path,
            "--pool_data", self.get_generation_file(data_dir),
            "--output_file", mutations_file
        ]
        if len(prev_data_files) > 0:
            command_args.extend([
                "--training_datas",
                *prev_data_files
            ])
        self.execute_in_shell(command_args)

        if not os.path.exists(mutations_file):
            raise Exception(f"Failed to acquisition for cycle {cycle}")
        logging.info(f"Acquisition file {mutations_file} created")

    def run_flex_ddg(self, cycle, data_dir, target_idx):
        pdb_file = self.config.pdb_files[target_idx]
        logging.info(f"Running flex ddG for cycle {cycle}")
        mutations_file = self.get_acquisition_file(data_dir)
        flex_ddG_dir = self.get_flex_ddG_dir(data_dir)
        if self.count_flex_ddG_scores(data_dir) >= self.acquisition_num * self.flx_num:
            logging.info(f"Flex ddG dir {flex_ddG_dir} already has enough scores")
            return

        if not os.path.exists(flex_ddG_dir):
            os.makedirs(flex_ddG_dir, exist_ok=True)
            os.makedirs(os.path.join(flex_ddG_dir, "inputs"), exist_ok=True)
            os.makedirs(os.path.join(flex_ddG_dir, "outputs"), exist_ok=True)
        if len(glob(os.path.join(flex_ddG_dir, "inputs/*/nataa_mutations.resfile"))) != self.acquisition_num:
            self.execute_in_shell([
                "python3", self.scripts["prep_flex_ddG"],
                "--mutations_file", mutations_file,
                "--pdbfile", pdb_file,
                "--chains_to_move", self.config.mutable_chain,
                "--mutations_col", self.config.mutations_col,
                "--output_dir", os.path.join(flex_ddG_dir, "inputs")
            ])

        if len(glob(os.path.join(flex_ddG_dir, "inputs/*/nataa_mutations.resfile"))) != self.acquisition_num:
            raise Exception(f"Failed to prepare flex ddG for cycle {cycle}")

        fle_ddG_dir_abs = os.path.abspath(flex_ddG_dir)
        scores_count = self.count_flex_ddG_scores(data_dir)
        if scores_count >= self.acquisition_num * self.flx_num:
            logging.info(f"Flex ddG dir {flex_ddG_dir} already has enough scores")
            return

        if self.use_shell:
            # Direct shell execution mode
            offset_num = self.acquisition_num # in shell mode, we run all the jobs at once
            command = [
                "bash",
                self.scripts['flex_ddG'],
                fle_ddG_dir_abs,
                str(self.flx_num),
                str(offset_num),
                str(self.flx_cpus)
            ]
            self.execute_in_shell(command)
        else:
            raise Exception("Job submission mode is not implemented yet")
        logging.info(f"Flex ddG finished for cycle {cycle}")

    def run_prepare_training_data(self, cycle, data_dir, prev_data_dir):
        training_data_file = self.get_train_data_file(data_dir)
        if os.path.exists(training_data_file):
            logging.info(f"Training data file {training_data_file} already exists")
            return

        logging.info(f"Running prepare training data for cycle {cycle}")
        mutations_file = self.get_acquisition_file(data_dir)
        flex_ddG_dir = self.get_flex_ddG_dir(data_dir)
        train_data_dir = self.get_train_data_dir(data_dir)
        prev_data = self.get_train_data_file(prev_data_dir) if prev_data_dir else None
        os.makedirs(train_data_dir, exist_ok=True)
        command_args = [
            "python3", self.scripts["prepare_train"],
            "--analyze", flex_ddG_dir,
            "--mutations_file", mutations_file,
            "--output_dir", train_data_dir,
            "--config", self.config_path,
            "--cycle", str(cycle)
        ]
        if prev_data is not None:
            command_args.append("--previous_data")
            command_args.append(prev_data)
        self.execute_in_shell(command_args)

        logging.info(f"Prepare training data finished for cycle {cycle}")

        if not os.path.exists(training_data_file):
            raise Exception(f"Failed to prepare training data for cycle {cycle}")

        return train_data_dir

    def run_training(self, cycle, data_dir):
        logging.info(f"Running train for cycle {cycle}")
        model_file = self.get_model_file(data_dir)
        if os.path.exists(model_file):
            logging.info(f"Model {model_file} already exists")
            return

        train_data_dir = self.get_train_data_dir(data_dir)
        train_data = self.get_train_data_file(data_dir)
        if self.config.get("model_type", "sequence") in ["ablang_gp", "blosum_gp"]:
            model_params = self.config.get("model_params", {})
            command_args = [
                "python3", self.scripts["train"],
                "--dms_input", train_data,
                "--model_type", self.config.get("model_type", "sequence"),
                "--output_dir", train_data_dir,
                "--seed", str(self.config.get("seed", 42)),
                "--kernel", str(model_params.get("kernel", "rbf")),
            ]
        else:
            command_args = [
                "python3", self.scripts["train"],
                "--dms_input", train_data,
                "--model_type", self.config.get("model_type", "sequence"),
                "--output_dir", train_data_dir,
                "--batch_size", str(self.batch_size),
                "--seed", str(self.config.get("seed", 42)),
                "--early_stop_patience", str(self.config.get("early_stop_patience", 20)),
                "--n_steps", str(self.config.get("n_steps", 1000)),
            ]
        if self.config.get("test_input", None):
            command_args.append("--test_input")
            command_args.append(self.config.get("test_input", None))

        self.execute_in_shell(command_args)
        if not os.path.exists(model_file):
            raise Exception(f"Failed to train model for cycle {cycle}")
        logging.info(f"Train finished for cycle {cycle}")

    def run_cycle(self, cycle):
        logging.info(f"Starting cycle {cycle}")
        base_data_dir = os.path.join(self.config.data_dir, str(cycle), "target_" + str(0))
        generation_dir_base = os.path.join(base_data_dir, "generation")
        acquisition_dir_base = os.path.join(base_data_dir, "acquisition")
        os.makedirs(generation_dir_base, exist_ok=True)
        os.makedirs(acquisition_dir_base, exist_ok=True)
        data_dir_targets = []
        prev_data_dir_targets = []
        for target_idx, _ in enumerate(self.config.pdb_files):
            data_dir = os.path.join(self.config.data_dir, str(cycle), "target_" + str(target_idx))
            prev_data_dir = os.path.join(self.config.data_dir, str(cycle-1), "target_" + str(target_idx)) if cycle > 0 else None
            acquisition_dir = os.path.join(data_dir, "acquisition")
            generation_dir = os.path.join(data_dir, "generation")
            # acquisition, generation -> share directory
            os.makedirs(data_dir, exist_ok=True)
            if not os.path.lexists(acquisition_dir):
                os.symlink(os.path.abspath(acquisition_dir_base), acquisition_dir)
            if not os.path.lexists(generation_dir):
                os.symlink(os.path.abspath(generation_dir_base), generation_dir)
            data_dir_targets.append(data_dir)
            prev_data_dir_targets.append(prev_data_dir)
        # Run pipeline steps
        # 1. Mutant Generation
        self.run_generation(cycle, base_data_dir, prev_data_dir_targets)
        # 2. Acquisition
        self.run_acquisition(cycle, base_data_dir, prev_data_dir_targets)

        # 2. Execute Flex ddG
        with multiprocessing.Pool() as pool:
            flex_args = [(cycle, data_dir_target, target_idx)
                        for target_idx, data_dir_target in enumerate(data_dir_targets)]
            pool.starmap(self.run_flex_ddg, flex_args)

        # 3. Analyze Flex ddG and prepare training data in parallel
        with multiprocessing.Pool() as pool:
            prep_args = [(cycle, data_dir_target, prev_data_dir_target)
                        for data_dir_target, prev_data_dir_target in zip(data_dir_targets, prev_data_dir_targets)]
            pool.starmap(self.run_prepare_training_data, prep_args)

        # 4. Train model in parallel
        with multiprocessing.Pool() as pool:
            train_args = [(cycle, data_dir_target) for data_dir_target in data_dir_targets]
            pool.starmap(self.run_training, train_args)
        logging.info(f"Cycle {cycle} finished")

    def run(self):
        logging.info("Starting active learning workflow")
        for cycle in range(self.max_cycles):

            self.run_cycle(cycle)

def main():
    parser = argparse.ArgumentParser(description='Run active learning pipeline')
    parser.add_argument("config", type=str, help="config file")
    parser.add_argument("--script_dir", type=str, default="scripts")
    parser.add_argument("--flex_job_name", "-j", type=str, default="flex_ddG")
    parser.add_argument("--parallel_num", "-n", type=int, default=1)
    parser.add_argument("--priority", "-p", type=str, default="-5")
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--cycles", "-c", type=int, default=10)
    parser.add_argument("--flx_cpus", type=int, default=80)
    args = parser.parse_args()

    pipeline = ActiveLearningPipeline(
        args.config,
        script_dir=args.script_dir,
        flex_job_name=args.flex_job_name,
        parallel_num=args.parallel_num,
        priority=args.priority,
        batch_size=args.batch_size,
        use_shell=True,
        max_cycles=args.cycles,
        flx_cpus=args.flx_cpus
    )
    pipeline.run()

    logging.info("Active learning workflow finished")

if __name__ == '__main__':
    main()
