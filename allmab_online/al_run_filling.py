import os
import sys
import subprocess
import argparse
import time
import logging
import shutil
import pandas as pd
import threading
from scripts.utils import load_config

SLEEP_TIME = 10

class ActiveLearningPipeline:
    def __init__(self, config_path, script_dir="scripts",
                 batch_size=16, max_cycles=10):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.max_cycles = min(self.config.max_cycles, max_cycles)
        self.script_dir = script_dir
        self.batch_size = batch_size
        self.acquisition_num = self.config.acquisition.acquisition_num
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
        acquisition_script = "acquisition_multi_hv.py" if self.config.acquisition.get("is_multi_objective", False) else "acquisition.py"
        self.scripts = {
            "generation": os.path.join(self.script_dir, f"sampling.py"),
            "offline_inference": os.path.join(self.script_dir, offline_inference_script),
            "acquisition": os.path.join(self.script_dir, acquisition_script),
            "prepare_train": os.path.join(self.script_dir, "prepare_train_filling.py"),
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

    def get_mutations_csv_file(self, data_dir):
        return os.path.join(data_dir, "mutations_list.csv")


    def run_generation(self, cycle, data_dir, prev_data_dir):
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

        model_path = self.get_model_file(prev_data_dir) if prev_data_dir else None
        if self.online_generation:
            command_args = ["python3", self.scripts["generation"], "--target_dir", generation_dir,
                       "--config", self.config_path]
        else:
            command_args = ["python3", self.scripts["offline_inference"], "--target_dir", generation_dir,
                       "--config", self.config_path, "--pool_data", self.generation_file]
        if model_path:
            command_args.append("--model_path")
            command_args.append(model_path)
        if self.config.generation.get("bias", False):
            command_args.append("--bias")
        if self.config.generation.get("calculate_fitness", False):
            command_args.append("--calculate_fitness")
        self.execute_in_shell(command_args)

        if not os.path.exists(generation_file):
            raise Exception(f"Failed to generation for cycle {cycle}")
        logging.info(f"Generation file {generation_file} created")

    def run_acquisition(self, cycle, data_dir, prev_data_dir):
        logging.info(f"Running acquisition for cycle {cycle}")
        mutations_file = self.get_acquisition_file(data_dir)
        if os.path.exists(mutations_file):
            logging.info(f"mutation file {mutations_file} already exists")
            return

        prev_data = self.get_train_data_file(prev_data_dir) if prev_data_dir else None
        acquisition_dir = os.path.join(data_dir, "acquisition")
        if not os.path.exists(acquisition_dir):
            os.makedirs(acquisition_dir, exist_ok=True)
        command_args = [
            "python3", self.scripts["acquisition"],
            "--config", self.config_path,
            "--pool_data", self.get_generation_file(data_dir),
            "--output_file", mutations_file
        ]
        if prev_data:
            command_args.extend([
                "--training_data",
                prev_data
            ])
        self.execute_in_shell(command_args)

        if not os.path.exists(mutations_file):
            raise Exception(f"Failed to acquisition for cycle {cycle}")
        logging.info(f"Acquisition file {mutations_file} created")

    def create_mutations_csv(self, cycle, data_dir):
        logging.info(f"Creating mutations CSV for cycle {cycle}")
        mutations_file = self.get_acquisition_file(data_dir)
        mutations_csv_file = self.get_mutations_csv_file(data_dir)

        if os.path.exists(mutations_csv_file):
            logging.info(f"Mutations CSV file {mutations_csv_file} already exists")
            return

        mutations_df = pd.read_csv(mutations_file)
        mutations_df['DMS_score'] = None
        mutations_df['updated'] = False
        mutations_df.to_csv(mutations_csv_file, index=False)
        logging.info(f"Mutations CSV file {mutations_csv_file} created")

    def check_all_mutations_updated(self, data_dir):
        mutations_csv_file = self.get_mutations_csv_file(data_dir)
        if not os.path.exists(mutations_csv_file):
            return False

        df = pd.read_csv(mutations_csv_file)
        return df['updated'].all()


    def run_prepare_training_data(self, cycle, data_dir, prev_data_dir):
        training_data_file = self.get_train_data_file(data_dir)
        if os.path.exists(training_data_file):
            logging.info(f"Training data file {training_data_file} already exists")
            return

        logging.info(f"Running prepare training data for cycle {cycle}")
        mutations_csv_file = self.get_mutations_csv_file(data_dir)
        train_data_dir = self.get_train_data_dir(data_dir)
        prev_data = self.get_train_data_file(prev_data_dir) if prev_data_dir else None
        os.makedirs(train_data_dir, exist_ok=True)

        command_args = [
            "python3", self.scripts["prepare_train"],
            "--mutations_file", mutations_csv_file,
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
        data_dir = os.path.join(self.config.data_dir, str(cycle))
        prev_data_dir = os.path.join(self.config.data_dir, str(cycle-1)) if cycle > 0 else None

        os.makedirs(data_dir, exist_ok=True)

        # Run pipeline steps
        # 1. Mutant Generation
        self.run_generation(cycle, data_dir, prev_data_dir)
        # 2. Acquisition
        self.run_acquisition(cycle, data_dir, prev_data_dir)

        # 3. Create mutations CSV with updated column
        self.create_mutations_csv(cycle, data_dir)

        # 4. Wait for all mutations to be updated
        while not self.check_all_mutations_updated(data_dir):
            logging.info(f"Waiting for all mutations to be updated for cycle {cycle}")
            time.sleep(SLEEP_TIME)

        logging.info(f"All mutations updated for cycle {cycle}")

        # 5. Analyze and prepare training data
        self.run_prepare_training_data(cycle, data_dir, prev_data_dir)

        # 6. Train model
        self.run_training(cycle, data_dir)

        logging.info(f"Cycle {cycle} finished")

    def run(self):
        logging.info("Starting active learning filling workflow")
        for cycle in range(self.max_cycles):
            self.run_cycle(cycle)

def main():
    parser = argparse.ArgumentParser(description='Run active learning filling pipeline')
    parser.add_argument("config", type=str, help="config file")
    parser.add_argument("--script_dir", type=str, default="scripts")
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--cycles", "-c", type=int, default=10)
    args = parser.parse_args()

    pipeline = ActiveLearningPipeline(
        args.config,
        script_dir=args.script_dir,
        batch_size=args.batch_size,
        max_cycles=args.cycles
    )
    pipeline.run()

    logging.info("Active learning workflow finished")

if __name__ == '__main__':
    main()
