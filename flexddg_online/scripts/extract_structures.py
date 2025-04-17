#!/usr/bin/env python3

import os
import sys
import subprocess
import re
import shutil
import datetime
import math
import collections
import threading
import argparse
import fnmatch
from tqdm import tqdm

use_multiprocessing = False
if use_multiprocessing:
    import multiprocessing

def ts(td):
    """Convert timedelta to seconds"""
    return (td.microseconds + (td.seconds + td.days * 24 * 3600) * 1e6) / 1e6

def mean(values):
    """Calculate mean of list of values"""
    return float(sum(values)) / float(len(values))

class Reporter:
    """Progress reporter for long-running tasks"""

    def __init__(self, task, entries='files', print_output=True, eol_char='\r'):
        self._lock = threading.Lock()
        self.print_output = print_output
        self.start = datetime.datetime.now()
        self.entries = entries
        self.lastreport = self.start
        self.task = task
        self.report_interval = datetime.timedelta(seconds=1)
        self.n = 0
        self.completion_time = None
        self.total_count = None
        self.maximum_output_string_length = 0
        self.rolling_est_total_time = collections.deque(maxlen=50)
        self.kv_callback_results = {}
        self.list_results = []
        self.eol_char = eol_char

        if self.print_output:
            print('\nStarting ' + task)

    def set_total_count(self, count):
        self.total_count = count
        self.rolling_est_total_time = collections.deque(maxlen=max(1, int(0.05 * count)))

    def decrement_total_count(self):
        if self.total_count:
            self.total_count -= 1

    def _format_output(self, n, time_now):
        if self.total_count:
            percent_done = float(n) / float(self.total_count)
            est_total_time_seconds = ts(time_now - self.start) * (1.0 / percent_done)
            self.rolling_est_total_time.append(est_total_time_seconds)
            est_total_time = datetime.timedelta(seconds=mean(self.rolling_est_total_time))
            time_remaining = est_total_time - (time_now - self.start)
            eta = time_now + time_remaining
            time_remaining_str = f'ETA: {eta.strftime("%Y-%m-%d %H:%M:%S")} Est. time remaining: '
            time_remaining_str += str(datetime.timedelta(seconds=int(ts(time_remaining))))
            return f"  Processed: {n} {self.entries} ({percent_done*100.0:.1f}%) {time_remaining_str}"
        return f"  Processed: {n} {self.entries}"

    def report(self, n):
        with self._lock:
            self.n = n
            time_now = datetime.datetime.now()
            if self.print_output and self.lastreport < (time_now - self.report_interval):
                self.lastreport = time_now
                output_string = self._format_output(n, time_now) + self.eol_char

                self.maximum_output_string_length = max(len(output_string), self.maximum_output_string_length)
                if len(output_string) < self.maximum_output_string_length:
                    output_string = output_string.ljust(self.maximum_output_string_length)

                sys.stdout.write(output_string)
                sys.stdout.flush()

    def increment_report(self):
        self.report(self.n + 1)

    def increment_report_callback(self, cb_value):
        self.increment_report()

    def increment_report_keyval_callback(self, kv_pair):
        key, value = kv_pair
        self.kv_callback_results[key] = value
        self.increment_report()

    def increment_report_list_callback(self, new_list_items):
        self.list_results.extend(new_list_items)
        self.increment_report()

    def decrement_report(self):
        self.report(self.n - 1)

    def add_to_report(self, x):
        self.report(self.n + x)

    def done(self):
        self.completion_time = datetime.datetime.now()
        if self.print_output:
            duration = self.completion_time - self.start
            print(f'Done {self.task}, processed {self.n} {self.entries}, took {duration}\n')

    def elapsed_time(self):
        if self.completion_time:
            return self.completion_time - self.start
        return time.time() - self.start


struct_db3_file = 'struct.db3'
trajectory_stride = 5

def recursive_find_struct_dbs(input_dir):
    """Recursively find all struct.db3 files in directory"""
    return_list = []
    for path in [os.path.join(input_dir, x) for x in os.listdir(input_dir)]:
        if os.path.isdir(path):
            return_list.extend(recursive_find_struct_dbs(path))
        elif os.path.isfile(path) and os.path.basename(path) == struct_db3_file:
            return_list.append(path)
    return return_list

def extract_structures(struct_db, rename_function=None, pattern=None):
    """Extract structures from database file"""
    working_directory = os.path.dirname(struct_db)

    if rename_function:
        expected_output = os.path.join(working_directory, rename_function(1))
        if os.path.exists(expected_output):
            print(f"Output {expected_output} already exists, skipping...")
            return 0

    args = [
        os.path.expanduser('score_jd2.default.linuxgccrelease'),
        '-inout:dbms:database_name', struct_db3_file,
        '-in:use_database',
        '-out:pdb',
    ]

    rosetta_outfile_path = os.path.join(working_directory, 'structure_output.txt')
    if not use_multiprocessing:
        print(rosetta_outfile_path)
        print(' '.join(args))

    with open(rosetta_outfile_path, 'w') as rosetta_outfile:
        rosetta_process = subprocess.Popen(
            ' '.join(args),
            stdout=rosetta_outfile,
            stderr=subprocess.STDOUT,
            close_fds=True,
            cwd=working_directory,
            shell=True,
        )
        return_code = rosetta_process.wait()

    if return_code == 0:
        os.remove(rosetta_outfile_path)

    if rename_function:
        for path in [os.path.join(working_directory, x) for x in os.listdir(working_directory)]:
            if match := re.match(r'(\d+)_0001.pdb', os.path.basename(path)):
                struct_id = int(match.group(1))
                new_name = rename_function(struct_id)
                if pattern and not fnmatch.fnmatch(new_name, pattern):
                    os.remove(path)
                    continue
                dest_path = os.path.join(working_directory, new_name)
                shutil.move(path, dest_path)

    return return_code

def flex_ddG_rename(struct_id):
    """Generate output filename based on structure ID"""
    steps = ['backrub', 'wt', 'mut']
    step = steps[(struct_id-1) % len(steps)]
    frame = (((struct_id-1) // len(steps)) + 1) * trajectory_stride
    return f'{step}_{frame:05d}.pdb'

def main(input_dir, pattern=None):
    """Main function to extract structures from all databases"""
    struct_dbs = recursive_find_struct_dbs(input_dir)
    print(f'Found {len(struct_dbs)} structure database files to extract')

    if use_multiprocessing:
        pool = multiprocessing.Pool()

    reporter = Reporter('extracting structure database files', entries='.db3 files')
    reporter.set_total_count(len(struct_dbs))

    for struct_db in struct_dbs:
        if use_multiprocessing:
            pool.apply_async(
                extract_structures,
                args=(struct_db,),
                kwds={'rename_function': flex_ddG_rename, 'pattern': pattern},
                callback=reporter.increment_report_callback
            )
        else:
            reporter.increment_report_callback(
                extract_structures(struct_db, rename_function=flex_ddG_rename, pattern=pattern)
            )

    if use_multiprocessing:
        pool.close()
        pool.join()
    reporter.done()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dirs', nargs='+', help='Input directories containing struct.db3 files')
    parser.add_argument('--pattern', type=str, help='Pattern to match output filenames (e.g. mut_*, mut_00025.pdb)')
    args = parser.parse_args()

    for path in args.input_dirs:
        if os.path.isdir(path):
            main(path, pattern=args.pattern)
        else:
            print(f'ERROR: {path} is not a valid directory')
