"""
    Test suite for the model_comparison_experiment: evaluates the SD1 20x10
    models trained by run_train.sh (seeds 0/1/2) with both the greedy and the
    sampling strategy against every dataset under ./data/data_test, and saves
    the results as per-(seed, dataset, mode) excel files.
"""
import os
import time

import pandas as pd

from params import configs
from data_utils import text_to_matrix
from test_trained_model import test_greedy_strategy, test_sampling_strategy

DATA_SOURCE = 'SD1'
N_J, N_M = 20, 10
TEST_ROOT = './data/data_test'

# optional tag to keep a run's models/results separate from the baseline
# (e.g. RUN_TAG=eligible -> trained_network/SD1/20x10+eligible_seed0.pth,
# ./eligible_experiment/seed0/...), matching train.py's
# --model_suffix "${RUN_TAG}_seed${SEED}" convention and the {method}_experiment
# layout expected by model_comparison_experiment/common.py
RUN_TAG = os.environ.get('RUN_TAG', '')
MODEL_SUFFIX = f'{RUN_TAG}_seed' if RUN_TAG else 'seed'
OUTPUT_ROOT = f'./{RUN_TAG}_experiment' if RUN_TAG else './model_comparison_experiment'
RESULT_PREFIX = 'seed'


def load_instances_with_names(directory):
    """
        load all .fjs instances within a directory, keeping their file names
    :param directory: the directory of instance files
    :return: job_length list, op_pt list, instance name list (same order)
    """
    files = sorted(f for f in os.listdir(directory) if f.endswith('.fjs'))

    job_lengths, op_pts, names = [], [], []
    for f in files:
        with open(os.path.join(directory, f), 'r') as fh:
            lines = fh.readlines()
        job_length, op_pt = text_to_matrix(lines)
        job_lengths.append(job_length)
        op_pts.append(op_pt)
        names.append(os.path.splitext(f)[0])

    return job_lengths, op_pts, names


def save_results(out_dir, names, result):
    """
        save per-instance makespan/solve_time results as a 2-sheet excel file
    :param out_dir: the directory to save 'results.xlsx' into
    :param names: instance names (same order as result rows)
    :param result: array with shape [n_instances, 2] -> [makespan, solve_time]
    """
    os.makedirs(out_dir, exist_ok=True)

    makespan_df = pd.DataFrame({'file_name': names, 'makespan': result[:, 0]})
    solvetime_df = pd.DataFrame({'file_name': names, 'solve_time': result[:, 1]})

    with pd.ExcelWriter(os.path.join(out_dir, 'results.xlsx')) as writer:
        makespan_df.to_excel(writer, sheet_name='makespan', index=False)
        solvetime_df.to_excel(writer, sheet_name='solve_time', index=False)


def main():
    subfolders = sorted(os.listdir(TEST_ROOT))
    override = os.environ.get('TEST_SUBFOLDERS')
    if override:
        subfolders = [s.strip() for s in override.split(',') if s.strip()]

    for seed in configs.seeds:
        model_path = f'./trained_network/{DATA_SOURCE}/{N_J}x{N_M}+{MODEL_SUFFIX}{seed}.pth'
        if not os.path.exists(model_path):
            print(f'Skipping seed {seed}: model not found at {model_path}')
            continue

        for subfolder in subfolders:
            data_dir = os.path.join(TEST_ROOT, subfolder)
            job_lengths, op_pts, names = load_instances_with_names(data_dir)
            data_set = (job_lengths, op_pts)

            print('-' * 25 + f' seed {seed} | {subfolder} ' + '-' * 25)

            t0 = time.time()
            greedy_result = test_greedy_strategy(data_set, model_path, configs.seed_test)
            save_results(os.path.join(OUTPUT_ROOT, f'{RESULT_PREFIX}{seed}', f'{subfolder}_greedy'), names, greedy_result)
            print(f'greedy done in {time.time() - t0:.1f}s, mean makespan {greedy_result[:, 0].mean():.2f}')

            t0 = time.time()
            sample_result = test_sampling_strategy(data_set, model_path, configs.sample_times, configs.seed_test)
            save_results(os.path.join(OUTPUT_ROOT, f'{RESULT_PREFIX}{seed}', f'{subfolder}_sample'), names, sample_result)
            print(f'sample done in {time.time() - t0:.1f}s, mean makespan {sample_result[:, 0].mean():.2f}')


if __name__ == '__main__':
    main()
