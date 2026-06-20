#!/usr/bin/env python3

# Copyright 2023 Jacob Hartzer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
A collection of function for running the multi-IMU, multi-Camera simulation.

Typical usage is:
```
python3 eval/run_sim.py config/example.yaml
```

To get help:
```
python3 eval/run_sim.py --help
```
"""

import copy
import math
import multiprocessing
import os
import random
import subprocess
import tempfile
import traceback

from input_parser import InputParser
import yaml


def add_gitignore(directory):
    """Add gitignore to a directory."""
    f_path = os.path.join(directory, '.gitignore')
    with open(f_path, 'w') as file_id:
        file_id.write('*\n')


def print_err(err):
    """Print errors experienced in asynchronous pool."""
    print('error_callback()', err)
    traceback.print_exception(type(err), err, err.__traceback__)


def run_sim(
    yaml_path: str,
    sim_bin: str = None,
    output_dir: str = None,
    override_yaml: dict = None
):
    """Run simulation given an input yaml."""
    # Get (and create) yaml directory
    yaml_dir = output_dir if output_dir is not None else yaml_path.split('.yaml')[0] + os.sep
    if (not os.path.isdir(yaml_dir)):
        os.mkdir(yaml_dir)
        with open(os.path.join(yaml_dir, '.gitignore'), 'w') as f_git_ignore:
            f_git_ignore.write('*\n')

    run_yaml_path = yaml_path
    temp_yaml_path = None
    if override_yaml is not None:
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.yaml',
            prefix='.sim_override_',
            dir=yaml_dir,
            delete=False
        ) as temp_yaml:
            yaml.dump(override_yaml, temp_yaml)
            temp_yaml_path = temp_yaml.name
            run_yaml_path = temp_yaml_path

    # Run simulation
    if sim_bin is None:
        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        candidate_sim_bin_paths = [
            os.path.join(base_path, '..', '..', 'build', 'ekf_cal', 'sim'),
            os.path.join(base_path, 'build', 'RelWithDebInfo', 'sim'),
        ]
        max_mtime = -1
        for path in candidate_sim_bin_paths:
            if os.path.isfile(path):
                mtime = os.path.getmtime(path)
                if mtime > max_mtime:
                    max_mtime = mtime
                    sim_bin = path

    if not sim_bin or not os.path.isfile(sim_bin):
        err_msg = "The 'sim' binary was not found in any of the candidate paths."
        raise FileNotFoundError(err_msg)

    try:
        proc = subprocess.run([sim_bin, run_yaml_path, yaml_dir],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
    finally:
        if temp_yaml_path is not None and os.path.isfile(temp_yaml_path):
            os.remove(temp_yaml_path)

    # Write stdout
    if (proc.stdout):
        with open(os.path.join(yaml_dir, 'sim.stdout'), 'wb') as output:
            output.write(proc.stdout)

    # Write stderr
    if (proc.stderr):
        print(f'Warning: run {yaml_path} has errors', flush=True)
        with open(os.path.join(yaml_dir, 'sim.stderr'), 'wb') as output:
            output.write(proc.stderr)


def generate_mc_from_yaml(
    yaml_file,
    runs=None,
    time=None,
    generate_video=False
):
    """Generate a Monte Carlo list of inputs given a top-level yaml."""
    yaml_jobs = []
    with open(yaml_file, 'r') as yaml_stream:
        try:
            top_yaml = yaml.safe_load(yaml_stream)
            sim_yaml = top_yaml['/EkfCalNode']['ros__parameters']['sim_params']
            num_runs = runs if runs is not None else sim_yaml['number_of_runs']
            yaml_dir = yaml_file.split('.yaml')[0] + os.sep
            seed = sim_yaml['seed']

            if (num_runs > 1):
                top_name = os.path.basename(yaml_file).split('.yaml')[0]
                if (not os.path.isdir(yaml_dir)):
                    os.mkdir(yaml_dir)
                add_gitignore(yaml_dir)
                runs_dir = os.path.join(yaml_dir, 'runs')
                if (not os.path.isdir(runs_dir)):
                    os.mkdir(runs_dir)
                if (seed):
                    random.seed(seed)
                n_digits = math.ceil(math.log10(num_runs))

            for i in range(num_runs):
                if (num_runs == 1 and runs is None and time is None and not generate_video):
                    yaml_jobs.append({
                        'yaml_path': yaml_file,
                        'output_dir': None,
                        'override_yaml': None
                    })
                    continue

                sub_yaml = copy.deepcopy(top_yaml)
                sub_sim_params = sub_yaml['/EkfCalNode']['ros__parameters']['sim_params']
                sub_sim_params['number_of_runs'] = 1

                if (num_runs > 1):
                    if (seed):
                        sub_sim_params['seed'] = random.randint(0, 1000000000)
                    sub_sim_params['run_number'] = sim_yaml['run_number'] + i
                if (time is not None):
                    sub_sim_params['max_time'] = time
                if (generate_video):
                    sub_sim_params['generate_video'] = True

                if (num_runs > 1):
                    sub_file = os.path.join(
                        runs_dir, '{}_{:0{:d}.0f}.yaml'.format(top_name, i, n_digits))
                    with open(sub_file, 'w') as f:
                        yaml.dump(sub_yaml, f)
                    yaml_jobs.append({
                        'yaml_path': sub_file,
                        'output_dir': None,
                        'override_yaml': None
                    })
                else:
                    if (not os.path.isdir(yaml_dir)):
                        os.mkdir(yaml_dir)
                    add_gitignore(yaml_dir)
                    yaml_jobs.append({
                        'yaml_path': yaml_file,
                        'output_dir': yaml_dir,
                        'override_yaml': sub_yaml
                    })
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_jobs


def add_jobs(args):
    """Add simulation jobs to pool given list of top-level input yaml files."""
    cpu_count = args.jobs if (args.jobs) else multiprocessing.cpu_count() - 1
    pool = multiprocessing.Pool(cpu_count)

    sim_bin = getattr(args, 'sim_bin', None)

    for yaml_file in args.inputs:
        input_yaml_path = os.path.abspath(yaml_file)
        job_list = generate_mc_from_yaml(
            input_yaml_path,
            runs=args.runs,
            time=args.time,
            generate_video=args.generate_video
        )
        for job in job_list:
            pool.apply_async(
                run_sim,
                kwds={
                    'yaml_path': job['yaml_path'],
                    'sim_bin': sim_bin,
                    'output_dir': job['output_dir'],
                    'override_yaml': job['override_yaml']
                },
                error_callback=print_err)

    pool.close()
    pool.join()


# TODO(jhartzer): Write tests
if __name__ == '__main__':
    parser = InputParser()
    args = parser.parse_args()
    add_jobs(args)
