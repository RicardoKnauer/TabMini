import subprocess
import os
import json

conda_envs = [
    'tabpfn',
    'hyperfast',
]

def run_experiment_in_env(env_name):
    process = subprocess.Popen(['bash', 'run_experiment.sh', env_name])
    return process

os.makedirs('results', exist_ok=True)

processes = []
for env in conda_envs:
    process = run_experiment_in_env(env)
    processes.append((process, env))

for process, env in processes:
    process.wait()

