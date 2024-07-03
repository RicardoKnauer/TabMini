import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd

def run_experiment_in_env(env_name, output_path, time_limit):
    return subprocess.Popen(['bash', 'run_experiment.sh', env_name, str(output_path), str(time_limit)])

def update_results(path, time_limit):
    results = pd.read_csv(path / f"results_{time_limit}.csv")
    aggregated_results_file = Path('results') / f'aggregated_results_{time_limit}.csv'
    
    if aggregated_results_file.exists():
        aggregated_results = pd.read_csv(aggregated_results_file)
        aggregated_results = pd.merge(aggregated_results, results, on=['PMLB dataset', 'Logistic Regression'])
    else:
        aggregated_results = results
    
    aggregated_results_file.parent.mkdir(exist_ok=True)
    aggregated_results.to_csv(aggregated_results_file, index=False)

def main():
    working_directory = Path.cwd() / "workdir"
    frameworks = ['tabpfn', 'hyperfast']
    processes = []
    time_limits = [60, 120]
    datetime_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    for time_limit in time_limits:
        for env in frameworks:
            output_path = working_directory / f"Exp{datetime_str}_{env}"
            output_path.mkdir(parents=True, exist_ok=True)
            process = run_experiment_in_env(env, output_path, time_limit)
            processes.append((process, env, output_path, time_limit))
        
        for process, env, output_path, time_limit in processes:
            process.wait()
            update_results(output_path, time_limit)
    
    print("All experiments completed and results updated.")

if __name__ == "__main__":
    main()
