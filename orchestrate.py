import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd

def run_experiment_in_env(env_name, output_path, time_limit):
    return subprocess.Popen(['bash', 'run_experiment.sh', env_name, str(output_path), str(time_limit)])

def update_results(path, time_limit, framework):
    results = pd.read_csv(path / f"results_{time_limit}.csv")
    aggregated_results_file = path.parent / f'aggregated_results_{time_limit}.csv'
    
    if aggregated_results_file.exists():
        aggregated_results = pd.read_csv(aggregated_results_file)
        aggregated_results = pd.merge(aggregated_results, results, on=['PMLB dataset', 'Logistic Regression'])
 
        aggregated_results.set_index(['PMLB dataset', 'Logistic Regression'], inplace=True)
        results.set_index(['PMLB dataset', 'Logistic Regression'], inplace=True)

        aggregated_results.update(results[framework])

        # Reset index if necessary
        aggregated_results.reset_index(inplace=True)
    else:
        aggregated_results = results
    
    aggregated_results_file.parent.mkdir(exist_ok=True)
    aggregated_results.to_csv(aggregated_results_file, index=False)

def main():
    datetime_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    working_directory = Path.cwd() / "workdir" / f"Exp_{datetime_str}"
    frameworks = ['tabpfn', 'hyperfast']
    processes = []
    time_limits = [60, 120]

    for time_limit in time_limits:
        for env in frameworks:
            output_path = working_directory / env
            output_path.mkdir(parents=True, exist_ok=True)
            process = run_experiment_in_env(env, output_path, time_limit)
            processes.append((process, env, output_path, time_limit))
        
        for process, env, output_path, time_limit in processes:
            process.wait()
            update_results(output_path, time_limit, env)
    
    print("All experiments completed and results updated.")

if __name__ == "__main__":
    main()