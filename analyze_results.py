import os
import json
from tabulate import tabulate
import argparse

def load_metrics(metrics_path):
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Analyze MedAgents-2 experiment results.")
    parser.add_argument("--experiments", nargs='+', required=True, help="Experiment names (space separated, e.g. search search_features)")
    parser.add_argument("--datasets", nargs='+', required=True, help="Dataset names (space separated, e.g. medqa medmcqa)")
    parser.add_argument("--runs", nargs='+', required=True, help="Run numbers (space separated, e.g. 0 1 2)")
    parser.add_argument("--models", nargs='+', required=True, help="Model names (space separated, e.g. gpt-4o-mini)")
    parser.add_argument("--output_base", type=str, default="output", help="Base output directory")
    args = parser.parse_args()

    table_rows = []
    for dataset in args.datasets:
        for experiment in args.experiments:
            for run_id in args.runs:
                for model in args.models:
                    metrics_path = os.path.join(
                        args.output_base,
                        dataset,
                        experiment,
                        f"run_{run_id}",
                        model,
                        "metrics.json"
                    )
                    metrics = load_metrics(metrics_path)
                    if metrics is None:
                        continue
                    acc = metrics['accuracy_metrics']
                    usage = metrics['token_usage_metrics']
                    time = metrics['time_metrics']
                    summary = metrics['experiment_summary']
                    row = [
                        experiment,
                        dataset,
                        run_id,
                        model,
                        f"{acc['overall_accuracy']:.2%}",
                        acc['correct_answers'],
                        acc['total_answers'],
                        usage['total_usage']['total_tokens'],
                        f"{time['total_time']:.1f}",
                        f"{time['average_time']:.2f}"
                    ]
                    table_rows.append(row)
    headers = [
        "Experiment", "Dataset", "Run", "Model", "Accuracy", "Correct", "Total", "Total Tokens", "Total Time (s)", "Avg Time/Prob (s)"
    ]
    if table_rows:
        print(tabulate(table_rows, headers=headers, tablefmt="github"))
    else:
        print("No results found.")

if __name__ == "__main__":
    main() 