import os
import graph_world.beam.pipeline
import argparse
import yaml
import wandb

def main(argv=None):
    graph_world.beam.pipeline.entry(argv)

if __name__ == '__main__':
    print("reached here")
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_yaml', default='sweep.yaml', help='Path to wandb sweep config file')
    args, remaining_args = parser.parse_known_args()
    # Build absolute path to sweep.yaml relative to this file.
    sweep_yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.sweep_yaml)
    with open(sweep_yaml_path, 'r') as f:
        sweep_config = yaml.safe_load(f)
    sweep_id = wandb.sweep(sweep_config, project="meta_classifier_data_gen")
    wandb.agent(sweep_id, function=graph_world.beam.pipeline.entry, count=25)