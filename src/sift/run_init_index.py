from init_index import init_index
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str)

args = parser.parse_args()

init_index(args.dataset)
