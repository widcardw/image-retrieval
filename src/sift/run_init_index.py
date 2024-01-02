from init_index import init_index
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str)
parser.add_argument('--n_clusters', type=int)

args = parser.parse_args()

init_index(args.dataset, args.n_clusters)
