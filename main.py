import numpy as np
import pandas as pd
from MantaRayOA import MantaRayOA
from OBHSA import OBHSA
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--csv_name', type=str, required = True, help='Name of csv file- Example: SpectEW.csv')
parser.add_argument('--csv_header', type=str, default = 'no', help='Does csv file have header?: yes/no')
parser.add_argument('--generations', type=int, default = 5, help='Number of Generations to run the Genetic Algorithm')
parser.add_argument('--popSize', type=int, default = 20, help='Population Size to be used in MRFO and OBHSA')
args = parser.parse_args()

root = "./"

if root[-1]!='/':
    root+='/'
csv_path = args.csv_name
if args.csv_header=='yes':
    df = np.asarray(pd.read_csv(root+csv_path))
else:
    df = np.asarray(pd.read_csv(root+csv_path,header=None))
data = df[:,0:-1]
target = df[:,-1]

pop_size = args.popSize
num_gen = args.generations

pop1 = MantaRayOA(data,target, popSize = pop_size, num_generations=num_gen)
OBHSA(data,target, pop1, num_generations=num_gen)
