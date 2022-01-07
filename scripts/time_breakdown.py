import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import pandas as pd

parser = argparse.ArgumentParser(description="rsvd time breakdown graph maker")
parser.add_argument('--input_dir', type=str)

args = parser.parse_args()

files = glob.glob(args.input_dir + "/*.csv")

data_list = []
label_list = []

for file in files:
    params = file.split('-')
    m = params[-4]
    n = params[-3]
    k = params[-2]
    p = params[-1].split('.')[0]

    label = 'm' + str(m) + '-n' + str(n) + '-k' + str(k) + '-p' + str(p)
    label_list += [label]

    df = pd.read_csv(file)
    rows = df['name']
    s = df.sum(axis=0)['sum_us']
    ratio = df['sum_us'] / s
    d = {rows[i]: ratio[i] for i in range(len(rows))}
    data_list += [d]

df = pd.DataFrame(data=data_list, index=label_list)
df = df.sort_values(by=['matmul_1'])

fig, ax = plt.subplots(figsize=(10, 8))
for i in range(len(df.index)):
    ax.bar(df.columns, df.iloc[i], bottom=df.iloc[:i].sum())

plt.savefig('time_breakdown.pdf', transparent=True)
