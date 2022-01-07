import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import pandas as pd

include_rand_gen = False

def df_column_swap(df, column1, column2):
    i = list(df.columns)
    a, b = i.index(column1), i.index(column2)
    i[b], i[a] = i[a], i[b]
    df = df[i]
    return df

parser = argparse.ArgumentParser(description="rsvd time breakdown graph maker")
parser.add_argument('--input_dir', type=str)

args = parser.parse_args()

files = glob.glob(args.input_dir + "/*.csv")

fig, axs = plt.subplots(2, 1, figsize=(30, 8))


for index, include_rand_gen in enumerate([True, False]):
    ax = axs[index]
    if include_rand_gen:
        ax.set_title("w/ gen_rand")
    else:
        ax.set_title("w/o gen_rand")
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
        if not include_rand_gen:
            df = df.query('name!="gen_rand"')
        rows = df['name']
        rows = rows.values.tolist()
        s = df.sum(axis=0)['sum_us']
        ratio = df['sum_us'] / s * 100
        ratio = ratio.values.tolist()
        d = {rows[i]: ratio[i] for i in range(len(rows))}
        data_list += [d]

    df = pd.DataFrame(data=data_list, index=label_list)

    if include_rand_gen:
        df = df_column_swap(df, 'gen_rand', 'matmul_1')
        df = df_column_swap(df, 'gen_rand', 'matmul_2')

    df = df.sort_values(by=['matmul_1'])

    n_rows, n_cols = df.shape
    positions = np.arange(n_rows)
    offsets = np.zeros(n_rows, dtype=df.values.dtype)
    colors = plt.get_cmap("tab20c")(np.linspace(0, 1, n_cols))

    for i in range(len(df.columns)):
        bar = ax.bar(positions, df.iloc[:, i], bottom=offsets, color=colors[i], label=df.columns[i])
        offsets += df.iloc[:, i]

    ax.grid()
    ax.legend(bbox_to_anchor=(1, 1))

plt.ylabel("time breakdown")
plt.xlabel("input matrix shape")
plt.savefig('time_breakdown.pdf', transparent=True, bbox_inches='tight')
