import matplotlib.pyplot as plt
import pandas as pd

# Output
output_file_name = "figure.pdf"

# The list of `type` in the input csv file
imp_list = ['cusolver_svdr', 'selfmade', 'svdj']
implementation_name_table = {
        'cusolver_svdr': 'cuSOLVER-svdr',
        'selfmade': 'Selfmade',
        'svdj': 'cuSOLVER-svdj',
        }
color_table = {
        'cusolver_svdr': '#33ff33',
        'selfmade': '#ff3333',
        'svdj': '#3333ff',
        }

n_row = 6
rank_div_list = [2**i for i in range(1, n_row)]
matrix_size_list = [2**i for i in range(5, 13)]

# Figure config
plt.figure(figsize=(8, 3))
plt.xlabel("matrix size $N \\times N$, $N$")
plt.ylabel("Residual")
plt.grid()

# Load input data
df = pd.read_csv("data.csv", encoding="UTF-8")
df = df.query('m==n')

# Create graph
fig, axs = plt.subplots(len(rank_div_list), 3, figsize=(16, n_row * 3))

n_line_added = 0
line_list = []
label_list = []
for rank_div_index, rank_div in enumerate(rank_div_list):
    axs[rank_div_index][0].set_title('res(rank=n/' + str(rank_div) + ')')
    axs[rank_div_index][1].set_title('ort-U(rank=n/' + str(rank_div) + ')')
    axs[rank_div_index][2].set_title('ort-V(rank=n/' + str(rank_div) + ')')

    for index, value in enumerate(['residual', 'u_orthogonality', 'v_orthogonality']):
        axs[rank_div_index][index].set_xscale("log", base=2)
        axs[rank_div_index][index].set_yscale("log", base=10)
        axs[rank_div_index][index].set_ylim((1e-7, 1e-2))
        axs[rank_div_index][index].grid()
        for imp in imp_list:
            x_list = []
            y_list = []
            for x in matrix_size_list:
                v = df.query('implementation==\'' + imp + '\'&m==' + str(x) + '&k==' + str(x / rank_div))[value].to_numpy()
                if len(v) != 0:
                    x_list += [x]
                    y_list += [v[0]]
            print(y_list)
            l = axs[rank_div_index][index].plot(
                    x_list,
                    y_list,
                    markersize=4,
                    marker="*",
                    color=color_table[imp])
            if n_line_added < 3:
                line_list += [l]
                label_list += [implementation_name_table[imp]]
                n_line_added += 1

# Legend config
fig.legend(line_list,
        labels=label_list,
        loc='upper center',
        ncol=len(imp_list))

# Save to file
fig.savefig(output_file_name, bbox_inches="tight")
