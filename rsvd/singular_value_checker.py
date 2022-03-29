import numpy as np
import sys
import matfile


s_list = []
for arg in range(1, len(sys.argv)):
    file_name = sys.argv[arg]
    fpbit = matfile.get_fp_bit(file_name)

    mat0 = np.array([])
    if fpbit == 32:
        mat0 = matfile.load_dense_fp32(file_name)
    else:
        mat0 = matfile.load_dense_fp64(file_name)

    s = np.linalg.svd(mat0, full_matrices=False, compute_uv=False)

    print(s)

    if len(s_list) == 0:
        s_list = s
    else:
        s_list = np.c_[s_list, s]

matfile.save_dense_fp32(s_list, 's_list.matrix')
