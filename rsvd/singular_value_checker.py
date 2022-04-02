import numpy as np
import sys
import matfile
import cupy

s_list = []
norm_list = []
for arg in range(1, len(sys.argv)):
    file_name = sys.argv[arg]
    fpbit = matfile.get_fp_bit(file_name)

    if fpbit == 32:
        mat0 = matfile.load_dense_fp32(file_name)
    else:
        mat0 = matfile.load_dense_fp64(file_name)

    mat0 = cupy.array(mat0, dtype=cupy.float64)

    mat0 = (mat0 @ mat0.T) @ mat0
    #mat0 = (mat1 @ mat1.T)
    #mat0 = (mat2 @ mat2) @ mat0

    s = cupy.linalg.svd(mat0, full_matrices=False, compute_uv=False)

    s = cupy.asnumpy(s)

    n = float(cupy.linalg.norm(mat0, ord='fro'))
    print(s/n)
    norm_list += [n]
    #print(cupy.linalg.norm(mat0, ord='fro'))

    if len(s_list) == 0:
        s_list = s
    else:
        s_list = np.c_[s_list, s]

    matfile.save_dense_fp32(cupy.asnumpy(mat0), file_name + ".ip4")

matfile.save_dense_fp32(s_list, 's_list_2.matrix')
print(norm_list)
