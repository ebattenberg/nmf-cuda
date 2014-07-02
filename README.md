nmf-cuda
========

A CUDA implementation of non-negative matrix factorization for GPUs.

A description of the implementation is available in [1] below.  If you use this in your work, please cite [1].

[1] E. Battenberg and D. Wessel, “Accelerating non-negative matrix factorization for audio source separation on multi-core and many-core architectures,” in International Society for Music Information Retrieval Conference (ISMIR 2009), 2009.

Bibtex entry:
'''
@inproceedings{battenberg2009accelerating,
    title={Accelerating nonnegative matrix factorization for audio source separation on multi-core and many-core architectures},
    author={Battenberg, E. and Wessel, D.},
    booktitle={10th International Society for Music Information Retrieval Conference (ISMIR 2009)},
    year={2009}
}
'''



#Implementation Details:

Iterative NMF on Cuda: X = W*H  
    multiplicative updates
    divergence cost function


nmf.cu contains an example usage of the update_div function.

The function read_matrix reads binary floats in from files and stores them in 
a matrix struct

* values need to be stored in column-major order, and the first two values in the file are integer dimensions of the matrix
* an example of how to create the binary files (from e.g. Matlab data) is contained in the matlab script matrix_export.m
* data can also just be stored in a column-major float array using the matrix struct and proper assignment of the dim values

The function update_div is where the work is done.
update_div(matrix W,matrix H,matrix X,float CONVERGE_THRESH,int max_iter,double t[10],int verbose);

* W, H are initial values for the factor matrices
* X is the target matrix to be decomposed
* CONVERGE_THRESH is the convergence threshold (expressed as a ratio of cost function change to cost function value)
* max_iter is the maximum number of iterations
* double t[10] is a pointer to a double array of at least size 10 that will contain individual timing results for different computational pieces.  (set this to NULL for normal use)
* verbose set to 1 if you want more text output, 0 otherwise



