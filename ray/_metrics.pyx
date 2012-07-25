cimport cython

cimport numpy as cnp
import numpy as np
cnp.import_array()

@cython.boundscheck(False)
cdef count_ones(cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] a):
    cdef int i
    cdef double count
    for i in range(a.shape[0]):
        if a[i] == 1.0:
            count += 1.0
    return count

@cython.boundscheck(False)
cpdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] continuous_confusion(
        cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] sorted_y_true,
        cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] sorted_probas_pred):
    """Return a confusion matrix for every threshold in the prediction."""
    cdef int i
    cdef int curr_threshold = 0
    cdef int num_samples = sorted_y_true.shape[0]
    cdef int it = 0, itp = 1, itn = 2, ifp = 3, ifn = 4
    cdef cnp.float64_t pr = sorted_probas_pred[0], tn = 0, \
        fn = 0, tp = count_ones(sorted_y_true), fp = num_samples - tp
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] confusion
    confusion = np.zeros((len(sorted_y_true), 5), dtype=np.float64)
    for i in range(num_samples):
        pr = sorted_probas_pred[i]
        if sorted_y_true[i] == 1:
            tp -= 1
            fn += 1
        else:
            fp -= 1
            tn += 1
        confusion[i, 0] = pr
        confusion[i, 1] = tp
        confusion[i, 2] = tn
        confusion[i, 3] = fp
        confusion[i, 4] = fn
    thresholds, idxs = np.unique(sorted_probas_pred, True)
    return confusion[idxs]
