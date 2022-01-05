from torsk.numpy_accelerate import bh, to_bh


def sparse_dense_mm(A, X, timer=None):
    """Matrix-matrix multiplication for fixed nonzero per row sparse matrix"""
    #    start_timer(timer,"spmm")

    assert A.dense_shape[1] == X.shape[0]
    AXv = A.values[:, None] * X[A.col_idx]
    result = bh.sum(AXv.reshape((-1, X.shape[1], A.nonzeros_per_row)), axis=2)

    #    end_timer(timer)
    return result


def sparse_dense_mv(A, x, timer=None):
    """Matrix-vector multiplication for fixed nonzero per row sparse matrix"""
    # assert A.dense_shape[1] == x.shape[0]
    # assert bh_check(A.values)
    # assert bh_check(A.col_idx)
    # assert bh_check(x)
    # start_timer(timer,"spmv")

    values = A.values.reshape((A.m, A.n_nz))
    col_idx = A.col_idx.reshape((A.m, A.n_nz))

    Axv = values * x[col_idx]
    result = bh.sum(Axv, axis=1)

    # end_timer(timer)
    return result


class SparseMatrix:
    """A simple COO sparse matrix implementation that allows only matrices
    with a fixed number of nonzero elements per row.
    """

    def __init__(self, values, col_idx, nonzeros_per_row, dense_shape, timer=None):
        assert values.shape == col_idx.shape

        (m, n, n_nz) = (dense_shape[0], dense_shape[1], nonzeros_per_row)

        self.values = to_bh(values).reshape((m, n_nz))
        self.col_idx = to_bh(col_idx).reshape((m, n_nz))
        self.nonzeros_per_row = nonzeros_per_row
        self.dense_shape = dense_shape
        self.timer = timer

    @classmethod
    def from_dense(cls, dense_matrix, nonzeros_per_row):
        row_idx, col_idx = bh.nonzero(dense_matrix)
        # FIXME: Assumes nonzeros are *exactly* nonzeros_per_row, not at most.
        values = dense_matrix[row_idx, col_idx]
        return cls(values, col_idx, nonzeros_per_row, dense_matrix.shape)

    def sparse_dense_mm(self, X):
        return sparse_dense_mm(self, X, self.timer)

    def sparse_dense_mv(self, x):
        return sparse_dense_mv(self, x, self.timer)

    @property
    def m(self):
        return self.dense_shape[0]

    @property
    def n(self):
        return self.dense_shape[1]

    @property
    def n_nz(self):
        return self.nonzeros_per_row

    def __str__(self):
        nz = self.nonzeros_per_row
        ds = self.dense_shape
        name = self.__class__.__name__
        string = f"<{name} nonzeros_per_row:{nz} values:{self.values} shape:{ds}>"
        return string
