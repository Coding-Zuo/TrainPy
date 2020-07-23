from playLA.Matrix import Matrix
from playLA.Vector import Vector
from playLA._global import is_zero
from keras.utils import to_categorical

def lu(matrix):
    assert matrix.row_num() == matrix.col_num(), "matrix must be a square matrix"
    n = matrix.row_num()
    A = [matrix.row_vector(i) for i in range(n)]
    L = [[1.0 if i == j else 0.0 for i in range(n)] for j in range(n)]

    for i in range(n):
        # 看A[i][i]是否可以是主元
        if is_zero(A[i][i]):
            return None, None
        else:
            for j in range(i + 1, n):
                p = A[j][i] / A[i][i]
                A[j] = A[j] - p * A[i]
                L[j][i] = p
    return Matrix(L), Matrix([A[i].underlying_list() for i in range(n)])


if __name__ == "__main__":
    A = Matrix([[1, 2, 3], [4, 5, 6], [3, -3, 5]])
    L, U = lu(A)
    print(L)
    print(U)
    print(L.dot(U))
