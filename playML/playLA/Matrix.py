from .Vector import Vector


class Matrix:
    def __init__(self, list2d):
        self._values = [row[:] for row in list2d]

    @classmethod
    def zero(cls, r, c):
        return cls([0] * c for _ in range(r))

    @classmethod
    def identity(cls, n):
        m = [[0] * n for _ in range(n)]
        for i in range(n):
            m[i][i] = 1
        return cls(m)

    def T(self):
        return Matrix([[e for e in self.col_vector(i)]
                       for i in range(self.col_num())])

    def __add__(self, another):
        assert self.shape() == another.shape(), \
            "Error in adding Shape of matrix must be same"
        return Matrix([[a + b for a, b in zip(self.row_vector(i), another.row_vector(i))]
                       for i in range(self.row_num())])

    def __sub__(self, another):
        assert self.shape() == another.shape(), \
            "Error in adding Shape of matrix must be same"
        return Matrix([[a - b for a, b in zip(self.row_vector(i), another.row_vector(i))]
                       for i in range(self.row_num())])

    def __mul__(self, k):
        return Matrix([e * k for e in self.row_vector(i)] for i in range(self.row_num()))

    def __rmul__(self, k):
        return k * self

    def __truediv__(self, k):
        return (1 / k) * self

    def dot(self, another):
        if isinstance(another, Vector):
            assert self.col_num() == len(another), \
                "Error in Matrix-Vector Multiplication"
            return Vector([self.row_vector(i).dot(another) for i in range(self.row_num())])
        if isinstance(another, Matrix):
            assert self.col_num() == another.row_num(), \
                "Error in Matrix-Matrix Multiplication"
            return Matrix([[self.row_vector(i).dot(another.col_vector(j)) for j in range(another.col_num())]
                           for i in range(self.row_num())])

    def __pos__(self):
        return 1 * self

    def __neg__(self):
        return -1 * self

    def row_vector(self, index):
        """返回矩阵的第index个行向量"""
        return Vector(self._values[index])

    def col_vector(self, index):
        """列"""
        return Vector([row[index] for row in self._values])

    def __getitem__(self, pos):
        """返回矩阵pos的位置"""
        r, c = pos
        return self._values[r][c]

    def row_num(self):
        return self.shape()[0]

    def col_num(self):
        return self.shape()[1]

    def size(self):
        r, c = self.shape()
        return r * c

    def __len__(self):
        return self.row_num()

    def shape(self):
        return len(self._values), len(self._values[0])

    def __repr__(self):
        return "Matrix({})".format(self._values)

    __str__ = __repr__
