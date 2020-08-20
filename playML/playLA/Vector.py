import math
from ._global import is_zero, is_equal


class Vector:

    def __init__(self, lst):
        self._values = list(lst)

    def __add__(self, another):
        assert len(self) == len(another), \
            "Error in adding. Length of vectors must be same"
        return Vector([a + b for a, b in zip(self, another)])

    @classmethod
    def zero(cls, dim):
        return cls([0] * dim)

    def underlying_list(self):
        """返回向量的底层列表 副本"""
        return self._values[:]

    def norm(self):
        """向量的模"""
        return math.sqrt(sum(e ** 2 for e in self))

    def dot(self, another):
        """向量点乘，返回标量"""
        assert len(self) == len(another), \
            "Error in dot product. length of Vectors must be same"
        return sum(a * b for a, b in zip(self, another))

    def normlize(self):
        """向量的模"""
        # return 1 / self.norm() * Vector(self._values)
        if is_zero(self.norm()):
            raise ZeroDivisionError("Nomalize error. norm is zero")
        return Vector(self._values) / self.norm()

    def __eq__(self, other):
        other_list = other.underlying_list()
        if (len(other_list) != len(self._values)):
            return false
        return all(is_equal(x, y) for x, y in zip(self._values, other_list))

    def __neq__(self, other):
        return not (self == other)

    def __sub__(self, another):
        assert len(self) == len(another), \
            "Error in adding. Length of vectors must be same"
        return Vector([a - b for a, b in zip(self, another)])

    def __mul__(self, k):
        # 数量乘法
        return Vector([k * e for e in self._values])

    def __truediv__(self, k):
        # 数量除法
        return (1 / k) * self

    def __rmul__(self, k):
        return self * k

    def __pos__(self):
        # 向量取正
        return 1 * self

    def __neg__(self):
        return -1 * self

    def __iter__(self):
        return self._values.__iter__()

    def __len__(self):
        return len(self._values)

    def __getitem__(self, index):
        return self._values[index]

    def __repr__(self):
        return "Vector({})".format(self._values)

    def __str__(self):
        return "({})".format(", ".join(str(e) for e in self._values))
