from playLA.Vector import Vector
from playLA.Matrix import Matrix
from playLA.LinearSystem import LinearSystem

if __name__ == "__main__":
    vec = Vector([5, 2])
    print(vec)

    vec2 = Vector([3, 1])
    print("{}+{}={}".format(vec, vec2, vec + vec2))
    print("{}+{}={}".format(vec, vec2, vec - vec2))
    print("{}+{}={}".format(vec, 3, vec * 3))
    print("{}+{}={}".format(3, vec2, 3 * vec2))
    print("+{}={}".format(vec, +vec))
    print("-{}={}".format(vec, -vec))

    A7 = Matrix([[1, -1, 2, 0, 3],
                 [-1, 1, 0, 2, -5],
                 [1, -1, 4, 2, 4],
                 [-2, 2, -5, -1, -3]])
    b7 = Vector([1, 5, 13, -1])
    ls7 = LinearSystem(A7, b7)
    ls7.gauss_jordan_elimination()
    ls7.fancy_print()
