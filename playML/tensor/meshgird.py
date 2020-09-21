import tensorflow as tf
import matplotlib.pyplot as plt


def func(x):
    z = tf.math.sin(x[..., 0]) + tf.math.sin(x[..., 1])
    return z


x = tf.linspace(0., 2 * 3.14, 500)
y = tf.linspace(0., 2 * 3.14, 500)

point_x, point_y = tf.meshgrid(x, y)

points = tf.stack([point_x, point_y], axis=2)

print('points:', points.shape)
z = func(points)
print("z:", z.shape)

plt.figure('plot 2d func value')
plt.imshow(z, origin='lower', interpolation='none')
plt.colorbar()

plt.figure('plot 2d func contour')
plt.contour(point_x, point_y, z)
plt.show()