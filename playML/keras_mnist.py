import numpy as np


def load_data_npz(path='mnist.npz'):
    """
    path:mnist.npz文件的路径
    """
    f = np.load(path)  # np.load文件可以加载npz，npy格式的文件
    x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f['x_test'], f['y_test']
    f.close()
    return x_train, y_train, x_test, y_test


# 调用load_data函数加载mnist.npz数据集
path = '../2020/mnist.npz'
x_train, y_train, x_test, y_test = load_data_npz(path)
print('x_train_npz:{}'.format(x_train.shape))
print('y_train_npz:{}'.format(y_train.shape))
print('x_test_npz:{}'.format(x_test.shape))
print('y_test_npz:{}'.format(y_test.shape))

img1 = x_train[0]
from matplotlib import pyplot as plt

fig1 = plt.figure(figsize=(5, 5))
plt.imshow(img1)

feature_size = img1.shape[0] * img1.shape[1]
X_train_format = x_train.reshape(x_train.shape[0], feature_size)
X_test_format = x_test.reshape(x_test.shape[0], feature_size)
X_train_normal = X_train_format / 255
X_test_normal = X_test_format / 255

from keras.utils import to_categorical

y_train_format = to_categorical(y_train)
y_test_format = to_categorical(y_test)
print(y_train_format[0])

from keras.models import Sequential
from keras.layers import Dense, Activation

mlp = Sequential()
mlp.add(Dense(units=392, activation='sigmoid', input_dim=feature_size))
mlp.add(Dense(units=392, activation='sigmoid'))
mlp.add(Dense(units=10, activation='softmax'))
mlp.summary()

mlp.compile(loss='categorical_crossentropy', optimizer='adam')
mlp.fit(X_train_normal, y_train_format, epochs=10)

y_train_predict = mlp.predict_classes(X_train_normal)
print(y_train_predict)
from sklearn.metrics import accuracy_score

accuracy_train = accuracy_score(y_train, y_train_predict)
print(accuracy_train)
y_test_predict = mlp.predict_classes(X_test_normal)
accuracy_test = accuracy_score(y_test, y_test_predict)
print(accuracy_test)

img2 = x_test[10]
fig2 = plt.figure(figsize=(3, 3))
plt.imshow(img2)
plt.title(y_test_predict[10])
plt.show()
