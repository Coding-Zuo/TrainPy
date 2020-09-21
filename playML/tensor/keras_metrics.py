import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


batchsze = 128
(x, y), (x_val, y_val) = datasets.mnist.load_data()
print("datasets:", x.shape, y.shape, x.min(), x.max())

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(10000).batch(batchsze).repeat(10)

ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.map(preprocess).batch(batchsze)

network = Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10)
])
network.build(input_shape=(None, 28 * 28))
network.summary()

optimizers = optimizers.Adam(lr=0.01)
acc_meter = metrics.Accuracy()
loss_meter = metrics.Mean()

for step, (x, y) in enumerate(db):
    with tf.GradientTape() as tape:
        x = tf.reshape(x, (-1, 28 * 28))
        out = network(x)
        y_onehot = tf.one_hot(y, depth=10)
        loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, out, from_logits=True))
        loss_meter.update_state(loss)

    grads = tape.gradient(loss, network.trainable_variables)
    optimizers.apply_gradients(zip(grads, network.trainable_variables))

    if step % 100 == 0:
        print(step, 'loss:', loss_meter.result().numpy())
        loss_meter.reset_states()

    if step % 500 == 0:
        total, total_correct = 0., 0
        acc_meter.reset_states()

        for step, (x, y) in enumerate(ds_val):
            x = tf.reshape(x, (-1, 28 * 28))
            out = network(x)

            pred = tf.argmax(out, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.equal(pred, y)
            total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
            total += x.shape[0]
            acc_meter.update_state(y, pred)

        print(step, 'Evaluate Acc:', total_correct / total, acc_meter.result().numpy())
