import tensorflow as tf

# tf.Variable 生成的变量，每次迭代都会变化
# 这个变量也就是我们要计算的结果，所以你要计算什么，你是不是就把什么定义为Variable
"""
Tensorflow程序可以通过tf.device函数来指定运行每一个操作的设备
这个设备可以是本地的CPU或者GPU，也可以是某一台远程服务器
TensorFlow会给每一个可用的设备一个名称，tf.device函数可以通过设备的名称，来指定执行运算的设备。比如CPU在Tensorflow中名称为/cpu:0

在默认情况下，即使多个CPU，TensorFlow也不会区分它们，所有的CPU都使用/gpu:n
比如一个GPU的名称为/gpu:0,第二个叫/gpu:1
TensorFlow提供了一个快捷的方式，来查看运行每个运算的设备
在生成会话时，可通过设置log_device_placement参数来打印运行每个运算的设备

除了可以看到最后的计算结果外，还可以看到类似"add: /job:localhost/replica:0/task:0/cpu:0"这样的输出
这些输出显示了执行每个运算的设备，比如加法操作add是通过CPU来运行的，因为设备名称中包含了/cpu:0
在配置好GPU环境的Tensorflow中，如果操作没有明确指定运行设备，那么Tensorflow会优先选择gpu
"""
with tf.device('/cpu:0'):
    x = tf.Variable(3, name='x')

y = tf.Variable(4, name='y')
f = x * x * y + y + 2
# print(f)

# 创建一个计算图的上下文环境
# 配置里面是把具体运行过程在哪里执行打印出来
# 原版 config = tf.ConfigProto(allow_soft_placement=True)
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
tf.compat.v1.disable_eager_execution()  # 保证sess.run()能够正常运行
hello = tf.constant('hello,tensorflow')
# 原版 sess = tf.Session(config=config)
sess = tf.compat.v1.Session()  # 注意 ，这里为tensorflow2.0版本，与第1.0有差距。
# 碰到session.run()就会立即去调用计算
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
sess.close()
