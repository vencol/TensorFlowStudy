import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input
mnist = input.read_data_sets('MNIST_data', one_hot=True)
#占位符变量
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder("float", [None,10])
#变量
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#初始化变量
sess = tf.Session()
sess.run(tf.initialize_all_variables())
#回归模型
y = tf.nn.softmax(tf.matmul(x,W) + b)
#评估我们的模型
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#训练步长设置
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#开始训练
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#训练精度计算
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print ("the accuracy is:", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}), "end")
