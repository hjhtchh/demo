#coding:utf-8
#分类器
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from TensorFlowDemo import TFdemo5 #添加层的方法
#number 1-10
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#define placeholder for inputs to network
xs = tf.placeholder(tf.float32,[None,784]) #28*28
ys = tf.placeholder(tf.float32,[None,10])

def computer_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result


#add output layer
prediction = TFdemo5.add_layer(xs,784,10,activation_function=tf.nn.softmax)

#the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                              reduction_indices=[1])) #loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess=tf.Session()
# important step
sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%50 == 0:
        print(computer_accuracy(mnist.test.images,mnist.test.labels))