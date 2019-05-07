#coding:utf-8
#可视化神经网络

import tensorflow as tf
import numpy as np

#层的构造
def add_layer(input,in_size,out_size,n_layer,activation_function=None):
    layer_name = 'layer%s'%n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
            tf.summary.histogram(layer_name+'/weights',Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1)
            tf.summary.histogram(layer_name+'/biases',biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(input,Weights)+biases
        if activation_function is None:
            output = Wx_plus_b
        else:
            output = activation_function(Wx_plus_b)
            tf.summary.histogram(layer_name+'/output',output)
        return  output
#数据构造
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)-0.5+noise
with tf.name_scope('inputs'):
    xs =tf.placeholder(tf.float32,[None,1],name='x_inputs')
    ys =tf.placeholder(tf.float32,[None,1],name='y_inputs')

feed ={xs:x_data,ys:y_data}
#定义隐藏层
l1 = add_layer(xs,1,10,n_layer=1,activation_function=tf.nn.relu)
#定义输出层
prediction = add_layer(l1,10,1,n_layer=2,activation_function=None)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                     reduction_indices=[1]))
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#变量初始化
sess =tf.Session()
merged = tf.summary.merge_all()
writer =tf.summary.FileWriter('logs/',sess.graph)
sess.run(tf.initialize_all_variables())

for i in range(1000):
    sess.run(train_step,feed_dict=feed)
    if i%50==0:
        result = sess.run(merged,
                          feed_dict=feed)
        writer.add_summary(result,i)
