#coding:utf-8
#RNN LSTM 分类例子
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#this is data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#hyperparameters
lr = 0.001
training_iters = 100000
batch_size = 128


n_inputs= 28 #MNIST data input (img shape:28*28)
n_step = 28 #time step
n_hidden_unis = 128 #neurons in hidden layer
n_classes = 10 #MNIST classes (0-9 digits)

#tf Graph input
x = tf.placeholder(tf.float32,[None,n_step,n_inputs])
y = tf.placeholder(tf.float32,[None,n_classes])

#Define weights
weights = {
    #(28,128)
    'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_unis])),
    #(128,10)
    'out':tf.Variable(tf.random_normal([n_hidden_unis,n_classes]))
}

#Define biases
biases = {
    #(128,)
    'in':tf.Variable(tf.constant(0.1,shape=[ n_hidden_unis,])),
    #(10,)
    'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))
}

def RNN(X,weights,biases):
    #hidden layer for input to cell
    #X shape(128 batch,28steps ,28 inputs) ==>(128*28,28)
    # X = tf.reshape(X,[-1,n_inputs])
    # ==>(128batch*28steps,128 hidden)
    # X_in= tf.matmul(X,weights['in'])+biases['in']
    # ==>(128batch,28steps,128hidden)
    # X_in = tf.reshape(X_in,[-1,n_step,n_hidden_unis])
    #cell is divide into two parts (c_states,m_states)
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_unis,forget_bias = 1.0,state_is_tuple=True)
    _init_state =lstm_cell.zero_state(batch_size,dtype=tf.float32)
    #states is divide into two parts (c_states,m_states)
    output,states = tf.nn.dynamic_rnn(lstm_cell,X,initial_state=_init_state,time_major=False)
     #hidden layer for output as the final result
    result = tf.matmul(states[1],weights['out'])+biases['out']

    #or
    # unpack to list[(batch,outputs)......]*steps
    #outputs = tf.unpack(tf.transpose(outputs,[1,0,2])) #states is the last output
    #result = tf.matmul(outputs[-1],weights['out'])+biases['out']

    return result


pred = RNN(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(None,y,pred))
train_op =tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step*batch_size < training_iters:
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size,n_step,n_inputs])
        sess.run([train_op],feed_dict={
            x:batch_xs,
            y:batch_ys
        })
        if step%20 == 0:
            print(sess.run(accuracy,feed_dict={
                x:batch_xs,
                y:batch_ys
            }))