#coding:utf-8
#建造神经网络

import tensorflow as tf
import numpy as np
from TensorFlowDemo import TFdemo5
import matplotlib.pyplot as plt #结果可视化
#数据构造
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)-0.5+noise

xs =tf.placeholder(tf.float32,[None,1])
ys =tf.placeholder(tf.float32,[None,1])

feed ={xs:x_data,ys:y_data}
#定义隐藏层
l1 = TFdemo5.add_layer(xs,1,10,activation_function=tf.nn.relu)
#定义输出层
prediction = TFdemo5.add_layer(l1,10,1,activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                     reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#变量初始化
init = tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()
for i in range(1000):
    #train
    sess.run(train_step,feed_dict=feed)
    if i%50 == 0:
        #to see the step improvement
        # print(sess.run(loss,feed_dict=feed))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction,feed_dict=feed)
        lines = ax.plot(x_data,prediction_value,'r-',lw=5)
        plt.pause(0.1)



