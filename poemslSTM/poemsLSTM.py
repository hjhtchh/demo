#coding:utf-8
import collections
import tensorflow as tf
import numpy as np
import codecs
#---------------------数据预处理----------------------#

#数据的读取
#txt中每一行是一首诗，其中题目与内容通过 ： 隔开
poems_path = './data/poetry.txt'
poetries = []
with codecs.open(poems_path,'r',encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        try:
            # line = line.decode('utf-8')
            line = line.strip('\n')
            title,content = line.strip(' ').split(':')
            content = content.replace(' ','')
            if '_'in content or '(' in content or '《' in content or '[' in content:
                continue
            if len(content)< 5 or len(content)>79:
                continue
            content = '['+content+']'
            poetries.append(content)
        except Exception as e:
            pass

#按照诗的字数进行排序
poetries = sorted(poetries,key=lambda line:len(line))
#预计可用于训练的唐诗数量为34692
# print('唐诗的总数为：',len(poetries))

 #统计每个词出现的次数
all_words = []
for poetry in poetries:
    all_words += [word for word in poetry]
counter = collections.Counter(all_words)
count_pairs = sorted(counter.items(),key=lambda x:-x[1])
words,_ = zip(*count_pairs)
# print(words)

#取前多少个常用字
words = words[:len(words)]+(' ',)
#每个字映射成一个数字id
word_num_map = dict(zip(words,range(len(words))))
# print(word_num_map)
#把诗歌转换成为向量形式
to_num = lambda word : word_num_map.get(word,len(words))
poetries_vector = [list(map(to_num,poetry)) for poetry in poetries]
# print(poetries_vector)


#每次取64首古诗进行训练
#当使用模型进行诗词生成的时候，要将batch_size 改为1
batch_size = 1   #or 64
n_chunk = len(poetries_vector)//batch_size


class DataSet(object):
    def __init__(self,data_size):
        self._data_size = data_size
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._data_index = np.arange(data_size)

    def next_batch(self,batch_size):
        start = self._index_in_epoch
        if start+batch_size > self._data_size:
            np.random.shuffle(self._data_index)
            self._epochs_completed = self._epochs_completed+1
            self._index_in_epoch = batch_size
            full_batch_features,full_batch_labels = self.data_batch(0,batch_size)
            return full_batch_features,full_batch_labels
        else:
            self._index_in_epoch+=batch_size
            end = self._index_in_epoch
            full_batch_features,full_batch_labels = self.data_batch(start,end)
            if self._index_in_epoch == self._data_size:
                self._index_in_epoch = 0
                self._epochs_completed = self._epochs_completed+1
                np.random.shuffle(self._data_index)
            return full_batch_features,full_batch_labels

    def data_batch(self,start,end):
        batches=[]
        for i in range(start,end):
            batches.append(poetries_vector[self._data_index[i]])

        length = max(map(len,batches))

        xdata = np.full((end-start,length),word_num_map[' '],np.int32)
        for row in range(end-start):
            xdata[row,:len(batches[row])] = batches[row]
        ydata = np.copy(xdata)
        ydata[:,:-1] = xdata[:,1:]
        return xdata,ydata


#------------------------------RNN-------------------------------------#

input_data = tf.placeholder(tf.int32,[batch_size,None])
output_targets = tf.placeholder(tf.int32,[batch_size,None])

#定义RNN结构
def neural_network(model='lstm',rnn_size=128,num_layers=2):
    if model == 'rnn':
        cell_fun = tf.nn.rnn_cell.BasicRNNCell
    elif model =='gru':
        cell_fun = tf.nn.rnn_cell.GRUCell
    elif model == 'lstm':
        cell_fun = tf.nn.rnn_cell.BasicLSTMCell

    cell = cell_fun(rnn_size,state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell]*num_layers,state_is_tuple=True)

    initial_state = cell.zero_state(batch_size,tf.float32)

    with tf.variable_scope('rnnlm'):
        softmax_w = tf.get_variable('softmax_w',[rnn_size,len(words)])
        softmax_b = tf.get_variable('softmax_b',[len(words)])
        with tf.device("/cpu:0"):
            embedding = tf.get_variable('embedding',[len(words),rnn_size])
            inputs = tf.nn.embedding_lookup(embedding,input_data)


    outputs,last_state = tf.nn.dynamic_rnn(cell,inputs,initial_state=initial_state,scope='rnnlm')
    output = tf.reshape(outputs,[-1,rnn_size])

    logits = tf.matmul(output,softmax_w)+softmax_b
    probs = tf.nn.softmax(logits)
    return logits,last_state,probs,cell,initial_state


def load_model(sess,saver,ckpt_path):
    latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
    if latest_ckpt:
        print('resume from',latest_ckpt)
        saver.restore(sess,latest_ckpt)
        return int(latest_ckpt[latest_ckpt.rindex('-')+1:])
    else:
        print('building model from scratch')
        sess.run(tf.global_variables_initializer())
        return -1

#训练模型
def train_neural_network():
    logits,last_state,_,_,_ = neural_network()
    targets = tf.reshape(output_targets,[-1])
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],[targets],[tf.ones_like(targets,dtype=tf.float32)])
    cost = tf.reduce_mean(loss)
    learning_rate = tf.Variable(0.0,trainable=False)
    tvars = tf.trainable_variables()
    grads,_ = tf.clip_by_global_norm(tf.gradients(cost,tvars),5)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(grads,tvars))

    Session_config = tf.ConfigProto(allow_soft_placement=True)
    Session_config.gpu_options.allow_growth = True

    trainds = DataSet(len(poetries_vector))


    with tf.Session(config=Session_config) as sess:
        with tf.device("/gpu:2"):
            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver(tf.all_variables())
            last_epoch = load_model(sess,saver,'model/')


            for epoch in range(last_epoch+1,100):
                sess.run(tf.assign(learning_rate,0.002*(0.97**epoch)))

                all_loss = 0.0
                for batche in range(n_chunk):
                    x,y = trainds.next_batch(batch_size)
                    train_loss,_,_ = sess.run([cost,last_state,train_op],
                                              feed_dict={input_data:x,output_targets:y})
                    all_loss = all_loss+train_loss

                    if batche %50 == 1:
                        print(epoch,batche,0.002*(0.97**epoch),train_loss)

                saver.save(sess,'model/poetry.module',global_step=epoch)
                print(epoch,'Loss:',all_loss*1.0/n_chunk)

# train_neural_network()

#--------------------------------生成古诗--------------------------------#
def gen_poetry():
    def to_word(weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        sample = int(np.searchsorted(t, np.random.rand(1) * s))
        return words[sample]

    _, last_state, probs, cell, initial_state = neural_network()
    Session_config = tf.ConfigProto(allow_soft_placement=True)
    Session_config.gpu_options.allow_growth = True

    with tf.Session(config=Session_config) as sess:
        with tf.device('/gpu:1'):
            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver(tf.all_variables())
            saver.restore(sess, 'model/poetry.module-99')

            state_ = sess.run(cell.zero_state(1, tf.float32))

            x = np.array([list(map(word_num_map.get, '['))])
            [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
            word = to_word(probs_)
            # word = words[np.argmax(probs_)]
            poem = ''
            while word != ']':
                poem += word
                x = np.zeros((1, 1))
                x[0, 0] = word_num_map[word]
                [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
                word = to_word(probs_)
                # word = words[np.argmax(probs_)]
            return poem


# print(gen_poetry())


#-----------------------------------------------------生成藏头诗-------------------------------------------------------#
def gen_head_poetry(heads,type):
    if type != 5 and type != 7:
        print('the second param has to be 5 or 7！')
        return
    def to_word(weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        sample = int(np.searchsorted(t,np.random.rand(1)*s))
        return words[sample]

    _,last_state,probs,cell,initial_state = neural_network()
    Session_config = tf.ConfigProto(allow_soft_placement=True)
    Session_config.gpu_options.allow_growth = True

    with tf.Session(config=Session_config) as sess:
        with tf.device('/gpu:1'):
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver(tf.all_variables())
            saver.restore(sess,'model/poetry.module-99')
            poem = ''
            for head in heads:
                flag =True
                while flag:
                    state_ = sess.run(cell.zero_state(1,tf.float32))
                    x = np.array([list(map(word_num_map.get,u'['))])
                    [probs_,state_] = sess.run([probs,last_state],feed_dict={input_data:x,initial_state:state_})

                    sentence = head

                    x = np.zeros((1,1))
                    x[0,0] = word_num_map[sentence]
                    [probs_,state_] = sess.run([probs,last_state],feed_dict={input_data:x,initial_state:state_})
                    word = to_word(probs_)
                    sentence +=word

                    while word !=u'。':
                        x = np.zeros((1,1))
                        x[0,0] = word_num_map[word]
                        [probs_, state_] = sess.run([probs, last_state],feed_dict={input_data: x, initial_state: state_})
                        word = to_word(probs_)
                        sentence +=word

                    if len(sentence) == 2+2*type:
                        sentence +=u'\n'
                        poem +=sentence
                        flag =False

            return poem



print(gen_head_poetry(u'谭志浩美',type=7))






