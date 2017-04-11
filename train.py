#coding:utf-8


""" 训练模块
    需要设定 lyric_document = 专辑名
            LEARNING_RATE  学习率
            HIDDEN_DIM     隐藏层节点数
            NEPOCH         训练循环次数
            BPTT_STEPS     反向传播训练步数
            PRINT_EVERY    每训过多少个样本（句子）计算打印Loss
    训练得到的参数文件（.npz）在data文件夹内:   'GRU-专辑名-年-月-日-XXXXXXXX.npz'
"""

__author__ = 'Yunpeng Liu'

import sys
import os
os.environ['THEANO_FLAGS'] = "device=gpu"  # 改变环境变量添加THEANO_FLAGS,在导入theano的时候可以设置使用gpu
import time
import numpy as np
from datetime import datetime
from gru_theano import GRUTheano

#theano.config.floatX = 'float32' # float32 才能使用GPU

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.002"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "128"))
NEPOCH = int(os.environ.get("NEPOCH", "100"))
BPTT_STEPS = int(os.environ.get("BPTT_STEPS", "8"))
MODEL_OUTPUT_FILE = os.environ.get("MODEL_OUTPUT_FILE")
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "20"))
lyric_document = 'Fantasy'   # 歌词文件名 TODO: 人工设定

# 获取分词之后的结果变量，需要建dict类型的word2num
npzfile = np.load('./data/%s_format_sen.npz'%(lyric_document))
num2word = npzfile['num2word']
sentence_set = npzfile['sentence_set']
word_trash = npzfile['word_trash']
word2num = {}
for i in range(len(num2word)):
    word2num[num2word[i]] = i
# word2num,num2word 建立完成

# 构造训练的输入，输出(即输入向右移动一位)
x_train = []
y_train = []
for i in sentence_set:
    x_train.append(i[:-1])
    y_train.append(i[1:])
# 构造完毕，训练输入 x_train, 对应训练输出 y_train

# 模型变量存取函数, 取值函数在generate.py中使用
def save_model_parameters_theano(model, outfile):
    np.savez(outfile,
        E=model.E.get_value(),
        U=model.U.get_value(),
        W=model.W.get_value(),
        V=model.V.get_value(),
        b=model.b.get_value(),
        c=model.c.get_value())
    print "Saved model parameters to %s." % outfile


if not MODEL_OUTPUT_FILE:
  ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
  MODEL_OUTPUT_FILE = "./data/GRU-%s-%s-%s-%s.dat" % (lyric_document,ts, len(num2word), HIDDEN_DIM)

# 建立模型，第一个参数是 word_dim，即词汇数量
# 第二个参数，隐藏层节点数。 第三个参数，训练时向后回溯的步数。
model = GRUTheano(len(num2word), hidden_dim=HIDDEN_DIM, bptt_truncate=BPTT_STEPS)

"""
#这几行代码测试执行一步训练需要多少时间。
# Print SGD step time
t1 = time.time()
model.sgd_step(x_train[10], y_train[10], LEARNING_RATE)
t2 = time.time()
print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)
sys.stdout.flush()
"""


# We do this every few examples to understand what's going on
def sgd_callback(model, num_examples_seen):
    dt = datetime.now().isoformat()
    loss = model.calculate_loss(x_train[:10000], y_train[:10000])
    print("\n%s (%d)" % (dt, num_examples_seen))
    print("--------------------------------------------------")
    print("Loss: %f" % loss)
    # 下面一行是生成语句函数，之后的py文件再补上
    #generate_sentences(model, 10, index_to_word, word_to_index)
    save_model_parameters_theano(model, MODEL_OUTPUT_FILE)
    print("\n")
    sys.stdout.flush()

def train_with_sgd(model, X_train, y_train, learning_rate=0.001, nepoch=20, decay=0.9,
    callback_every=10000, callback=None):
    num_examples_seen = 0
    for epoch in range(nepoch):
        # For each training example... 下面的permutation()函数将句子打乱,返回一个乱序的编号(无重复)。
        for i in np.random.permutation(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate, decay)
            num_examples_seen += 1   #每看一句话，该变量+1
            # Optionally do callback
            if (callback and callback_every and num_examples_seen % callback_every == 0):
                callback(model, num_examples_seen)
    return model

for epoch in range(NEPOCH):
  train_with_sgd(model, x_train, y_train, learning_rate=LEARNING_RATE, nepoch=1, decay=0.9,
    callback_every=PRINT_EVERY, callback=sgd_callback)
