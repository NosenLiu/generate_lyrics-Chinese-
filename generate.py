#coding:utf-8

""" 生成歌词模块
    需要设定 lyric_document = 专辑名
            parameter_document  使用的参数文件，即经过train.py训练得到的 .npz文件
            LYRIC_NUMBER   生成的歌词数量（句）(实际输出小于次数目，因为有部分生成失败的句子被剔除)
            MIN_LENGTH     每句话最短的词语数量（词语少于此数量，这句话会被剔除）
    生成的歌词文件在lyrics文件夹内
"""

__author__ = 'Yunpeng Liu'

import sys
import os
#os.environ['THEANO_FLAGS'] = "device=gpu"  # 改变环境变量添加THEANO_FLAGS,在导入theano的时候可以设置使用gpu
import time
import numpy as np
from datetime import datetime
from gru_theano import GRUTheano

SENTENCE_START_TOKEN = "SENTENCE_START"
SENTENCE_END_TOKEN = "SENTENCE_END"
UNKNOWN_TOKEN = "UNKNOWN_TOKEN"
lyric_document = 'Fantasy'   # 歌词文件名 TODO: 人工设定(根据专辑名称选定)
parameter_document = 'GRU-Fantasy-2017-04-10-16-04-700-48-128.dat.npz'
# 该变量控制生成歌词由多少句话组成
LYRIC_NUMBER = 40
MIN_LENGTH = 4
SONG_NAME = u'%s_生成'%(lyric_document)

# 模型变量存取函数
def load_model_parameters_theano(path, modelClass=GRUTheano):
    npzfile = np.load(path)
    E, U, W, V, b, c = npzfile["E"], npzfile["U"], npzfile["W"], npzfile["V"], npzfile["b"], npzfile["c"]
    hidden_dim, word_dim = E.shape[0], E.shape[1]
    print "Building model model from %s with hidden_dim=%d word_dim=%d" % (path, hidden_dim, word_dim)
    sys.stdout.flush()
    model = modelClass(word_dim, hidden_dim=hidden_dim)
    model.E.set_value(E)
    model.U.set_value(U)
    model.W.set_value(W)
    model.V.set_value(V)
    model.b.set_value(b)
    model.c.set_value(c)
    return model

# 依据训练记录的文件，建立模型  TODO: 每次从data文件夹中找对应的训练后文件
model = load_model_parameters_theano('./data/%s'%(parameter_document))

# 依据分词记录的文件，获取分词之后的结果变量，需要建dict类型的word2num
npzfile = np.load('./data/%s_format_sen.npz'%(lyric_document))
num2word = npzfile['num2word']
sentence_set = npzfile['sentence_set']
word_trash = npzfile['word_trash']
word2num = {}
for i in range(len(num2word)):
    word2num[num2word[i]] = i
# word2num,num2word 建立完成

# TODO: 生成句子工作
def generate_sentence(model, num2word, word2num, min_length=5):
    # We start the sentence with the start token
    new_sentence = [word2num[SENTENCE_START_TOKEN]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word2num[SENTENCE_END_TOKEN]:
        #print len(model.predict(new_sentence)[-1])
        #print sum(model.predict(new_sentence)[-1])
        next_word_probs = model.predict(new_sentence)[-1]  # 预测新序列的最后一位即为预测位
        next_word_probs = next_word_probs*0.9999   # 排除因精度引起的预测位所有概率和大于1的问题
        # 掷骰子函数, 投掷一次， 可能性向量为 next_word_probs, 保证生成的语句具有变化性
        samples = np.random.multinomial(1, next_word_probs)
        sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
        # Seomtimes we get stuck if the sentence becomes too long, e.g. "........" :(
        # And: We don't want sentences with UNKNOWN_TOKEN's
        if len(new_sentence) > 100 or sampled_word == word2num[UNKNOWN_TOKEN]:
            return None
    #句子太短，也舍弃
    if len(new_sentence) < min_length:
        return None
    sentence_str = [num2word[x] for x in new_sentence[1:-1]]
    return sentence_str

# 生成歌词并写入文件 TODO:不知道Unicode编码的汉字在文件里会不会有问题。
def generate_lyric():
    lyric = []
    for i in range(LYRIC_NUMBER):
        lyric.append(generate_sentence(model, num2word, word2num, min_length=MIN_LENGTH))
    with open('./lyrics/'+SONG_NAME+'.txt','wb') as f:   #  按二进制文件读写('wb')，否则会默认ASCII编码
        for i in lyric:
            if i != None:
                for j in i:
                    f.write(j.encode('utf-8'))  # 写入之前先encode，否则会报错
                f.write(u'\t\n')
        #f.write(lyric)
    f.close()

# 测试代码
#temp_s = generate_sentence(model, num2word, word2num, min_length=5)
#for i in temp_s:
#    print i

if __name__ == '__main__':
    generate_lyric()
