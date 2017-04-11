#coding:utf-8
import os
os.environ['THEANO_FLAGS'] = "device=gpu"  # 改变环境变量添加THEANO_FLAGS,在导入theano的时候可以设置使用gpu
import numpy as np
import theano as theano
import theano.tensor as T
from theano.gradient import grad_clip
import time
import operator

theano.config.floatX = 'float32' # float32 才能使用GPU

class GRUTheano:

    def __init__(self, word_dim, hidden_dim=128, bptt_truncate=-1):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Initialize the network parameters
        E = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        U = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (6, hidden_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (6, hidden_dim, hidden_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        b = np.zeros((6, hidden_dim))
        c = np.zeros(word_dim)
        # Theano: Created shared variables
        self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))
        # SGD / rmsprop: Initialize parameters
        self.mE = theano.shared(name='mE', value=np.zeros(E.shape).astype(theano.config.floatX))
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        E, V, U, W, b, c = self.E, self.V, self.U, self.W, self.b, self.c

        x = T.ivector('x')
        y = T.ivector('y')

        def forward_prop_step(x_t, s_t1_prev, s_t2_prev):
            # This is how we calculated the hidden state in a simple RNN. No longer!
            # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))

            # Word embedding layer
            x_e = E[:,x_t]

            # GRU Layer 1
            z_t1 = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_t1_prev) + b[0])
            r_t1 = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_t1_prev) + b[1])
            c_t1 = T.tanh(U[2].dot(x_e) + W[2].dot(s_t1_prev * r_t1) + b[2]) # *乘 表示两个向量对应位置相乘,结果不改变维度
            s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev

            # GRU Layer 2
            z_t2 = T.nnet.hard_sigmoid(U[3].dot(s_t1) + W[3].dot(s_t2_prev) + b[3])
            r_t2 = T.nnet.hard_sigmoid(U[4].dot(s_t1) + W[4].dot(s_t2_prev) + b[4])
            c_t2 = T.tanh(U[5].dot(s_t1) + W[5].dot(s_t2_prev * r_t2) + b[5])
            s_t2 = (T.ones_like(z_t2) - z_t2) * c_t2 + z_t2 * s_t2_prev

            # Final output calculation
            # Theano's softmax returns a matrix with one row, we only need the row
            o_t = T.nnet.softmax(V.dot(s_t2) + c)[0]

            return [o_t, s_t1, s_t2]
            # 上面的return方式与下面 scan函数的outputs_info参数有关。
            # None表示不会对这个位置的输出做任何循环使用，即不使用 tap
            # 输出出现在一个封装的字典中，scan函数会按照输出的最后一步来封装它，即tap值为 -1
            # 即下面循环中，函数第一参数使用x中的值，
            # 第二个、第三个参数第一个loop使用初始化的值，之后循环使用上一步对应位置的输出，来作为输入

        [o, s, s2], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            truncate_gradient=self.bptt_truncate,   # bptt的截断步数
            outputs_info=[None,
                          dict(initial=T.zeros(self.hidden_dim)),   #初始化 s_t1_prev
                          dict(initial=T.zeros(self.hidden_dim))])  #初始化 s_t2_prev

        prediction = T.argmax(o, axis=1)    #axis=1 是横着
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))

        # Total cost (could add regularization here)
        cost = o_error

        # Gradients
        dE = T.grad(cost, E)
        dU = T.grad(cost, U)
        dW = T.grad(cost, W)
        db = T.grad(cost, b)
        dV = T.grad(cost, V)
        dc = T.grad(cost, c)

        # Assign functions
        # 根据序列 x ，生成序列 o，o是一个矩阵(len(x)*word_dim)，每一行代表各个词出现的概率
        self.predict = theano.function([x], o)
        # 跟上面函数相似，但是直接返回，可能性最大的序列
        self.predict_class = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], cost)
        self.bptt = theano.function([x, y], [dE, dU, dW, db, dV, dc])

        # SGD parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')

        # rmsprop cache updates
        mE = decay * self.mE + (1 - decay) * dE ** 2
        mU = decay * self.mU + (1 - decay) * dU ** 2
        mW = decay * self.mW + (1 - decay) * dW ** 2
        mV = decay * self.mV + (1 - decay) * dV ** 2
        mb = decay * self.mb + (1 - decay) * db ** 2
        mc = decay * self.mc + (1 - decay) * dc ** 2

        self.sgd_step = theano.function(
            [x, y, learning_rate, theano.Param(decay, default=0.9)],
            [],
            updates=[(E, E - learning_rate * dE / T.sqrt(mE + 1e-6)),  # 1e-6 = 1*10^(6) =0.000001
                     (U, U - learning_rate * dU / T.sqrt(mU + 1e-6)),
                     (W, W - learning_rate * dW / T.sqrt(mW + 1e-6)),
                     (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
                     (b, b - learning_rate * db / T.sqrt(mb + 1e-6)),
                     (c, c - learning_rate * dc / T.sqrt(mc + 1e-6)),
                     (self.mE, mE),
                     (self.mU, mU),
                     (self.mW, mW),
                     (self.mV, mV),
                     (self.mb, mb),
                     (self.mc, mc)
                    ])


    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])

    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)
