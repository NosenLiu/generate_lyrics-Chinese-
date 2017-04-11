#coding:utf-8

""" 分词模块
    使用斯坦福分词器
    需要设定 lyric_document = 专辑名
            volume_of_word  词汇量， 如词汇量=600， 即保留出现频率最多的,600个词汇。
    分词得到的参数文件（.npz）在data文件夹内:     '专辑名_format_sen.npz'
"""

__author__ = 'Yunpeng Liu'

import pandas as pd
import os
import numpy as np
from nltk.tokenize.stanford_segmenter import StanfordSegmenter

# 初始化segmenter
segmenter = StanfordSegmenter(path_to_jar="./stanford-segmenter-2014-08-27/stanford-segmenter-3.4.1.jar",
    path_to_sihan_corpora_dict="./stanford-segmenter-2014-08-27/data",
    path_to_model="./stanford-segmenter-2014-08-27/data/pku.gz",
    path_to_dict="./stanford-segmenter-2014-08-27/data/dict-chris6.ser.gz")

# 3类词语的特殊标签
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
volume_of_word = 700        # 需要进行学习的词汇量  TODO: 人工设定
lyric_document = 'Fantasy'   # 歌词文件名 TODO: 人工设定

# 词语和句子向量中 数字的 相互转化向量
word2num = {}
num2word = []
word_dict = {}

lyrics = pd.read_csv('./data/%s.csv'%(lyric_document), encoding='utf-8', index_col='title')   #

song_num = len(lyrics['lyric'])  # 统计歌曲数目
title_set = lyrics.index[0]    # 初始化歌名集 和 歌词集
song_set = lyrics['lyric'][0]
for i in range(song_num):
    if i != 0:
        title_set = title_set + u'\n' + lyrics.index[i]
        song_set = song_set + u'\n' +lyrics['lyric'][i]

title_result = segmenter.segment(title_set)
song_result = segmenter.segment(song_set)

sentences = []
count_num = 0

#将每句话放入sentences[]中
temp = ''
for i in song_result:
    if i != u'\n':
        temp += i
    else:
        temp += i
        sentences.append(temp)
        temp = ''
        count_num += 1

#分解sentences[]中的词放入字典
temp = ''
for sentence in sentences:
    for word in sentence:
        if word != u' ' and word != u'\n' and word != u'\r':
            temp += word
        elif word == u' ' or word == u'\r':
            if temp in word_dict:
                word_dict[temp] = word_dict[temp] + 1  # 字典中计数 + 1
                temp = ''
            else:
                word_dict[temp] = 1
                temp = ''
        else:
            pass

#将dict中的一对键、值转为tuple，输出为List类型
word_dict = sorted(word_dict.iteritems(),key = lambda tt:tt[1], reverse = True)
print u'共出现：%s个词汇！'%(len(word_dict))
print u'出现最多的词：%s。 共出现：%s次！'%(word_dict[0][0],word_dict[0][1])
if volume_of_word > len(word_dict):
    print u"volume_of_word 设置错误，大于词汇总数量"
word_trash = word_dict[volume_of_word-3:] # 存放被遗弃的词汇 List套 tuple格式
word_dict = word_dict[:volume_of_word-3]

# 字符到数字的 转换向量，dict格式。 根据汉字获得对应的数字
word2num = {unknown_token:0,sentence_start_token:1,sentence_end_token:2}
for i in range(len(word_dict)):
    word2num[word_dict[i][0]] = i+3
# 数字到字符的 转换向量，List格式。 根据list下标获取对应汉字
num2word = [unknown_token,sentence_start_token,sentence_end_token]
for word in word_dict:
    num2word.append(word[0])

print u'词库容量：%s 。'%(len(word2num))

# 分解句子成为数字向量
sentence_set = []
temp_s = []
temp = ''
for sentence in sentences:
    temp_s.append(word2num[sentence_start_token])
    for word in sentence:
        if word != u' ' and word != u'\n' and word != u'\r':
            temp += word
        elif word == u' ' or word == u'\r':
            if temp in word2num:
                temp_s.append(word2num[temp])
                temp = ''
            else:
                temp_s.append(word2num[unknown_token])
                temp = ''
        else:
            temp_s.append(word2num[sentence_end_token])
            temp = ''
    sentence_set.append(temp_s)
    temp_s = []
print u'一共有句子：%s 个。'%(len(sentence_set))

#  TODO: 要存4个变量，word2num, num2word, sentence_set, word_trash

#with open('./data/word2num.txt', 'wb') as f:
#    f.write(word2num)   # TODO 存不了dict

np.savez('./data/%s_format_sen.npz'%(lyric_document),
    num2word=num2word,
    sentence_set=sentence_set,
    word_trash=word_trash)
