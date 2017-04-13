# generate_lyrics-Chinese-
generate Chinese lyrics using Recurrent Neural Network.
"!! Caution !! The content in folder 'stanford-segmenter-2014-08-27' is not enough, because two big files(more than 100M) could not be upload. Please download stanford-segmenter-2014-08-27.zip and release the documents instead of this folder. URL http://nlp.stanford.edu/software/stanford-segmenter-2014-08-27.zip"


【注：根目录下的stanford-segmenter-2014-08-27文件夹里面内容不充分，缺少两个100M以上的字典文件。请自行下载stanford-segmenter-2014-08-27.zip解压并代替掉文件夹stanford-segmenter-2014-08-27。 URL http://nlp.stanford.edu/software/stanford-segmenter-2014-08-27.zip 】

# utf-8

歌词生成器（中文）
使用斯坦福分词器进行分词，
神经网络训练部分是使用的是Denny Britz 的双层GRU结构的循环神经网络。（感谢Denny先生在博客中的细心讲解，帮我完成了作业）
( http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/ )  


环境： python 2.7 及常用科学计算包
需要 cuda GPU加速（不加速也可以，训练速度会慢一些）


使用之前首先要录入歌词文件，手动在data文件夹中建立csv文件，并保存为utf-8格式。同一首歌每一句歌词中间不要空行
先用word_segment.py 分词
再使用train.py 进行训练
最后使用generate.py 生成歌词并保存

