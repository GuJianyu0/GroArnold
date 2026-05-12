# -*- coding: utf-8 -*-
# !/usr/bin/env python

import tensorflow as tf
# tf.enable_eager_execution() #old version
import numpy as np
x=[5,2,0]
z=[b'cat', b'dog', b'chicken']
first=tf.train.Feature(int64_list=tf.train.Int64List(value=x))
second=tf.train.Feature(bytes_list=tf.train.BytesList(value=z))
dic={'first':first,'second':second}
fea=tf.train.Features(feature=dic)
a=tf.train.Example(features=fea)
print(a)
