# -*- coding: utf-8 -*-
#/usr/bin/python
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
class Hyperparams:
    '''Hyperparameters'''
    # data
    source_train = 'corpora/train.source-target.source'
    target_train = 'corpora/train.source-target.target'
    source_test = 'corpora/test.source-target.source'
    target_test = 'corpora/test.source-target.target'

    
    # training
    batch_size = 32 # alias = N
    lr = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir' # log directory
    
    # model
    maxlen = 30 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 10 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512 # alias = C
    num_blocks = 6 # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False # If True, use sinusoid. If false, positional embedding.


