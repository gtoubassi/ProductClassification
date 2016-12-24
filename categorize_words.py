#!/usr/bin/env python
#
import argparse
import database
import random
import hashlib
import os
import numpy as np
import re
import operator
import tensorflow as tf
import tflearn
import math

def normalizeText(t):
    t = re.sub('<.>|<..>', ' ', t.lower())
    return re.sub('[^a-z0-9 ]', '', t)

def computeTFDataForProducts(products, vocab_indices, category_indices, seenCategories):
    xs = []
    ys = []
    for p in products:
        text = normalizeText(p['name'] + ' ' + p['description'])
        words = text.split()
        bow = np.zeros(len(vocab_indices))
        for w in words:
            if w in vocab_indices:
                bow[vocab_indices[w]] = 1
        y = np.zeros(len(seenCategories))
        y[category_indices[p['category_id']]] = 1
        xs.append(bow)
        ys.append(y)
    return xs, ys
    
def classifyText(categories):
    if len(categories) <= 10:    
        name = ",".join(categories)
    else:
        name = ",".join(categories[0:5]) + ("... (%d total)" % + len(categories))
    experimentId = db.addExperiment("Text classification of: %s" % name)

    products = db.getProducts(categories)
    vocab = {}
    seenCategories = []
    for p in products:
        seenCategories.append(p['category_id'])
        text = normalizeText(p['name'] + ' ' + p['description'])
        words = text.split()
        for w in words:
            vocab[w] = 1 if w not in vocab else vocab[w]+1
    sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1))
    vocab_indices = dict((v,k) for k,v in enumerate([w[0] for w in sorted_vocab[-20000:]]))
    category_indices = dict((v,k) for k,v in enumerate(seenCategories))

    random.shuffle(products)
    
    if False:    
        net = tflearn.input_data(shape=[None, len(vocab_indices)])
        net = tflearn.fully_connected(net, 100)
        net = tflearn.fully_connected(net, len(seenCategories), activation='softmax')
        net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

        model = tflearn.DNN(net)
    
        X,Y = computeTFDataForProducts(products, vocab_indices, category_indices, seenCategories)
        model.fit(X, Y, validation_set=.1, n_epoch=1000, batch_size=32, show_metric=True)

    else:
        
        x = tf.placeholder(tf.float32, shape=[None, len(vocab_indices)])
        
        num_hidden_layers = 100
        
        stdv1 = 1.0 / math.sqrt(len(vocab_indices))
        w1 = tf.Variable(tf.random_uniform([len(vocab_indices), num_hidden_layers], minval=-stdv1, maxval=stdv1))
        b1  = tf.Variable(tf.random_uniform([num_hidden_layers], minval=-stdv1, maxval=stdv1))
        h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
        
        stdv2 = 1.0 / math.sqrt(num_hidden_layers)
        w2 = tf.Variable(tf.random_uniform([num_hidden_layers, len(seenCategories)], minval=-stdv2, maxval=stdv2))
        b2  = tf.Variable(tf.random_uniform([len(seenCategories)], minval=-stdv2, maxval=stdv2))

        y_logits = tf.matmul(h1, w2) + b2
        y = tf.nn.softmax(y_logits)
        
        y_ = tf.placeholder(tf.float32, [None, len(seenCategories)])

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_logits, y_))
        train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        prediction = tf.argmax(y, 1)
        
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        train = products[:(int(.9*len(products)))]
        test = products[len(train):]

        test_x, test_y = computeTFDataForProducts(test, vocab_indices, category_indices, seenCategories)

        # Train

        for _ in range(5):
            random.shuffle(train)

            print("eval (N=%d)" % len(test_x))
            correct = 0
            for i, t in enumerate(test_x):
                guess = sess.run(prediction, feed_dict={x:np.reshape(t, [1, len(t)])})
                if guess[0] == np.argmax(test_y[i]):
                    correct += 1
            print("Calculated accuracy: %f" % (float(correct)/len(test_x)))

            print "train"
            for i in xrange(0, len(train), 32):
                batch_xs, batch_ys = computeTFDataForProducts(train[i:(i+32)], vocab_indices, category_indices, seenCategories)
              
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
                if i % 100 == 0:
                    print "batch offset %d of %d" % (i, len(train))

        # Record experiment results
        correct = 0
        for i, p in enumerate(test):
            guess = sess.run(y, feed_dict={x:np.reshape(test_x[i], [1, len(test_x[i])])})
            guess = np.squeeze(guess)
            predictedCat = seenCategories[np.argmax(guess)]
            if p['category_id'] == predictedCat:
                correct += 1
            db.addPredictedCategory(experimentId, p['id'], predictedCat, np.max(guess))
        print("Calculated accuracy: %f" % (float(correct)/len(test)))

def dumpTextCorpus():
    products = db.getProducts()
    for p in products:
        print "%s %s" % (normalizeText(p['name']), normalizeText(p['description']))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default='crawl.db', help="Path to sqlite db file")
    parser.add_argument("--images-path", default='images', help="Path to directory in which images should be saved")
    args = parser.parse_args()

    global db
    db = database.Database(args.db_path)
    
    classifyText([])
    #classifyText(['skinny-jeans', 'bootcut-jeans'])

    #classifyText(['two-piece-swimsuits','sunglasses','cardigan-sweaters','stretch-jeans','plus-size-swimsuits'])
    #classifyText(['two-piece-swimsuits','sunglasses','cardigan-sweaters','stretch-jeans','plus-size-swimsuits','swimsuit-coverups','panties','distressed-jeans','camisole-tops','athletic-pants','brooches-and-pins','tunic-tops','scarves','teen-girls-intimates','gloves','coats','cropped-jeans','thongs','hats','sports-bras-and-underwear','cropped-pants','petite-jeans','blazers','halter-tops','diamond-necklaces','robes','shapewear','skinny-pants','flare-jeans'])
    #classifyText(['slippers','wedding-dresses','cashmere-sweaters','leather-jackets','plus-size-outerwear','chemises','plus-size-tops','tunic-tops','camisole-tops','diamond-bracelets','straight-leg-jeans','leggings','evening-dresses','flare-jeans','sunglasses','coats','socks','button-front-tops','stretch-jeans','shortsleeve-tops','bracelets','fur-and-shearling-coats','hats','teen-girls-jeans','one-piece-swimsuits','v-neck-sweaters','diamond-necklaces','athletic-jackets','gloves','blazers','bootcut-jeans','skinny-jeans','denim-jackets','maternity-intimates','sports-bras-and-underwear','wool-coats','womens-tech-accessories','brooches-and-pins','polo-tops','tees-and-tshirts','halter-tops','teen-girls-dresses','raincoats-and-trenchcoats','gowns','thongs','cropped-pants','pajamas','relaxed-jeans','leather-and-suede-coats','plus-size-swimsuits'])

if __name__ == "__main__":
    main()
