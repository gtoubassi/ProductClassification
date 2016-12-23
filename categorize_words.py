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
    
def trainText(categories):
    if len(categories) <= 10:    
        name = ",".join(categories)
    else:
        name = ",".join(categories[0:5]) + ("... (%d more)" % + len(categories))
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

        for _ in range(10):
            random.shuffle(train)

            print("eval (N=%d shape=%s)" % (len(test_x), test_x[0].shape))
  
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print(sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))

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
    
    #trainText([])
    #trainText(['skinny-jeans', 'bootcut-jeans'])
    #trainText(['two-piece-swimsuits','sunglasses','cardigan-sweaters','stretch-jeans','plus-size-swimsuits','swimsuit-coverups','panties','distressed-jeans','camisole-tops','athletic-pants','brooches-and-pins','tunic-tops','scarves','teen-girls-intimates','gloves','coats','cropped-jeans','thongs','hats','sports-bras-and-underwear','cropped-pants','petite-jeans','blazers','halter-tops','diamond-necklaces','robes','shapewear','skinny-pants','flare-jeans'])
    #trainText(['slippers','wedding-dresses','cashmere-sweaters','leather-jackets','plus-size-outerwear','chemises','plus-size-tops','tunic-tops','camisole-tops','diamond-bracelets','straight-leg-jeans','leggings','evening-dresses','flare-jeans','sunglasses','coats','socks','button-front-tops','stretch-jeans','shortsleeve-tops','bracelets','fur-and-shearling-coats','hats','teen-girls-jeans','one-piece-swimsuits','v-neck-sweaters','diamond-necklaces','athletic-jackets','gloves','blazers','bootcut-jeans','skinny-jeans','denim-jackets','maternity-intimates','sports-bras-and-underwear','wool-coats','womens-tech-accessories','brooches-and-pins','polo-tops','tees-and-tshirts','halter-tops','teen-girls-dresses','raincoats-and-trenchcoats','gowns','thongs','cropped-pants','pajamas','relaxed-jeans','leather-and-suede-coats','plus-size-swimsuits'])
    #trainText(['teen-girls-tops','plus-size-skirts','plus-size-jeans','flare-jeans','cropped-pants','sleeveless-tops','plus-size-pants','wedding-dresses','gloves','tank-tops','athletic-jackets','womens-tech-accessories','camisole-tops','brooches-and-pins','casual-jackets','coats','shorts','pajamas','wool-coats','diamond-necklaces','one-piece-swimsuits','rings','blazers','leather-and-suede-coats','day-dresses','socks','crewneck-sweaters','plus-size-jackets','petite-dresses','petite-jeans','athletic-pants','long-skirts','hats','petite-skirts','bracelets','chemises','slippers','gowns','necklaces','shortsleeve-tops','cashmere-sweaters','athletic-tops','casual-pants','hosiery','turleneck-sweaters','cropped-jeans','fur-and-shearling-coats','shapewear','relaxed-jeans','earrings','mini-skirts','belts','maternity-dresses','petite-jackets','cardigan-sweaters','distressed-jeans','stretch-jeans','scarves','sunglasses','polo-tops','robes','plus-size-sweaters','two-piece-swimsuits','raincoats-and-trenchcoats','sweatshirts','petite-tops','teen-girls-jeans','sports-bras-and-underwear','v-neck-sweaters','cashmere-tops','bootcut-jeans','teen-girls-dresses','longsleeve-tops','straight-leg-jeans','maternity-jeans'])
    trainText(['petite-jeans','tunic-tops','maternity-tops','gloves','bras','cropped-jeans','tees-and-tshirts','camisole-tops','plus-size-skirts','teen-girls-jeans','relaxed-jeans','shortsleeve-tops','plus-size-pants','sunglasses','turleneck-sweaters','diamond-rings','tank-tops','mid-length-skirts','pajamas','casual-jackets','bracelets','panties','leather-and-suede-coats','watches','vests','leather-jackets','cardigan-sweaters','maternity-jeans','wedding-dresses','camisoles','mini-skirts','raincoats-and-trenchcoats','cropped-pants','hosiery','gowns','diamond-bracelets','cashmere-sweaters','plus-size-outerwear','stretch-jeans','belts','athletic-jackets','v-neck-sweaters','leggings','distressed-jeans','charms','plus-size-intimates','petite-skirts','skinny-pants','polo-tops','maternity-intimates','petite-jackets','petite-dresses','brooches-and-pins','fur-and-shearling-coats','womens-suits','scarves','plus-size-jackets','long-skirts','casual-pants','diamond-necklaces','flare-jeans','maternity-dresses','halter-tops','womens-tech-accessories','teen-girls-tops','wide-leg-pants','wool-coats','petite-pants','coats','plus-size-sweaters','two-piece-swimsuits','button-front-tops','blazers','robes','straight-leg-jeans','thongs','bootcut-jeans','plus-size-dresses','teen-girls-dresses','shapewear','earrings','athletic-pants','petite-tops','one-piece-swimsuits'     ,'crewneck-sweaters','plus-size-jeans','necklaces','puffer-coats','chemises','diamond-earrings'     ,'sweatshirts','denim-jackets','sleeveless-tops'])#   ,'skinny-jeans','athletic-tops','swimsuit-coverups'     ,'shorts','bridal-gowns','day-dresses','athletic-shorts','teen-girls-intimates','cashmere-tops','sports-bras-and-underwear','key-chains','plus-size-swimsuits','longsleeve-tops','hats','evening-dresses','cocktail-dresses','slippers','petite-sweaters','plus-size-tops','socks','rings'])

if __name__ == "__main__":
    main()
