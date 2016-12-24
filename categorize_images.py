#!/usr/bin/env python
#
import argparse
import database
import random
import retrain
import hashlib
import os
import tensorflow as tf
from tensorflow.python.util import compat
import numpy as np
import re
import math

def classifyImages(image_dir, num_steps, categories):
    if len(categories) <= 10:    
        name = ",".join(categories)
    else:
        name = ",".join(categories[0:5]) + ("... (%d total)" % + len(categories))
    experimentId = db.addExperiment("Image classification of: %s" % name)

    products = db.getProducts(categories)

    image_lists = {}
    files_to_categories = {}
    files_to_productId = {}

    for p in products:
        file = p['image'].split('/')[-1]
        file = file[0:2] + '/' + file
        files_to_productId[file] = p['id']
        cat = p['category_id']
        files_to_categories[file] = cat
        
        if image_lists.get(cat) is None:
            image_lists[cat] = {
                'dir': image_dir,
                'training': [],
                'testing': [],
                'validation': []
            }
        hash_str = hashlib.sha1(compat.as_bytes(file)).hexdigest()
        percentage_hash = (int(hash_str, 16) % 10000) * 100.0 / 10000
        if percentage_hash < 10:
            image_lists[cat]['validation'].append(file)
        elif percentage_hash < 20:
            image_lists[cat]['testing'].append(file)
        else:
            image_lists[cat]['training'].append(file)
    
    x = tf.placeholder(tf.float32, shape=[None, 2048], name='bottleneck_input')
    y_target = tf.placeholder(tf.float32, shape=[None, len(image_lists)], name='groundtruth_input')

    stdv1 = 1.0 / math.sqrt(2048)
    w1 = tf.Variable(tf.random_uniform([2048, len(image_lists)], minval=-stdv1, maxval=stdv1))
    b1  = tf.Variable(tf.random_uniform([len(image_lists)], minval=-stdv1, maxval=stdv1))

    logits = tf.matmul(x, w1) + b1

    y = tf.nn.softmax(logits)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_target)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    prediction = tf.argmax(y, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(y_target, 1))
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    train_step = tf.train.AdamOptimizer().minimize(cross_entropy_mean)
    
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train
    for step in xrange(args.how_many_training_steps):

        batch_xs, batch_ys, filenames = retrain.get_random_cached_bottlenecks(None, image_lists, args.train_batch_size, 'training', args.bottleneck_dir, '', None, None)
        sess.run(train_step, feed_dict={x: batch_xs, y_target: batch_ys})

        validation_xs, validation_ys, validation_filenames = retrain.get_random_cached_bottlenecks(None, image_lists, args.validation_batch_size, 'validation', args.bottleneck_dir, '', None, None)
        accuracy = sess.run(evaluation_step, feed_dict={x: batch_xs, y_target: batch_ys})

        print("step %d of %d validation accuracy=%f" % (step, args.how_many_training_steps, accuracy))

    test_xs, test_ys, test_filenames = retrain.get_random_cached_bottlenecks(None, image_lists, -1, 'testing', args.bottleneck_dir, '', None, None)
    accuracy, test_results = sess.run([evaluation_step, y], feed_dict={x: test_xs, y_target: test_ys})
    print("Final testing accuracy %f" % accuracy)
    
    total_correct = 0
    for i, f in enumerate(test_filenames):
        file = re.sub('^' + image_dir + '/', '', f)
        prediction_index = np.argmax(test_results[i])
        prediction_score = np.max(test_results[i])
        predicted_cat = list(image_lists.keys())[prediction_index]
        productId = files_to_productId[file]
        db.addPredictedCategory(experimentId, productId, predicted_cat, prediction_score)
        correct_cat = files_to_categories[file]
        correct_index = list(image_lists.keys()).index(correct_cat)
        total_correct += 1 if prediction_index == correct_index else 0
    print("%d of %d (%f)" % (total_correct, len(test_filenames), float(total_correct)/len(test_filenames)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default='crawl.db', help="Path to sqlite db file")
    parser.add_argument("--images-path", default='images', help="Path to directory in which images should be saved")
    retrain.addargs(parser)
    global args
    args = parser.parse_args()
    retrain.setargs(args)

    global db
    db = database.Database(args.db_path)
    
    classifyImages(args.images_path, args.how_many_training_steps, ['skinny-jeans', 'bootcut-jeans'])
    #classifyImages(args.images_path, args.how_many_training_steps, ['clutches', 'bootcut-jeans'])

    #classifyImages(args.images_path, args.how_many_training_steps, [])

if __name__ == "__main__":
    main()
