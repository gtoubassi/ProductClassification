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
import operator
import math
import categorize_words
import categorize_images

def classifyImagesAndText(image_dir, num_steps, categories):
    if len(categories) <= 10:    
        name = ",".join(categories)
    else:
        name = ",".join(categories[0:5]) + ("... (%d total)" % + len(categories))
    experimentId = db.addExperiment("Image+Text classification of: %s (%d steps)" % (name, args.how_many_training_steps))

    products = db.getProducts(categories)
    random.shuffle(products)

    words_x, words_y, words_h1, words_y_logits, words_y_, words_train_step, words_prediction, vocab_indices, category_indices, seenCategories = categorize_words.prepWordTraining(products)
    
    images_x, images_y, images_y_logits, images_y_target, images_train_step, images_evaluation_step, image_lists, files_to_categories, files_to_productId = categorize_images.prepImageTraining(image_dir, products)

    productId_to_product = {}
    for p in products:
        productId_to_product[p['id']] = p

    # build the combined network

    merged_input = tf.concat(1, [words_h1, images_x])

    y_target = tf.placeholder(tf.float32, shape=[None, len(seenCategories)], name='combined_target')

    stdv1 = 1.0 / math.sqrt(int(merged_input.get_shape()[-1]))
    w1 = tf.Variable(tf.random_uniform([int(merged_input.get_shape()[-1]), len(seenCategories)], minval=-stdv1, maxval=stdv1))
    b1  = tf.Variable(tf.random_uniform([len(seenCategories)], minval=-stdv1, maxval=stdv1))

    y_logits = tf.matmul(merged_input, w1) + b1

    y = tf.nn.softmax(y_logits)

    prediction = tf.argmax(y, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(y_target, 1))
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y_logits, y_target)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy_mean)
        
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train
    for step in xrange(args.how_many_training_steps):

        # Train
        images_batch_xs, images_batch_ys, filenames = retrain.get_random_cached_bottlenecks(None, image_lists, args.train_batch_size, 'training', args.bottleneck_dir, '', None, None)        
        batch_products = []
        for file in filenames:
            f = re.sub('^' + image_dir + '/', '', file)
            batch_products.append(productId_to_product[files_to_productId[f]])
        
        words_batch_xs, words_batch_ys = categorize_words.computeTFDataForProducts(batch_products, vocab_indices, category_indices, seenCategories)
        sess.run(train_step, feed_dict={images_x: images_batch_xs, words_x: words_batch_xs, y_target: images_batch_ys})

        # Validation
        images_batch_xs, images_batch_ys, filenames = retrain.get_random_cached_bottlenecks(None, image_lists, args.validation_batch_size, 'validation', args.bottleneck_dir, '', None, None)        
        batch_products = []
        for file in filenames:
            f = re.sub('^' + image_dir + '/', '', file)
            batch_products.append(productId_to_product[files_to_productId[f]])
        
        words_batch_xs, words_batch_ys = categorize_words.computeTFDataForProducts(batch_products, vocab_indices, category_indices, seenCategories)
        accuracy = sess.run(evaluation_step, feed_dict={images_x: images_batch_xs, words_x: words_batch_xs, y_target: images_batch_ys})
        print("step %d of %d validation accuracy=%f" % (step, args.how_many_training_steps, accuracy))

    # Test
    images_batch_xs, images_batch_ys, test_filenames = retrain.get_random_cached_bottlenecks(None, image_lists, args.validation_batch_size, 'validation', args.bottleneck_dir, '', None, None)        
    batch_products = []
    for file in test_filenames:
        f = re.sub('^' + image_dir + '/', '', file)
        batch_products.append(productId_to_product[files_to_productId[f]])
    
    words_batch_xs, words_batch_ys = categorize_words.computeTFDataForProducts(batch_products, vocab_indices, category_indices, seenCategories)
    accuracy, test_results = sess.run([evaluation_step, y], feed_dict={images_x: images_batch_xs, words_x: words_batch_xs, y_target: images_batch_ys})
    print("step %d of %d test accuracy=%f" % (step, args.how_many_training_steps, accuracy))

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
    
    #classifyImagesAndText(args.images_path, args.how_many_training_steps, ['skinny-jeans', 'bootcut-jeans'])
    classifyImagesAndText(args.images_path, args.how_many_training_steps, [])

if __name__ == "__main__":
    main()
