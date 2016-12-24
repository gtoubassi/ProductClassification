#!/usr/bin/env python
#
import argparse
import database
import random
import retrain
import hashlib
import os
from tensorflow.python.util import compat
import numpy as np
import re

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
    
    test_filenames, test_results = retrain.retrain(image_lists)
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
    args = parser.parse_args()
    retrain.setargs(args)

    global db
    db = database.Database(args.db_path)
    
    classifyImages(args.images_path, args.how_many_training_steps, ['skinny-jeans', 'bootcut-jeans'])
    #classifyImages(args.images_path, args.how_many_training_steps, ['clutches', 'bootcut-jeans'])

    #classifyImages(args.images_path, args.how_many_training_steps, [])

if __name__ == "__main__":
    main()
