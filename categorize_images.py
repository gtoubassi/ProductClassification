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

def categorize(image_dir, num_steps, categories):
    experimentId = db.addExperiment("Image categorization of: %s (steps: %d)" % (",".join(categories), num_steps))
    image_lists = {}
    files_to_categories = {}
    files_to_productId = {}
    for cat in categories:
        products = db.getProductsForCategory(cat)
        files = [p['image'].split('/')[-1] for p in products]
        files = [f[0:2]+'/'+f for f in files]
        
        for p in products:
            files_to_productId[os.path.basename(p['image'])] = p['id']

        train = []
        validation = []
        test = []
    
        for f in files:
            files_to_categories[os.path.basename(f)] = cat
            hash_str = hashlib.sha1(compat.as_bytes(f)).hexdigest()
            percentage_hash = (int(hash_str, 16) % 10000) * 100.0 / 10000
            if percentage_hash < 10:
                validation.append(f)
            elif percentage_hash < 20:
                test.append(f)
            else:
                train.append(f)

        image_lists[cat] = {
            'dir': image_dir,
            'training': train,
            'testing': test,
            'validation': validation,
        }
    test_filenames, test_results = retrain.retrain(image_lists)
    total_correct = 0
    for i, f in enumerate(test_filenames):
        prediction_index = np.argmax(test_results[i])
        prediction_score = np.max(test_results[i])
        predicted_cat = list(image_lists.keys())[prediction_index]
        productId = files_to_productId[os.path.basename(f)]
        db.addPredictedCategory(experimentId, productId, predicted_cat, prediction_score)
        correct_cat = files_to_categories[os.path.basename(f)]
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
    
    #categorize(args.images_path, args.how_many_training_steps, ['skinny-jeans', 'bootcut-jeans'])
    #categorize(args.images_path, args.how_many_training_steps, ['clutches', 'bootcut-jeans'])
    categorize(args.images_path, args.how_many_training_steps,['clutches', 'bootcut-jeans','shortsleeve-tops','distressed-jeans','womens-tech-accessories'])
    #categorize(args.images_path, args.how_many_training_steps,['shortsleeve-tops','distressed-jeans','womens-tech-accessories','maternity-pants','skinny-jeans','petite-sweatshirts','bras','teen-girls-shorts','long-skirts','swimsuit-coverups'])

if __name__ == "__main__":
    main()
