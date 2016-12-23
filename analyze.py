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

def analyzeCoverageForDesiredAccuracy(predictions, desired_accuracy):
    coverage = 0.0
    for i in xrange(0, 100):
        threshold = i/100.0
        coverage_count = 0
        num_correct = 0
        for p in predictions:
            if p['score'] > threshold:
                num_correct += 1 if p['category'] == p['prediction'] else 0
                coverage_count += 1
        accuracy = num_correct / float(coverage_count)
        if accuracy >= desired_accuracy:
            coverage = float(coverage_count) / len(predictions)
            break
    return accuracy, coverage

def analyze(predictions):
    accuracy, coverage = analyzeCoverageForDesiredAccuracy(predictions, 0)
    print("  Accuracy: %.2f%% (N=%d)" % (100*accuracy, len(predictions)))
    accuracy, coverage = analyzeCoverageForDesiredAccuracy(predictions, .95)
    print("  95%% Accuracy Coverage: %d%%" % (coverage*100))
    accuracy, coverage = analyzeCoverageForDesiredAccuracy(predictions, .98)
    print("  98%% Accuracy Coverage: %d%%" % (coverage*100))
    
    parentMap = db.getParentCategoryMap()
    parentPredictions = []
    for p in predictions:
        other = {}
        other['category'] = parentMap[p['category']]
        other['prediction'] = parentMap[p['prediction']]
        other['score'] = p['score']
        parentPredictions.append(other)
    
    accuracy, coverage = analyzeCoverageForDesiredAccuracy(parentPredictions, 0)
    print("  Parent Accuracy: %.2f%%" % (100*accuracy))
    accuracy, coverage = analyzeCoverageForDesiredAccuracy(parentPredictions, .95)
    print("  Parent 95%% Accuracy Coverage: %d%%" % (coverage*100))
    accuracy, coverage = analyzeCoverageForDesiredAccuracy(parentPredictions, .98)
    print("  Parent 98%% Accuracy Coverage: %d%%" % (coverage*100))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default='crawl.db', help="Path to sqlite db file")
    args = parser.parse_args()

    global db
    db = database.Database(args.db_path)
    
    experiments = db.getExperiments()
    for e in experiments:
        predictions = db.getPredictedCategories(e['id'])
        if len(predictions) > 0:
            print(e['name'])
            analyze(predictions)

if __name__ == "__main__":
    main()
