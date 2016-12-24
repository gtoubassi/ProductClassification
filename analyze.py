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
from itertools import chain

def analyzeCoverageForDesiredAccuracy(predictions, desired_accuracy):
    coverage = 0.0

    # Test thresholds in increments of .01, except for .99-1.00 where we test at .001.
    # This was found to be necessary when trying to achieve high coverages
    # (e.g. a score of .999 might be needed to get a 98% coverage so without this
    # extra precision we'd erroneously report 0)
    for i in chain(xrange(0, 900,10), xrange(900,1000)):
        threshold = i/1000.0
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

def dumpErrors(experimentId):
    experiment = db.getExperiment(experimentId)
    errors = db.getMisclassificationsForExperiment(experimentId, 500)
    print("<html><body>")
    print("<b>Experiment %d on %s: %s</b><br/><br/>" % (experiment['id'], experiment['date'], experiment['name']))
    print("<table border='1'>")
    
    for e in errors:
        print("<tr>")

        print('<td><a href="https://www.shopstyle.com/action/loadRetailerProductPage?id=%d"><img src="%s" xheight="200"></a></td>' % (e['id'], e['image']))
        
        print("<td><table>")
        print('<tr><td align=right>Name:</td><td>%s</td></tr>' % e['name'].encode('utf-8'))
        print('<tr><td align=right>Description:</td><td>%s</td></tr>' % e['description'].encode('utf-8'))
        print('<tr><td align=right>Category:</td><td>%s</td></tr>' % e['category_id'])
        print('<tr><td align=right>Predicted:</td><td>%s (%f)</td></tr>' % (e['predicted_category'], e['score']))
        print("</table></td>")

        print("</tr>")
    
    print("</table>")
    print("</body></html>")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default='crawl.db', help="Path to sqlite db file")
    parser.add_argument("--dump-errors-for-experiment", type=int, help="Dump html report of the 100 most egregious (highest scoring) errors")

    args = parser.parse_args()

    global db
    db = database.Database(args.db_path)

    if args.dump_errors_for_experiment:
        dumpErrors(args.dump_errors_for_experiment)
    else:
        experiments = db.getExperiments()
        for e in experiments:
            predictions = db.getPredictedCategories(e['id'])
            if len(predictions) > 0:
                print("#%d: %s" % (e['id'], e['name']))
                analyze(predictions)

if __name__ == "__main__":
    main()
