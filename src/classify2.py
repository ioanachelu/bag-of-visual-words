import cPickle
import argparse
from sklearn import preprocessing
import sklearn.metrics as metrics
import config
import libsvm
import utils
from sklearn.externals import joblib
import math
import numpy as np

def parseArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument('--ensemble', '-e')
  args = parser.parse_args()

  return args.ensemble

def main():

  print "---------------------"
  print "## extract Sift features from ensemble"
  all_files_labels = {}
  all_features = {}
  test_images_paths = []
  ground_truth_labels = {}

  with open("%s.txt" % config.val_random_path, 'rt') as f:
    lines = f.readlines()

  for line in lines:
    path, label = line.split(" ")
    test_images_paths.append(path)
    all_files_labels[path] = int(label)
    ground_truth_labels[path] = int(label)

  all_features = utils.extractSift(test_images_paths)

  y_pred = {}
  for model_number in xrange(0, 1):
    # model_number = parseArgs()
    # model_number = int(model_number)

    for i in test_images_paths:
      all_files_labels[i] = 0  # label is unknown

    print "---------------------"
    print "## loading codebook  from ensemble %d from %s" % (model_number, config.codebook_path)
    with open("%s_%d.pkl" % (config.codebook_path, model_number), 'rb') as f:
      codebook = cPickle.load(f)

    print "---------------------"
    print "## computing visual word histograms from the codebook from ensemble %d" % model_number
    all_images_histgrams = {}
    for test_image_path in all_features:
      image_histgram = utils.computeHistograms(codebook, all_features[test_image_path])
      all_images_histgrams[test_image_path] = image_histgram

    print "---------------------"
    print "## load model for ensemble %d" % model_number
    y_true = []
    X = []
    clf = joblib.load('../models/model_%d.pkl' % model_number)
    for image_path, predicted_label in zip(test_images_paths, lines):
      all_images_histgrams[image_path] = [0 if math.isnan(x) else x for x in all_images_histgrams[image_path]]
      X.append(all_images_histgrams[image_path])
      y_true.append(ground_truth_labels[image_path])

    X = preprocessing.normalize(X, norm='l2')
    probabilities = clf.predict_proba(X)
    y_pred[model_number] = probabilities

  y_pred_sum = np.zeros_like(y_pred[0])
  for model_number, proba in y_pred.iteritems():
    y_pred_sum += proba

  y_pred_sum /= len(y_pred)
  y_pred_labels = [1 if p[0] > p[1] else 2 for p in y_pred_sum]

  accuracy = metrics.accuracy_score(y_true, y_pred_labels)
  precision = metrics.precision_score(y_true, y_pred_labels)
  recall = metrics.recall_score(y_true, y_pred_labels)
  f1 = 2 * (precision * recall) / (precision + recall)

  print "---------------------"
  print "## precision from ensemble %d is %f" % (model_number, precision)
  print "## recall from ensemble %d is %f" % (model_number, recall)
  print "## f1 score from ensemble %d is %f" % (model_number, f1)
  print "## accuracy from ensemble %d is %f" % (model_number, accuracy)

  for i in xrange(len(y_pred_labels)):
    if (y_pred_labels[i] != y_true[i]):
      print "WRONG: %s => %d(TRUE) %d(PRED)" % (test_images_paths[i], y_true[i], y_pred_labels[i])



if __name__ == '__main__':
  main()