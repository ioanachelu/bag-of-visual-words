import cPickle

import libsvm
import numpy as np
import os
# import scipy.cluster.vq as vq
from scipy.cluster.vq import *
import utils
from src import config
import argparse
from sklearn import svm, preprocessing, cluster
from sklearn.externals import joblib
import math



def extract_features(i):
  # labels_folders = [f for f in os.listdir(config.img_train_path)]

  all_features = {}
  all_files = []
  all_files_labels = {}

  with open("%s_%d.txt" % (config.train_path, i), 'rt') as f:
    lines = f.readlines()

  for line in lines:
    path, label = line.split(" ")
    all_files.append(path)
    all_files_labels[path] = int(label)

  image_features = utils.extractSift(all_files)
  all_features.update(image_features)

  return all_files, all_features, config.image_label_dict, all_files_labels


def compute_codebook(all_features, i):
  if os.path.exists("%s_%d.pkl" % (config.codebook_path, i)):
    with open("%s_%d.pkl" % (config.codebook_path, i), 'rb') as f:
      codebook = cPickle.load(f)
  else:
    all_features_array = utils.dict2numpy(all_features)
    nfeatures = all_features_array.shape[0]
    nclusters = 2000#int(np.sqrt(nfeatures)) #2000
    # kmeans_estimator = cluster.KMeans(n_clusters=nclusters, init='k-means++', tol=config.K_THRESH, n_jobs=-1)
    # kmeans_estimator.fit(all_features_array)
    # codebook = kmeans_estimator.cluster_centers_
    print "Clustering %d features in %d clusters" % (len(all_features_array), nclusters)
    # all_features_array_whiten = whiten(all_features_array)
    all_features_array_whiten = all_features_array.astype(np.float32)
    # retVal, bestLabel, codebook = cv2.kmeans(all_features_array_whiten, nclusters, criteria, attempts, flags)
    codebook, distortion = kmeans(all_features_array_whiten,
                                     nclusters,
                                     thresh=config.K_THRESH, check_finite=True)


    with open("%s_%d.pkl" % (config.codebook_path, i), 'wb') as f:
      cPickle.dump(codebook, f, protocol=cPickle.HIGHEST_PROTOCOL)

  return codebook


def parseArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument('--ensemble', '-e')
  args = parser.parse_args()

  return args.ensemble

def main():
  # i = parseArgs()
  # i = int(i)
  for i in xrange(1, 100):
    print "---------------------"
    print "## loading the images and extracting the sift features for ensemble %d" % i
    all_files, all_features, image_label_dict, all_files_labels = extract_features(i)

    print "---------------------"
    print "## computing the visual words via k-means for ensemble %d" % i
    codebook = compute_codebook(all_features, i)

    print "---------------------"
    print "## compute the visual words histograms for each image for ensemble %d" % i
    all_images_histgrams = {}
    for image_path in all_features:
      image_histgram = utils.computeHistograms(codebook, all_features[image_path])
      all_images_histgrams[image_path] = image_histgram

    print "---------------------"
    print "## write the histograms to file to pass it to the svm for ensemble %d" % i

    X = []
    Y = []
    for path in all_files:
      all_images_histgrams[path] = [0 if math.isnan(x) else x for x in all_images_histgrams[path]]
      X.append(all_images_histgrams[path])
      Y.append(all_files_labels[path])

    X = preprocessing.normalize(X, norm='l2')

    clf = svm.SVC(C=1000, kernel='linear', probability=True)
    clf.fit(X, Y)
    joblib.dump(clf, '../models/model_%d.pkl' % i, protocol=cPickle.HIGHEST_PROTOCOL, compress=3)

if __name__ == '__main__':
  main()