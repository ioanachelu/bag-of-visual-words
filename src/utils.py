import numpy as np
import scipy.cluster.vq as vq
import os
import sift
import math

PRE_ALLOCATION_BUFFER = 1000  # for sift

def computeHistograms(codebook, descriptors):
  code, dist = vq.vq(descriptors, codebook)
  histogram_of_words, bin_edges = np.histogram(code,
                                            bins=range(codebook.shape[0] + 1),
                                            normed=True)
  return histogram_of_words


def writeHistogramsToFile(nclusters, labels, fnames, all_image_histgrams, features_fname, model_nr):
  data_rows = np.zeros(nclusters + 1)  # +1 for the category label
  for fname in fnames:
    histogram = all_image_histgrams[fname]
    if any([math.isnan(x) for x in histogram]):
      continue
    if (histogram.shape[0] != nclusters):  # scipy deletes empty clusters
      nclusters = histogram.shape[0]
      data_rows = np.zeros(nclusters + 1)
      print 'nclusters have been reduced to ' + str(nclusters)
    data_row = np.hstack((labels[fname], histogram))
    data_rows = np.vstack((data_rows, data_row))
  data_rows = data_rows[1:]
  fmt = '%i '
  for i in range(nclusters):
    fmt = fmt + str(i) + ':%f '
  np.savetxt("%s_%d.svm" % (features_fname, model_nr), data_rows, fmt)


def extractSift(input_files):
  print "extracting Sift features"
  all_features_dict = {}
  for i, fname in enumerate(input_files):
    rest_of_path = fname[:-(len(os.path.basename(fname)))]
    rest_of_path = os.path.join(rest_of_path, "sift")
    rest_of_path = os.path.join(rest_of_path, os.path.basename(fname))
    features_fname = rest_of_path + '.sift'
    if os.path.exists(features_fname) == False:
      # print "calculating sift features for", fname
      sift.process_image(fname, features_fname)
    # print "gathering sift features for", fname,
    locs, descriptors = sift.read_features_from_file(features_fname)
    # print descriptors.shape
    all_features_dict[fname] = descriptors
  return all_features_dict


def dict2numpy(dict):
  nkeys = len(dict)
  array = np.zeros((nkeys * PRE_ALLOCATION_BUFFER, 128))
  pivot = 0
  for key in dict.keys():
    value = dict[key]
    nelements = value.shape[0]
    while pivot + nelements > array.shape[0]:
      padding = np.zeros_like(array)
      array = np.vstack((array, padding))
    array[pivot:pivot + nelements] = value
    pivot += nelements
  array = np.resize(array, (pivot, 128))
  return array
