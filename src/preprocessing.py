import cv2

import os
from src import config
import numpy as np
import random


def preprocessing():
  all_files_labels = {}
  all_files_labels_inv = {}
  all_files_labels_inv[1] = []
  all_files_labels_inv[2] = []
  labels_folders = [f for f in os.listdir(config.img_train_path)]
  for label_folder in labels_folders:
    if label_folder not in config.image_label_dict:
      continue
    label_path = os.path.join(config.img_train_path, label_folder)
    label_sift_path = os.path.join(label_path, "sift")
    if not os.path.exists(label_sift_path):
      os.makedirs(label_sift_path)

    images_paths = [os.path.join(label_path, f)
                    for f in os.listdir(label_path)
                    if os.path.isfile(os.path.join(label_path, f))
                    and os.path.splitext(f)[-1].lower() in config.EXTENSIONS]
    for image_path in images_paths:
      # img = cv2.imread(image_path)
      # os.remove(image_path)

      filename, file_extension = os.path.splitext(image_path)
      filename = "_".join(filename.split(" "))
      filename += ".png"

      os.rename(image_path, filename)

      # cv2.imwrite(filename, img)
      all_files_labels[filename] = config.image_label_dict[label_folder]
      all_files_labels_inv[config.image_label_dict[label_folder]].append(filename)

  nr_positive_examples = len(all_files_labels_inv[1])
  nr_negative_examples = len(all_files_labels_inv[2])

  shuffle_indices = np.random.permutation(np.arange(nr_positive_examples))
  shuffled_positive_samples = np.asarray(all_files_labels_inv[1])[shuffle_indices].tolist()

  shuffle_indices = np.random.permutation(np.arange(nr_negative_examples))
  shuffled_negative_samples = np.asarray(all_files_labels_inv[2])[shuffle_indices].tolist()


  val_split_pos = int(len(shuffled_positive_samples) * 0.2)
  val_split_neg = int(len(shuffled_negative_samples) * 0.2)

  pos_train, pos_val = shuffled_positive_samples[:-val_split_pos], shuffled_positive_samples[-val_split_pos:]
  neg_train, neg_val = shuffled_negative_samples[:-val_split_neg], shuffled_negative_samples[-val_split_neg:]

  with open("%s.txt" % config.val_path, 'wt') as f:
    for path in pos_val:
      f.write("%s %d\n" % (path, 1))
    for path in neg_val:
      f.write("%s %d\n" % (path, 2))

  for i in xrange(100):
    indices = random.sample(range(0, len(neg_train)), len(pos_train))
    # negative_examples_ensemble = all_files_labels_inv[2][i*nr_positive_examples:(i+1)*nr_positive_examples]
    negative_examples_ensemble = np.asarray(neg_train)[indices].tolist()
    with open("%s_%d.txt" % (config.data_path, i), 'wt') as f:
      for path in pos_train:
        f.write("%s %d\n" % (path, 1))
      for path in negative_examples_ensemble:
        f.write("%s %d\n" % (path, 2))


def main():
  preprocessing()

if __name__ == '__main__':
  main()