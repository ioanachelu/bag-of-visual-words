import numpy as np
from src import config

def randomize_val():
  with open("%s.txt" % config.val_path, 'rt') as f:
    lines = f.readlines()
  paths = []
  labels = []
  for line in lines:
    path, label = line.split(" ")
    label = int(label)
    paths.append(path)
    labels.append(label)

  shuffle_indices = np.random.permutation(np.arange(len(labels)))
  x_shuffled = np.asarray(paths)[shuffle_indices].tolist()
  y_shuffled = np.asarray(labels)[shuffle_indices].tolist()

  with open("%s.txt" % config.val_random_path, 'wt') as f:
    for x, y in zip(x_shuffled, y_shuffled):
      f.write("%s %d\n" % (x, y))

def randomize_train():

  for i in xrange(100):
    with open("%s_%d.txt" % (config.data_path, i), 'rt') as f:
        lines = f.readlines()

    paths = []
    labels = []
    for line in lines:
      path, label = line.split(" ")
      label = int(label)
      paths.append(path)
      labels.append(label)

    shuffle_indices = np.random.permutation(np.arange(len(labels)))
    x_shuffled = np.asarray(paths)[shuffle_indices].tolist()
    y_shuffled = np.asarray(labels)[shuffle_indices].tolist()
    # val_split = int(len(x_shuffled) * 0.2)
    # x_train, x_val = x_shuffled[:-val_split], x_shuffled[-val_split:]
    # y_train, y_val = y_shuffled[:-val_split], y_shuffled[-val_split:]

    with open("%s_%d.txt" % (config.train_path, i), 'wt') as f:
      for x, y in zip(x_shuffled, y_shuffled):
        f.write("%s %d\n" % (x, y))


def main():
  randomize_train()
  randomize_val()

if __name__ == '__main__':
  main()