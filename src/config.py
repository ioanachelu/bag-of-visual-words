
codebook_path = '../models/codebook'

histogram_path = '../models/trainingdata'

histogram_test_path = '../models/testingdata'

img_train_path = '/home/ioana/crops'

model_file = '../models/trainingdata'
prediction_file = '../models/trainingdata'

image_label_dict = {
  "junk": 2,
  "signatures": 1,
}
ensembles = 34

EXTENSIONS = [".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]

K_THRESH = 1  # early stopping threshold for kmeans originally at 1e-5, increased for speedup

data_path = '../data/all/data'

train_path = '../data/train/train'
val_path = '../data/val/val'
val_random_path = '../data/val/val_random'
