import os

from dataset import DataLoaderTrain, DataLoaderVal, DataLoaderTest
def get_training_data(rgb_dir, img_options, debug):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options, None, debug)

def get_validation_data(rgb_dir,  debug=False):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, None, debug)

def get_test_data(rgb_dir,  debug=False):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, None, debug)



