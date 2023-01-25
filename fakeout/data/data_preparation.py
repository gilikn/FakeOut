# create tmp/downloads/extracted/ZIP.dataset_name_train.zip directory and copy videos to there
# create train and test txt files of all videos
# create wav file of every video
import logging
import os
import subprocess
from os import path

from absl import app
from absl import flags
from data_config import DATASET_CONFIG, RELATIVE_PATH, DIRECTORY_NAME, FILTERED_TEST_FILE
from tqdm import tqdm

flags.DEFINE_string('dataset_name', None, 'Dataset name.')
flags.DEFINE_string('split', None, 'Split, should be train or test.')
flags.DEFINE_string('videos_path', None, 'Path to directory in which all videos of the split are held.')
FLAGS = flags.FLAGS


def generate_directories(dataset_directory_name):
    subprocess.call(
        f'mkdir -p tmp/downloads/extracted/ZIP.train_test_split.zip',
        cwd=dataset_directory_name, shell=True)
    return


def create_soft_link_directory(videos_path, dataset_directory_path, dataset_directory_name, split):
    subprocess.call(
        f"ln -s {videos_path} ZIP.{dataset_directory_name}_{split}.zip", shell=True,
        cwd=os.path.join(dataset_directory_path, 'tmp/downloads/extracted'))


def generate_train_test_list_from_extracted(dataset_directory_path, dataset_directory_name, split):
    """
    create train, test txt files for training and inference.
    :return: None
    """
    files = []
    should_filter = False
    if path.exists(os.path.join(dataset_directory_path, FILTERED_TEST_FILE)):
        with open(os.path.join(dataset_directory_path, FILTERED_TEST_FILE), 'r') as f:
            files = [line.rstrip('\n') for line in f]
            should_filter = True

    # generate TRAIN list
    with open(f"{dataset_directory_path}/tmp/downloads/extracted/ZIP.train_test_split.zip/trainlist.txt", 'w') as f:
        for x in os.listdir(
                f"{dataset_directory_path}/tmp/downloads/extracted/ZIP.{dataset_directory_name}_{split}.zip"):
            if '.mp4' in x:
                if should_filter:
                    if x.split('.')[0] in files:
                        f.write("{}\n".format(x))
                else:
                    f.write("{}\n".format(x))

    # generate TEST list
    with open(
            f"{dataset_directory_path}/tmp/downloads/extracted/ZIP.train_test_split.zip/testlist.txt", 'w') as f:
        for x in os.listdir(
                f"{dataset_directory_path}/tmp/downloads/extracted/ZIP.{dataset_directory_name}_{split}.zip"):
            if '.mp4' in x:
                if should_filter:
                    if x.split('.')[0] in files:
                        f.write("{}\n".format(x))
                else:
                    f.write("{}\n".format(x))
    return


def create_wav_files(directory_path):
    files = [x for x in os.listdir(directory_path) if '.mp4' in x]
    for file_name in tqdm(files):
        subprocess.call(f"ffmpeg -y -i {file_name} -ac 2 -f wav {file_name.split('.')[0]}.wav", shell=True,
                        cwd=directory_path)


def main(argv):
    del argv
    logging.basicConfig()
    logging.root.setLevel(logging.NOTSET)
    if FLAGS.dataset_name not in DATASET_CONFIG.keys():
        raise Exception(f"'dataset_name' parameter should be set to one of {DATASET_CONFIG.keys()}.")
    if FLAGS.split not in ['train', 'test']:
        raise Exception("'split' parameter should be set to either 'train' or 'test'.")
    assert FLAGS.videos_path is not None, "Please enter the path to the videos!"

    dataset_directory_path = DATASET_CONFIG[FLAGS.dataset_name][RELATIVE_PATH]
    dataset_directory_name = DATASET_CONFIG[FLAGS.dataset_name][DIRECTORY_NAME]
    split = FLAGS.split
    videos_path = FLAGS.videos_path
    generate_directories(dataset_directory_path)
    create_soft_link_directory(videos_path, dataset_directory_path, dataset_directory_name, split)
    generate_train_test_list_from_extracted(dataset_directory_path, dataset_directory_name, split)
    create_wav_files(
        os.path.join(f'{dataset_directory_path}/tmp/downloads/extracted/', f'ZIP.{dataset_directory_name}_{split}.zip'))


if __name__ == '__main__':
    app.run(main)
