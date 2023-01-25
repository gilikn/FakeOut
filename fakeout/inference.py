# TODO - insert Apache license of MMV and ours
import json
import os
import pickle
import sys
from os import path
from typing import Any, Dict

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
from absl import app
from absl import flags

from config import MLP_LAYERS_NAMES
from data.data_config import DATASET_FOLDER, DATASET_CONFIG, DATASET_MIDFIX, AUDIO_FILES_JSON, \
    RELATIVE_PATH, DIRECTORY_NAME
# todo - see if there is a way to overcome this
from data.data_utils import define_usable_audio, get_dummy_data, generate_dataset, union_part_videos, get_auc_score

# todo - remove
sys.path.append('/tmp/pycharm_project_793')

# from fakeout import config
import config
from models import mm_embeddings
from utils.deepfake_dataset import DeepfakeConfig

# todo - add content to "fakeout/data/DFDC/filtered_test.txt" file

# TODO - choose relevant resnet path (generic)
flags.DEFINE_string('checkpoint_path', '/mnt/raid1/home/gili_knn/checkpoint_fakeout_video_audio_tsm_resnet_x2.pkl',
                    'The path to load the pre-trained weights from.')
flags.DEFINE_string('dataset_name', 'dfdc',
                    'The directory with the challenge dataset.')
flags.DEFINE_integer('eval_batch_size', 1,
                     'The batch size for evaluation.')
# TODO - put it in finetune.py only
# flags.DEFINE_integer('train_batch_size', 2,
#                      'The batch size for training.')
# TODO - put it in finetune.py only
# flags.DEFINE_integer('num_train_epochs', 6,
#                      'How many epochs to collect features during training.')
flags.DEFINE_integer('num_test_windows', 9,
                     'How many windows to average on during test.')
flags.DEFINE_integer('num_frames', 32,
                     'Number of video frames.')
flags.DEFINE_bool('use_audio', True, 'Whether to use audio in current run or not')
flags.DEFINE_integer('mlp_first_layer_size', 6144,
                     'First layer size of MLP classifier.')
flags.DEFINE_integer('mel_filters', 80,
                     'Number of mel filters for audio modality.')

FLAGS = flags.FLAGS


def mlp_builder(input_size, input=None):
    if input is None:
        input = jnp.zeros(input_size)

    first_layer = hk.Linear(
        name=MLP_LAYERS_NAMES[0],
        output_size=512,
        w_init=hk.initializers.TruncatedNormal(0.02)
    )(input)

    second_layer = hk.Linear(
        name=MLP_LAYERS_NAMES[1],
        output_size=128,
        w_init=hk.initializers.TruncatedNormal(0.02)
    )(first_layer)

    clf_logits = hk.Linear(
        name=MLP_LAYERS_NAMES[2],
        output_size=2,
        w_init=hk.initializers.TruncatedNormal(0.02)
    )(second_layer)

    return clf_logits


def helper_fn():
    clf_logits = mlp_builder(input_size=FLAGS.mlp_first_layer_size, input=None)
    return clf_logits


def forward_fn(images: jnp.ndarray,
               audio_spectrogram: jnp.ndarray,
               word_ids: jnp.ndarray,
               is_training: bool,
               model_config: Dict[str, Any]):
    """Forward pass of the model."""

    # This should contain the pre-trained weights. We set it to zero because it
    # will be loaded from the checkpoint.
    language_model_vocab_size = 65536
    word_embedding_dim = 300
    dummy_embedding_matrix = jnp.zeros(shape=(language_model_vocab_size,
                                              word_embedding_dim))

    module = mm_embeddings.AudioTextVideoEmbedding(
        **model_config,
        word_embedding_matrix=dummy_embedding_matrix)
    model_result = module(images=images,
                          audio_spectrogram=audio_spectrogram,
                          word_ids=word_ids,
                          is_training=is_training,
                          # return_intermediate_audio=True
                          )

    if FLAGS.use_audio:
        clf_logits = mlp_builder(input_size=FLAGS.mlp_first_layer_size,
                                 input=jnp.concatenate((model_result['vid_repr'], model_result['aud_repr']), axis=1))
        return clf_logits, model_result['vid_repr'], model_result['aud_repr']
    else:
        clf_logits = mlp_builder(input_size=FLAGS.mlp_first_layer_size, input=model_result['vid_repr'])
        return clf_logits, model_result['vid_repr']


def load_model(FLAGS):
    with open(FLAGS.checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)
    params = checkpoint['params']
    state = checkpoint['state']
    return params, state


def main(argv):
    del argv
    model_config = config.get_model_config(FLAGS.checkpoint_path)
    params, state = load_model(FLAGS)
    forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))

    dataset_configuration = DATASET_CONFIG.get(FLAGS.dataset_name, None)
    if dataset_configuration is None:
        raise Exception('Please use an available dataset!')
    dataset_folder = dataset_configuration.get(DATASET_FOLDER)
    dset_config = DeepfakeConfig(name=FLAGS.dataset_name)
    test_ds = generate_dataset(dset_config,
                               dataset_configuration,
                               model_config,
                               dataset_name=FLAGS.dataset_name,
                               FLAGS=FLAGS,
                               train=False)
    # TODO - change to train in finetune file
    dataset_path = os.path.join(dataset_folder, DATASET_MIDFIX,
                                f"ZIP.{dataset_configuration.get(DIRECTORY_NAME)}_test.zip")
    files_in_dir = os.listdir(dataset_path)
    dummy_audio, dummy_word_ids = get_dummy_data(FLAGS)

    proper_audio_files_path = os.path.join(dataset_configuration[RELATIVE_PATH], AUDIO_FILES_JSON)
    if path.exists(proper_audio_files_path):
        with open(proper_audio_files_path, 'rb') as f:
            proper_audio_files = json.load(f)
    else:
        proper_audio_files = None

    test_labels, files_names, test_softmax = [], [], []
    for test_chunk in test_ds:
        test_chunk_tfds = tfds.as_numpy(test_chunk)
        if FLAGS.use_audio:
            audio = define_usable_audio(test_chunk_tfds, files_in_dir, FLAGS, proper_audio_files)
        else:
            audio = dummy_audio
        file_name = test_chunk_tfds['path'][0].decode('utf-8')
        output = forward.apply(
            params=params,
            state=state,
            images=test_chunk_tfds['video'],
            audio_spectrogram=audio,
            word_ids=dummy_word_ids,
            is_training=False,
            model_config=model_config,
        )
        logits = output[0]
        softmax = jax.nn.softmax(logits[0])
        test_softmax.append(softmax)
        test_labels.append(test_chunk_tfds['label'])
        files_names.append(np.array([file_name]))
    test_softmax = np.concatenate(test_softmax, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    files_names = np.concatenate(files_names, axis=0)

    pred_test_reshaped = np.reshape(test_softmax, (len(test_labels), -1, FLAGS.num_test_windows))
    pred_test_smoothed = pred_test_reshaped[:, :, 0:FLAGS.num_test_windows].mean(axis=2)
    pred_test_smoothed = pred_test_smoothed[:, 0]
    results_pd = pd.DataFrame(zip(pred_test_smoothed, test_labels, files_names),
                              columns=['softmax', 'label', 'filename'])
    results_pd = union_part_videos(results_pd)
    softmax = results_pd['softmax']
    labels = results_pd['label']
    print(f"ROC-AUC score: {round(get_auc_score(labels, softmax) * 100, 1)}")


if __name__ == '__main__':
    app.run(main)
