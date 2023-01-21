from typing import Optional

import jax
import jax.numpy as jnp
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_io as tfio

from fakeout.data.data_config import WAV_BITRATE, FPS, DATASET_FOLDER, LABELS_PATH
from fakeout.utils import deepfake_dataset_new


def define_usable_audio(ex_tfds, files_in_dir, FLAGS, proper_audio_files, train=True):
    filenames = list(map(lambda x: x.decode('utf-8'), ex_tfds['path']))
    audios = ex_tfds['audio'][..., np.newaxis]
    dummy_and_real_audios = np.zeros(shape=audios.shape)
    relevant_indexes = []
    for i, name in enumerate(filenames):
        wav_path = f"{name.split('.')[0]}.wav"
        if wav_path in files_in_dir:
            if train:
                if proper_audio_files is not None:
                    if name.split('.')[0] in proper_audio_files:
                        # todo - VALIDATE check if the file should really get audio - based on csvs...
                        relevant_indexes.append(i)
            else:
                offset = FLAGS.num_test_windows * i
                relevant_indexes.append([j for j in range(offset, offset + FLAGS.num_test_windows)])
    dummy_and_real_audios[relevant_indexes] = audios[relevant_indexes, :, :, :]
    return jax.numpy.asarray(dummy_and_real_audios)


def get_dummy_text(FLAGS):
    num_tokens = 16
    dummy_word_ids = jnp.zeros(shape=(FLAGS.eval_batch_size, num_tokens), dtype=jnp.int32)
    return dummy_word_ids


def get_dummy_audio(FLAGS):
    audio_frames = 96
    mel_filters = 40
    dummy_audio = jnp.zeros(shape=(FLAGS.eval_batch_size * FLAGS.num_test_windows, audio_frames, mel_filters, 1))
    return dummy_audio


def get_dummy_data(FLAGS):
    dummy_audio = get_dummy_audio(FLAGS)
    dummy_word_ids = get_dummy_text(FLAGS)
    return dummy_audio, dummy_word_ids


def generate_dataset(dset_config, dataset_configuration, model_config, dataset_name, FLAGS, train=True):
    dataset_folder = dataset_configuration.get(DATASET_FOLDER)
    labels_path = dataset_configuration.get(LABELS_PATH)
    builder = deepfake_dataset_new.Deepfake(dataset_name=dataset_name,
                                            train=train,
                                            data_dir=dataset_folder,
                                            config=dset_config,
                                            labels_path=labels_path
                                            )

    # Create the tfrecord files (no-op if already exists)
    dl_config = tfds.download.DownloadConfig(verify_ssl=False)
    builder.download_and_prepare(download_config=dl_config)

    if train:
        # Generate the training dataset.
        train_ds = builder.as_dataset(split='train', shuffle_files=True)
        train_ds = train_ds.map(lambda x: process_samples(x,
                                                          num_frames=FLAGS.num_frames,
                                                          is_training=True,
                                                          mel_filters=FLAGS.mel_filters,
                                                          FLAGS=FLAGS))
        train_ds = train_ds.batch(batch_size=FLAGS.train_batch_size)
        if model_config['visual_backbone'] == 's3d':
            train_ds = train_ds.map(space_to_depth_batch)
        train_ds = train_ds.repeat(FLAGS.num_train_epochs)
        return train_ds

    else:  # test
        # Generate the test dataset.
        test_ds = builder.as_dataset(split='test', shuffle_files=False)
        test_ds = test_ds.map(map_func=lambda x: process_samples(x,
                                                                 num_frames=FLAGS.num_frames,
                                                                 is_training=False,
                                                                 mel_filters=FLAGS.mel_filters,
                                                                 num_windows=FLAGS.num_test_windows,
                                                                 FLAGS=FLAGS))
        test_ds = test_ds.batch(batch_size=FLAGS.eval_batch_size)
        test_ds = test_ds.map(lambda x: reshape_windows(x, FLAGS))
        if model_config['visual_backbone'] == 's3d':
            test_ds = test_ds.map(space_to_depth_batch)
        test_ds = test_ds.repeat(1)
        return test_ds


def flip_left_right(video):
    is_flipped = tf.random.uniform((), minval=0, maxval=2, dtype=tf.int32)
    video = tf.cond(tf.equal(is_flipped, 1),
                    true_fn=lambda: tf.image.flip_left_right(video),
                    false_fn=lambda: video)
    return video


def random_brightness(video):
    is_brightness_randomized = tf.random.uniform((), minval=0, maxval=2, dtype=tf.int32)
    video = tf.cond(tf.equal(is_brightness_randomized, 1),
                    true_fn=lambda: tf.image.random_brightness(video, max_delta=32 / 255),
                    false_fn=lambda: video)
    return video


def random_hue(video):
    is_hue_adjusted = tf.random.uniform((), minval=0, maxval=2, dtype=tf.int32)
    video = tf.cond(tf.equal(is_hue_adjusted, 1),
                    true_fn=lambda: tf.image.random_hue(video, max_delta=0.2),
                    false_fn=lambda: video)
    return video


def augment_video(video):
    """Augment video frames"""
    video = flip_left_right(video)
    video = random_brightness(video)
    video = random_hue(video)
    return video


def process_samples(features_dict, FLAGS, num_frames=32, stride=2, is_training=True, num_windows=1, mel_filters=80):
    """Process video frames."""
    video = features_dict['video']
    if FLAGS.use_audio:
        audio = features_dict['audio']
    if is_training:
        assert num_windows == 1
        video = sample_linspace_sequence(video, num_windows, num_frames, stride)
        if FLAGS.use_audio:
            audio = mel_spectogram_of_sequence(audio[:, 0][0:int((WAV_BITRATE / FPS) * 32)], mel_filters)
        video = augment_video(video)
    else:
        video = sample_linspace_sequence(video, num_windows, num_frames, stride)
        if FLAGS.use_audio:
            audio = sample_linspace_sequence(audio[:, 0], num_windows, int((WAV_BITRATE / FPS) * 32), stride=1)
            audio = tf.reshape(audio, (-1, int((WAV_BITRATE / FPS) * 32)))
            audio = alternative_mel_spectrogram_of_sequence(audio, mel_filters)
            # todo - keep "alternative" just after the results of INFERENCE are the same. change name to not "alternative"
    # Resize smallest side.
    video = tf.cast(video, tf.float32)
    video /= 255.0  # Set between [0, 1].

    features_dict['video'] = video
    if FLAGS.use_audio:
        features_dict['audio'] = audio
    return features_dict


def space_to_depth_batch(features_dict):
    images = features_dict['video']
    _, l, h, w, c = images.shape
    images = tf.reshape(images, [-1, l // 2, 2, h // 2, 2, w // 2, 2, c])
    images = tf.transpose(images, [0, 1, 3, 5, 2, 4, 6, 7])
    images = tf.reshape(images, [-1, l // 2, h // 2, w // 2, 8 * c])
    features_dict['video'] = images
    return features_dict


def reshape_windows(features_dict, FLAGS):
    num_frames = FLAGS.num_frames
    x = features_dict['video']
    if FLAGS.use_audio:
        y = features_dict['audio']
    x = tf.reshape(x, (-1, num_frames, x.shape[2], x.shape[3], x.shape[4]))
    if FLAGS.use_audio:
        y = tf.reshape(y, (-1, y.shape[2], y.shape[3]))
    features_dict['video'] = x
    if FLAGS.use_audio:
        features_dict['audio'] = y

    return features_dict


def get_sampling_offset(sequence: tf.Tensor,
                        num_steps: Optional[int],
                        is_training: bool,
                        stride: int = 1,
                        seed: Optional[int] = None) -> tf.Tensor:
    """Calculates the initial offset for a sequence where all steps will fit.

    Args:
      sequence: any tensor where the first dimension is timesteps.
      num_steps: The number of timesteps we will output. If None,
        deterministically start at the first frame.
      is_training: A boolean indicates whether the graph is for training or not.
        If False, the starting frame always the first frame.
      stride: distance to sample between timesteps.
      seed: a deterministic seed to use when sampling.
    Returns:
      The first index to begin sampling from. A best effort is made to provide a
      starting index such that all requested steps fit within the sequence (i.e.
      offset + 1 + (num_steps - 1) * stride < len(sequence)). If this is not
      satisfied, the starting index is chosen randomly from the full sequence.
    """
    if num_steps is None or not is_training:
        return tf.constant(0)
    sequence_length = tf.shape(sequence)[0]
    max_offset = tf.cond(
        tf.greater(sequence_length, (num_steps - 1) * stride),
        lambda: sequence_length - (num_steps - 1) * stride,
        lambda: sequence_length)
    offset = tf.random.uniform(
        (),
        maxval=tf.cast(max_offset, tf.int32),
        dtype=tf.int32,
        seed=seed)
    return offset


def sample_or_pad_sequence_indices(sequence: tf.Tensor,
                                   num_steps: Optional[int],
                                   is_training: bool,
                                   repeat_sequence: bool = True,
                                   stride: int = 1,
                                   offset: Optional[int] = None) -> tf.Tensor:
    """Returns indices to take for sampling or padding a sequence to fixed size.

    Samples num_steps from the sequence. If the sequence is shorter than
    num_steps, the sequence loops. If the sequence is longer than num_steps and
    is_training is True, then we seek to a random offset before sampling. If
    offset is provided, we use that deterministic offset.

    This method is appropriate for sampling from a tensor where you want every
    timestep between a start and end time. See sample_stacked_sequence_indices for
    more flexibility.

    Args:
      sequence: any tensor where the first dimension is timesteps.
      num_steps: how many steps (e.g. frames) to take. If None, all steps from
        start to end are considered and `is_training` has no effect.
      is_training: A boolean indicates whether the graph is for training or not.
        If False, the starting frame is deterministic.
      repeat_sequence: A boolean indicates whether the sequence will repeat to
        have enough steps for sampling. If False, a runtime error is thrown if
        num_steps * stride is longer than sequence length.
      stride: distance to sample between timesteps.
      offset: a deterministic offset to use regardless of the is_training value.

    Returns:
      Indices to gather from the sequence Tensor to get a fixed size sequence.
    """
    sequence_length = tf.shape(sequence)[0]
    sel_idx = tf.range(sequence_length)

    if num_steps:
        if offset is None:
            offset = get_sampling_offset(sequence, num_steps, is_training, stride)

        if repeat_sequence:
            # Repeats sequence until num_steps are available in total.
            num_repeats = tf.cast(
                tf.math.ceil(
                    tf.math.divide(
                        tf.cast(num_steps * stride + offset, tf.float32),
                        tf.cast(sequence_length, tf.float32)
                    )), tf.int32)
            sel_idx = tf.tile(sel_idx, [num_repeats])
        steps = tf.range(offset, offset + num_steps * stride, stride)
    else:
        steps = tf.range(0, sequence_length, stride)
    return tf.gather(sel_idx, steps)


def random_sample_sequence(sequence: tf.Tensor,
                           num_steps: int,
                           stride: int = 1) -> tf.Tensor:
    """Randomly sample a segment of size num_steps from a given sequence."""

    indices = sample_or_pad_sequence_indices(
        sequence=sequence,
        num_steps=num_steps,
        is_training=True,  # Random sample.
        repeat_sequence=True,  # Will repeat the sequence if request more.
        stride=stride,
        offset=None)
    indices.set_shape((num_steps,))
    output = tf.gather(sequence, indices)
    return output


def sample_linspace_sequence(sequence: tf.Tensor,
                             num_windows: int,
                             num_steps: int,
                             stride: int = 1) -> tf.Tensor:
    """Samples num_windows segments from sequence with linearly spaced offsets.

    The samples are concatenated in a single Tensor in order to have the same
    format structure per timestep (e.g. a single frame). If num_steps * stride is
    bigger than the number of timesteps, the sequence is repeated. This function
    can be used in evaluation in order to extract enough segments in order to span
    the entire sequence.

    Args:
      sequence: Any tensor where the first dimension is timesteps.
      num_windows: Number of windows retrieved from the sequence.
      num_steps: Number of steps (e.g. frames) to take.
      stride: Distance to sample between timesteps.

    Returns:
      A single Tensor with first dimension num_windows * num_steps. The Tensor
      contains the concatenated list of num_windows tensors which offsets have
      been linearly spaced from input.
    """
    sequence_length = tf.shape(sequence)[0]
    max_offset = tf.maximum(0, sequence_length - num_steps * stride)
    offsets = tf.linspace(0.0, tf.cast(max_offset, tf.float32), num_windows)
    offsets = tf.cast(offsets, tf.int32)

    all_indices = []
    for i in range(num_windows):
        all_indices.append(
            sample_or_pad_sequence_indices(
                sequence=sequence,
                num_steps=num_steps,
                is_training=False,
                repeat_sequence=True,  # Will repeat the sequence if request more.
                stride=stride,
                offset=offsets[i]))

    indices = tf.concat(all_indices, axis=0)
    indices.set_shape((num_windows * num_steps,))
    output = tf.gather(sequence, indices)

    return output


def _tf_log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def power_to_db(magnitude, amin=1e-16, top_db=80.0):
    """
    https://librosa.github.io/librosa/generated/librosa.core.power_to_db.html
    """
    ref_value = tf.reduce_max(magnitude)
    log_spec = 10.0 * _tf_log10(tf.maximum(amin, magnitude))
    log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref_value))
    log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

    return log_spec


def alternative_mel_spectrogram_of_sequence(audio, mel_filters):
    mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=mel_filters,
        num_spectrogram_bins=2048 // 2 + 1,
        sample_rate=44100,
        lower_edge_hertz=0,
        upper_edge_hertz=44100 / 2)
    spectrograms = tf.signal.stft(audio,
                                  frame_length=2048,
                                  frame_step=256,
                                  pad_end=False)
    magnitude_spectrograms = tf.abs(spectrograms)

    mel_spectrograms = tf.matmul(tf.square(magnitude_spectrograms),
                                 mel_filterbank)

    log_mel_spectrograms = power_to_db(mel_spectrograms)
    return log_mel_spectrograms


# TODO - RETURN TO NORMAL
def mel_spectogram_of_sequence(audio, mel_filters):
    spectrogram = tfio.experimental.audio.spectrogram(audio, nfft=2048, window=2048, stride=256)
    mel_spectrogram = tfio.experimental.audio.melscale(spectrogram, rate=WAV_BITRATE, mels=mel_filters, fmin=0,
                                                       fmax=10000)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 0.00001)
    return log_mel_spectrogram
