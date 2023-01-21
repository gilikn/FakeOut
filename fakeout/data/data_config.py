DATASET_MIDFIX = "downloads/extracted"
DATASET_FOLDER = "dataset_folder"
RELATIVE_PATH = 'relative_path'
DIRECTORY_NAME = 'directory_name'
AUDIO_FILES_JSON = 'proper_audio_files.json'
LABELS_PATH = 'labels.csv'
TRAIN_SET = "train_set"
TEST_SET = "test_set"
SPLITS_DIR = "train_test_split.zip"
DEFAULT_SPLIT_NUMBER = 1
WAV_BITRATE = 44100.0
FPS = 29.0

DATASET_CONFIG = {
    "dfdc": {
        RELATIVE_PATH: "fakeout/data/DFDC",
        DATASET_FOLDER: "~/DFDC/tmp",
        DIRECTORY_NAME: 'DFDC',
        TRAIN_SET: "dfdc_train_set",
        TEST_SET: "dfdc_test_set",
    },
    "celeb_df": {
        RELATIVE_PATH: "fakeout/data/Celeb_DF_v2",
        DATASET_FOLDER: "~/Celeb_DF_v2/tmp",
        DIRECTORY_NAME: 'Celeb_DF_v2',
        TRAIN_SET: "celeb_df_train_set",
        TEST_SET: "celeb_df_test_set",
    },
    "deeper_forensics": {
        RELATIVE_PATH: "fakeout/data/DeeperForensics",
        DATASET_FOLDER: "~/DeeperForensics/tmp",
        DIRECTORY_NAME: 'DeeperForensics',
        TRAIN_SET: "deeper_forensics_train_set",
        TEST_SET: "deeper_forensics_test_set",
    },
    "face_shifter": {
        RELATIVE_PATH: "fakeout/data/FaceShifter",
        DATASET_FOLDER: "~/FaceShifter/tmp",
        DIRECTORY_NAME: 'FaceShifter',
        TRAIN_SET: "face_shifter_forensics_train_set",
        TEST_SET: "face_shifter_forensics_test_set",
    },
    "face_forensics": {
        RELATIVE_PATH: "fakeout/data/FaceForensics",
        DATASET_FOLDER: "~/FaceForensics/tmp",
        DIRECTORY_NAME: 'FaceForensics',
        TRAIN_SET: "face_forensics_forensics_train_set",
        TEST_SET: "face_forensics_forensics_test_set",
    }
}
