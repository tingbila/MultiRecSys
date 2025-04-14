Config = {
        "model_name": "xDeepFm",
        "large_file": False,
        'use_cuda': False,
        'device_id': 0,
        "num_workers": 8,
        'pretrain': False,
        'model_dir': '../../cache/track2/checkpoints/lgb.model',
        "train_path": "../../cache/track2/tmp/train.csv",  # or ../../data/track2/final_track2_train.txt # train + val
        "val_path": "../../cache/track2/tmp/val.csv",
        "test_path": "../../cache/track2/tmp/test.csv",
        "all_data_path": "../../cache/track2/tmp/all_data.csv",
        "raw_data_path": "../../data/track2/final_track2_train.txt",
        "predict_file": "../../cache/track2/result/result.csv",
        # for lgb
        "save_test_path": "../../cache/track2/tmp/test_lgb.csv",
        "save_all_data_path": "../../cache/track2/tmp/all_data_lgb.csv",
        "save_train_path": "../../cache/track2/tmp/train_lgb.csv",
        "save_val_path": "../../cache/track2/tmp/val_lgb.csv",
        # title data
        "raw_title_data_path": "../../data/track2/track2_title.txt",
        "title_data_path": "../../cache/track2/tmp/title_data.json",
        # face data
        "raw_face_data_path": "../../cache/track2/tmp/track2_face_attrs_fill.csv",
        "face_data_path": "../../cache/track2/tmp/face_data.json",
        # video data
        "raw_video_data_path": "../../data/track2/track2_video_features.txt",
        "video_data_path": "../../cache/track2/tmp/video_data.json",
        # audio data
        "raw_audio_data_path": "../../data/track2/track2_audio_features.txt",
        "audio_data_path": "../../cache/track2/tmp/audio_data.json"
    }
normal_config = {
    "title": 134409 + 2,
    "male_perc": 96 + 2,
    "female_perc": 90 + 2,
    "faces": 91 + 2,
    "maleBeauty": 69 + 2,
    "femaleBeauty": 72 + 2,
    "faceSquare": 92 + 2,
    "male_num": 43 + 2,
    "female_num":73 + 2
}