from datetime import datetime
from pathlib import Path
from platform import platform

import pandas as pd
from sklearn import preprocessing

from outlier_detection.knnoutlier import compute_vpc
from train_full_pipeline import Normalizer
from utils import path_to_windows


def run(hyperparameters):
    # k = 3
    vpc_def_val = 0.
    # distance_name = "norm"
    distance_name = "cosine"
    dim = 100
    centroid_if_none = True
    ladate = "2023-07-05"
    # scaler = preprocessing.MinMaxScaler()
    scaler = preprocessing.StandardScaler()
    # features_path = "features/mocov3/efficientnet_b2/260/"
    features_path = "features/binary_cnn/2048/"
    # features_path = "features/simsiam/resnet50/224/"
    # features_path = "features/100/"

    # gt_path = "/home/jules/Travail/isic/ISIC_2020/GroundTruth.csv"
    # splitted_path = "/".join(features_path.split('/')[1:])
    # outfile = "features/with_vpc/" + splitted_path
    # gt_path = "/home/jules/Travail/isic/ISIC_2020/y_train.csv"
    gt_path = "/media/jules/Transcend/Datasets/isic/ISIC_2020/y_train.csv"
    splitted_path = "/".join(features_path.split('/')[1:])
    outfile = "features/with_vpc/" + splitted_path

    if "Linux" not in platform():
        outfile = path_to_windows(outfile)
        gt_path = "ISIC_2020_Training_GroundTruth.csv"
    if "Ground" in gt_path:
        full_df = pd.read_csv(gt_path)
        full_df = full_df.rename(columns={"path": "image_name"})
        full_df["image_name"] = full_df["image_name"].apply(lambda x: x[:-4])
    else:
        full_df = pd.read_csv(gt_path)

    # k = -1
    # # normalizer = Normalizer(vpc_def_val, scaler=preprocessing.QuantileTransformer())
    # normalizer = Normalizer(vpc_def_val, scaler=preprocessing.StandardScaler())
    # for le_set in "train", "val", "test":
    #     print(f"Computing vpc for {le_set}...")
    #     df = pd.read_csv(f"{features_path}{le_set}_features.csv")
    #     df = correct_filename(df)
    #     df = df.merge(full_df[["image_name", "patient_id"]], on="image_name", how="inner")
    #     feat_df = compute_vpc(df, distance_name=distance_name, k=k, vpc_def_val=vpc_def_val)
    #     feat_df = normalizer.normalize_and_fillnan(feat_df, is_train=le_set == "train")
    #     feat_df = feat_df.drop('patient_id', axis=1)
    #     feat_df.to_csv(outfile + le_set + f"_features_{k}_centroid.csv", index=False)
    for k in range(11):
        print(f"k = {k}")
        # normalizer = Normalizer(vpc_def_val, scaler=preprocessing.StandardScaler())
        normalizer = Normalizer(scaler=scaler, default_value=vpc_def_val)
        les_sets = "train", "val", "test"
        # les_sets = "train", "test"
        # les_sets = "train", "val"
        for le_set in les_sets:
            # gt_path = f"/home/jules/Travail/isic/ISIC_2020/y_{le_set}.csv"
            gt_path = f"/media/jules/Transcend/Datasets/isic/ISIC_2020/y_{le_set}.csv"
            full_df = pd.read_csv(gt_path)
            print(f"Computing vpc for {le_set}...")
            # df = pd.read_csv(f"{features_path}{le_set}_{dim}.csv")
            # df = pd.read_csv(features_path+f"2020{le_set}_100_norm.csv")
            df = pd.read_csv(features_path+f"2020{le_set}_{dim}_{ladate}_norm.csv")
            if "Ground" in df.keys():
                df = correct_filename(df)
            else:
                df = df.rename(columns={"filename": "image_name"})
            df = df.merge(full_df[["image_name", "patient_id"]], on="image_name", how="inner")
            feat_df = compute_vpc(df, distance_name=distance_name, k=k, vpc_def_val=vpc_def_val,
                                  centroid_if_none=centroid_if_none)
            # normalizer = Normalizer(vpc_def_val, scaler=preprocessing.QuantileTransformer())
            feat_df = normalizer.normalize_and_fillnan(feat_df, is_train=le_set == "train")
            # feat_df = normalizer.normalize_and_fillnan(feat_df, is_train=True)
            feat_df = feat_df.drop('patient_id', axis=1)
            feat_df.to_csv(outfile + le_set + f"_features_{dim}_{k}_normalized_{datetime.today().strftime('%Y-%m-%d')}"
                                              f"_{type(scaler).__name__}_{centroid_if_none}_CORRECT.csv", index=False)
    # print(feat_df)


def correct_filename(features):
    if "filename" in features.columns:
        features["image_name"] = features["filename"]
        features = features.drop("filename", axis=1)
        features["image_name"] = [name.split('/')[-1][:-4] for name in features["image_name"]]
    return features


if __name__ == "__main__":
    main()
