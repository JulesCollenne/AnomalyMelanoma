import numpy as np
import scipy
from tqdm import tqdm


class KNNOutlier:
    def __init__(self, distance_name="norm", k=3, centroid_if_none=False):
        self.originals = None
        self.mean_of_mean = None
        self.distance_name = distance_name
        self.k = k
        self.max_distance = 0
        self.centroid_if_none = centroid_if_none
        if self.distance_name == "norm":
            self.compute_distance = np.linalg.norm
            self.use_difference = True
        elif self.distance_name == "cosine":
            self.compute_distance = scipy.spatial.distance.cosine
            self.use_difference = False
        elif self.distance_name == "manhattan":
            self.compute_distance = scipy.spatial.distance.cityblock
            self.use_difference = False
        elif self.distance_name == "mahalanobis":
            self.compute_distance = scipy.spatial.distance.mahalanobis
            self.use_difference = False
        else:
            raise NotImplementedError("Unknown distance")

    # TODO a mettre a jour
    # def fit(self, points):
    #     mean_distances = []
    #     self.originals = np.copy(points)
    #     distances = np.zeros((len(points), len(points)))
    #     for i in range(len(points) - 1):
    #         for j in range(i + 1, len(points)):
    #             res = self.distance(points[i], points[j])
    #             distances[i][j] = res
    #             distances[j][i] = res
    #         mean_distances.append(np.mean(distances[i][np.argpartition(distances[i], self.k)[:self.k]]))
    #         # mean_distances.append(np.mean([dist for dist in distances[:k]]))
    #     mean_distances.append(np.mean(distances[-1][np.argpartition(distances[-1], self.k)[:self.k]]))
    #     self.mean_of_mean = np.mean(mean_distances)
    #     return self
    #
    # def predict(self, points):
    #     if not points:
    #         print("List is empty, aborting predictions")
    #         exit(0)
    #     mean_distance = 0
    #     for point in points:
    #         distances = []
    #         for original in self.originals:
    #             distances.append(self.distance(point, original))
    #         distances = np.asarray(distances)
    #         mean_distance = np.mean(distances[np.argpartition(distances, self.k)[:self.k]])
    #     return mean_distance / self.mean_of_mean

    def fit_predict(self, points):
        # if 0 < self.k <= len(points):
        if 0 < self.k < len(points)-1:
            if len(points) == 1:
                return np.asarray(0)
            else:
                mean_distances = []
                distances = np.zeros((len(points), len(points)))
                for i in range(len(points) - 1):
                    for j in range(i + 1, len(points)):
                        res = self.distance(points[i], points[j])
                        distances[i][j] = res
                        distances[j][i] = res
                    mean_distances.append(np.mean(distances[i][np.argpartition(distances[i], self.k+1)[:self.k+1]]))
                    # mean_distances.append(np.mean([dist for dist in distances[:k]]))
                mean_distances.append(np.mean(distances[-1][np.argpartition(distances[-1], self.k+1)[:self.k+1]]))
                self.max_distance = np.max(mean_distances)
                mean_of_mean = np.mean(mean_distances)
                outliers = []
                for i, distance in enumerate(mean_distances):
                    outliers.append(distance / mean_of_mean)
                return np.asarray(outliers)

        elif self.k == 0 or (len(points)-1 <= self.k and self.centroid_if_none):
            centroid = centeroidnp(points)
            distances = []
            for i in range(len(points)):
                distances.append(self.distance(centroid, points[i]))
            return np.asarray(distances) / np.mean(distances)
        # elif self.k == 0:
        #     centroid = centeroidnp(points)
        #     distances = []
        #     for i in range(len(points)):
        #         distances.append(self.distance(centroid, points[i]))
        #     return np.asarray(distances) / np.mean(distances)

    def distance(self, u, v):
        if self.distance_name == "mahalanobis":
            cov = np.cov(np.array([u, v]).T)
            inv_cov = np.linalg.pinv(cov)
            return self.compute_distance(u, v, inv_cov)
        if self.use_difference:
            return self.compute_distance(u - v)
        else:
            return self.compute_distance(u, v)


def centeroidnp(arr):
    length = arr.shape[0]
    la_sum = []
    for i in range(arr.shape[1]):
        la_sum.append(np.sum(arr[:, i]))
    return np.asarray(la_sum) / length


def compute_vpc(feat_df, k=6, distance_name="norm", vpc_def_val=0., normalize=True, centroid_if_none=False, skip_lownb=False):
    """
    Add the column "VPC" to the df, corresponding to the ugly duckling rate value :param normalize: True for
    normalization between 0 and 1
    :param feat_df: the original dataframe
    :param k: argument for k nearest neighbors
    :param normalize: True to normalize values
    :param distance_name: norm, cosine, manhattan, mahalanobis
    :param vpc_def_val: int or string. default value of the vpc. default = 0. "median" or "average" are also possible
    values
    :return: feat_df with a new column "VPC"
    """
    # if not isinstance(vpc_def_val, str):
    #     feat_df['VPC'] = np.ones(len(feat_df)) * vpc_def_val
    # else:
    feat_df['VPC'] = np.ones(len(feat_df)) * -1
    for patient_id in tqdm(np.unique(feat_df['patient_id'])):
        images = feat_df[feat_df['patient_id'] == patient_id][['image_name']]
        x = feat_df[feat_df['image_name'].isin(images['image_name'])].filter(like='feature', axis=1)
        x = x.to_numpy()
        if x.shape[0] <= k + 1 and skip_lownb:
            continue
        knn = KNNOutlier(k=k, distance_name=distance_name, centroid_if_none=centroid_if_none)
        predictions = knn.fit_predict(x)
        vpc_names = images['image_name']
        for i, name in enumerate(vpc_names):
            feat_df.loc[feat_df['image_name'] == name, 'VPC'] = predictions[i]
    return feat_df
