import numpy as np
import plotly.express as px
import sklearn.cluster
from numpy import pi, sin,cos, arctan2, sqrt
import pandas as pd
def haversine_distance(p1,p2):
    lg1 = p1[0]
    lat1 = p1[1]
    lg2 = p2[0]
    lat2 = p2[1]

    R = 6371000
    phi1 = lat1 * pi / 180 #convert to radian
    phi2 = lat2 * pi / 180 #convert to radian
    delta_phi = (lat2 - lat1) * pi / 180
    delta_lambda = (lg2 - lg1) * pi / 180

    a = (sin(delta_phi/2))**2 + cos(phi1) * cos(phi2) * ((sin(delta_lambda/2))**2)
    c = 2 * arctan2(sqrt(a), sqrt(1-a))
    distance = R * c #haversine distance between point1 and point 2 in meters
    return round(distance, 2)

class KMeansClustering:
    def __init__(self, X, num_clusters):
        self.K = num_clusters  # cluster number
        self.max_iterations = 1000  # max iteration. don't want to run inf time
        self.num_examples, self.num_features = X.shape  # num of examples, num of features
        self.plot_figure = False  # plot figure

    # randomly initialize centroids
    def initialize_random_centroids(self, X):
        centroids = np.zeros((self.K, self.num_features))  # row , column full with zero
        for k in range(self.K):  # iterations of
            centroid = X[np.random.choice(range(self.num_examples))]  # random centroids
            centroids[k] = centroid
        return centroids  # return random centroids

    # create cluster Function
    def create_cluster(self, X, centroids):
        clusters = [[] for _ in range(self.K)]
        for point_idx, point in enumerate(X):
            #TODO: Eval fungsi ini, masih perlu perbaikan
            closest_centroid = np.argmin(
                np.sqrt(np.sum((point - centroids) ** 2, axis=1))
            )  # closest centroid using euler distance equation(calculate distance of every point from centroid)
            clusters[closest_centroid].append(point_idx)
        return clusters

        # new centroids

    def calculate_new_centroids(self, cluster, X):
        centroids = np.zeros((self.K, self.num_features))  # row , column full with zero
        for idx, cluster in enumerate(cluster):
            new_centroid = np.mean(X[cluster], axis=0)  # find the value for new centroids
            centroids[idx] = new_centroid
        return centroids

    # prediction
    def predict_cluster(self, clusters, X):
        y_pred = np.zeros(self.num_examples)  # row1 fillup with zero
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                y_pred[sample_idx] = cluster_idx
        return y_pred

    # plotinng scatter plot
    def plot_fig(self, X, y):
        fig = px.scatter(X[:, 0], X[:, 1], color=y)
        fig.show()  # visualize

    # fit data
    def fit(self, X):
        centroids = self.initialize_random_centroids(X)  # initialize random centroids
        for _ in range(self.max_iterations):
            clusters = self.create_cluster(X, centroids)  # create cluster
            previous_centroids = centroids[:]
            centroids = self.calculate_new_centroids(clusters, X)  # calculate new centroids
            diff = centroids - previous_centroids  # calculate difference
            if not diff.any():
                break
        y_pred = self.predict_cluster(clusters, X)  # predict function
        if self.plot_figure:  # if true
            self.plot_fig(X, y_pred)  # plot function
        return y_pred

if __name__ == "__main__":
    flname = "dataset_cluster_warehouse_exp_1.csv"
    current_loc = (-6.221509, 106.819269)
    df = pd.read_csv(flname)

    # add distancce in dataframe from current loc
    l_distance = []
    data_lat = []
    data_long = []
    for idx, row in df.iterrows():
        point = row['lat_long'].split(',')
        lat = float(point[0])
        long = float(point[1])
        data_lat.append(lat)
        data_long.append(long)
        l_distance.append(haversine_distance((lat, long), current_loc))

    df = df.assign(distance=l_distance, lat=data_lat, long=data_long)
    dataset = df[['city_name', 'district_name', 'avg_demand_baseline', 'demand_target', 'lat', 'long']]

    unscaled_dataset = (dataset[['lat', 'long']]).to_numpy()
    print(unscaled_dataset)
    km = KMeansClustering(unscaled_dataset,6)
    y_pred = km.fit(unscaled_dataset)
    print(y_pred)