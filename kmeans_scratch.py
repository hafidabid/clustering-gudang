import matplotlib.pyplot as plt
import pandas as pd
import random
from vincenty import vincenty

class KMeansAlgorithm:
    def __init__(self, df: pd.DataFrame, K:int, lat_column, long_column, weight_factor=None):
        self.K = K
        self.df = df
        self.lat_data = df[lat_column].tolist()
        self.long_data = df[long_column].tolist()
        self.n_population = len(df)
        self.classification = [-1 for _ in range(self.n_population)]
        self.weight_factor = weight_factor if weight_factor is not None else [1 for _ in range(self.n_population)]
        self.wcss = 0
        self.centroid_lock = False

    def set_weight_factor(self, weight_factor):
        self.weight_factor = weight_factor

    def lock_init_centroid(self, stats=True):
        self.centroid_lock = stats
        self.__init_centroid()
        self.first_centroid = self.centroid.copy()


    def __init_centroid(self):
        """
        format of centroid is (lat, long)
        :return:
        """
        if self.centroid_lock:
            self.centroid = self.first_centroid.copy()
            return

        self.centroid = [(0, 0) for _ in range(self.K)]
        selected_index = []
        for i in range(self.K):
            r = random.randrange(0, self.n_population)
            while r in selected_index:
                r = random.randrange(0, self.n_population)
            selected_index.append(r)

        for i in range(self.K):
            self.centroid[i] = (
                self.lat_data[selected_index[i]],
                self.long_data[selected_index[i]],
            )

    def __calculate_mean(self, data, weight, n):
        s = 0
        w = 0
        for i in range(n):
            s += data[i] * weight[i]
            w += weight[i]

        return s/w

    def __calculate_new_centorid(self, old_centroid, *args):
        new_centroid = [(0, 0) for _ in range(self.K)]

        """pack into dataframe first"""
        dframe = pd.DataFrame()
        dframe = dframe.assign(lat=self.lat_data, long=self.long_data, pred=self.classification, idx=[i for i in range(self.n_population)])

        for i in range(self.K):
            filtered_data = dframe[dframe.pred==i]
            new_lat = self.__calculate_mean(filtered_data["lat"].tolist(), self.weight_factor, len(filtered_data))
            new_long = self.__calculate_mean(filtered_data["long"].tolist(), self.weight_factor, len(filtered_data))
            new_centroid[i] = (new_lat, new_long)

        return new_centroid

    def fit(self, max_iter, *args):
        self.__init_centroid()
        temp_classification = [-1 for _ in range(self.n_population)]
        iter = 0
        while iter < max_iter:
            """
            Calculate nearest centroid to join as cluster
            """
            self.wcss = 0
            for i_node in range(self.n_population):
                ichoice, dist = 0, vincenty((self.lat_data[i_node], self.long_data[i_node]), self.centroid[0])
                for i_cluster in range(1, self.K):
                    newdist = vincenty((self.lat_data[i_node], self.long_data[i_node]), self.centroid[i_cluster])
                    if newdist<dist:
                        ichoice, dist = i_cluster, newdist

                self.wcss += dist
                temp_classification[i_node] = ichoice


            if self.classification == temp_classification:
                iter = max_iter
            else:
                self.classification = temp_classification.copy()
                iter += 1

            """
            Recalculate the centroid
            """
            self.centroid = self.__calculate_new_centorid(self.centroid)

        return self.classification

    def predict(self):
        return self.classification

    def get_wcss(self):
        return self.wcss

    def visualize_kmeans(self, figsize=(6,6), centroid_size=None, title='Plot of K Means Clustering Algorithm'):
        centroid_size  = [350 for _ in range(self.K)] if centroid_size is None else centroid_size
        # create arrays for colors and labels based on specified K
        colors = ["red", "green", "blue", "purple", "black", "pink", "orange"]
        labels = ['cluster_' + str(i + 1) for i in range(self.K)]

        ax1 = plt.subplot(1,1,1)
        plt.figure(figsize=figsize)
        # plot each cluster
        dframe = pd.DataFrame()
        dframe = dframe.assign(lat=self.lat_data, long=self.long_data, pred=self.classification,
                               idx=[i for i in range(self.n_population)])
        for k in range(self.K):
            plt.scatter(dframe[dframe.pred==k]['long'].tolist(), dframe[dframe.pred==k]['lat'].tolist(),
                        c=colors[k], label=labels[k])
        # plot centroids
        i=0
        for c in self.centroid:
            plt.scatter(c[1], c[0],  # alpha=.5,
                        s=centroid_size[i], c='gold', label='cluster ke-'+str(i), marker="*")
            i+=1
        plt.xlabel("lat")  # first column of df
        plt.ylabel("long")  # second column of df
        plt.title(title)

        plt.legend()

        return plt.show(block=True)

if __name__ == "__main__":
    flname = "dataset_cluster_warehouse_exp_2.csv"
    df = pd.read_csv(flname)
    data_lat = []
    data_long = []
    for idx, row in df.iterrows():
        point = row['lat_long'].split(',')
        lat = float(point[0])
        long = float(point[1])
        data_lat.append(lat)
        data_long.append(long)

    df = df.assign(lat=data_lat, long=data_long)
    dataset = df[['city_name', 'district_name', 'avg_demand_baseline', 'demand_target', 'lat', 'long']]
    km = KMeansAlgorithm(dataset, 5, 'lat', 'long')
    km.fit(100)
    print(km.predict())
    km.visualize_kmeans()

