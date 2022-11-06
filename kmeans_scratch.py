import pandas as pd
import random
from vincenty import vincenty

class KMeansAlgorithm:
    def __init__(self, df: pd.DataFrame, K:int, lat_column, long_column, weight_factor=None):
        self.K = K
        self.df = df
        self.lat_data = df[[lat_column]].tolist()
        self.long_data = df[[long_column]].tolist()
        self.n_population = len(df)
        self.classification = [-1 for _ in range(self.n_population)]
        self.weight_factor = weight_factor if not None else [1 for _ in range(self.n_population)]
        self.wcss = 0


    def __init_centroid(self):
        """
        format of centroid is (lat, long)
        :return:
        """
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
        dframe.assign(lat=self.lat_data, long=self.long_data, pred=self.classification, idx=[i for i in range(self.n_population)])

        for i in range(self.K):
            filtered_data = dframe[dframe.pred==i]
            new_lat = self.__calculate_mean(filtered_data[["lat"]].tolist(), self.weight_factor, self.n_population)
            new_long = self.__calculate_mean(filtered_data[["long"]].tolist(), self.weight_factor, self.n_population)
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

