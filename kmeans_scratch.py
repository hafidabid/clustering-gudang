import matplotlib.pyplot as plt
import pandas as pd
import random
from vincenty import vincenty
import plotly.graph_objects as go

MAPBOX_API_KEY = "pk.eyJ1IjoiaGFmaWRhYmkiLCJhIjoiY2tuNXZ2N25uMDg1MjJyczlna3VndmFmNSJ9.VKoc34AfkqZ5uUUODIUBVA"


class KMeansAlgorithm:
    def __init__(
        self, df: pd.DataFrame, K: int, lat_column, long_column, weight_factor=None
    ):
        self.K = K
        self.df = df
        self.lat_data = df[lat_column].tolist()
        self.long_data = df[long_column].tolist()
        self.n_population = len(df)
        self.classification = [-1 for _ in range(self.n_population)]
        self.weight_factor = (
            weight_factor
            if weight_factor is not None
            else [1 for _ in range(self.n_population)]
        )
        self.wcss = 0
        self.centroid_lock = False

    def set_weight_factor(self, weight_factor):
        self.weight_factor = weight_factor

    def lock_init_centroid(self, stats=True):
        self.__init_centroid()
        self.centroid_lock = stats
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

        if w == 0:
            return 0
        return s / w

    def __calculate_new_centorid(self, old_centroid, *args):
        new_centroid = [(0, 0) for _ in range(self.K)]

        """pack into dataframe first"""
        dframe = pd.DataFrame()
        dframe = dframe.assign(
            lat=self.lat_data,
            long=self.long_data,
            pred=self.classification,
            idx=[i for i in range(self.n_population)],
        )

        for i in range(self.K):
            filtered_data = dframe[dframe.pred == i]
            new_lat = self.__calculate_mean(
                filtered_data["lat"].tolist(), self.weight_factor, len(filtered_data)
            )
            new_long = self.__calculate_mean(
                filtered_data["long"].tolist(), self.weight_factor, len(filtered_data)
            )
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
                ichoice, dist = 0, vincenty(
                    (self.lat_data[i_node], self.long_data[i_node]), self.centroid[0]
                )
                for i_cluster in range(1, self.K):
                    newdist = vincenty(
                        (self.lat_data[i_node], self.long_data[i_node]),
                        self.centroid[i_cluster],
                    )
                    if newdist < dist:
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

    def visualize_kmeans(
        self,
        figsize=(6, 6),
        centroid_size=None,
        title="Plot of K Means Clustering Algorithm",
        alpha_col=None,
    ):

        # create arrays for colors and labels based on specified K
        colors = ["red", "green", "blue", "purple", "black", "pink", "orange"]
        labels = ["cluster_" + str(i + 1) for i in range(self.K)]

        plt.figure(figsize=figsize)

        # plot each cluster
        dframe = pd.DataFrame()
        dframe = dframe.assign(
            lat=self.lat_data,
            long=self.long_data,
            pred=self.classification,
            idx=[i for i in range(self.n_population)],
            demand=self.df["demand_target"],
        )
        if alpha_col:
            dframe = dframe.assign(**{alpha_col: self.df[alpha_col]})
        for k in range(self.K):
            predicted_dataset = dframe[dframe.pred == k]
            # plt.scatter(predicted_dataset['long'].tolist(), predicted_dataset[dframe.pred==k]['lat'].tolist(),
            #   c=colors[k], label=labels[k])
            alp_min, alp_max = 0, 0
            if alpha_col:
                alp_max = predicted_dataset[alpha_col].max()
                alp_min = predicted_dataset[alpha_col].min()
            for _, d in predicted_dataset.iterrows():
                if alpha_col:
                    plt.scatter(
                        d["long"],
                        d["lat"],
                        c=colors[k],
                        s=d["demand"],
                        alpha=0.3
                        + ((d[alpha_col] - alp_min) / (alp_max - alp_min) * 0.7),
                    )
                else:
                    plt.scatter(d["long"], d["lat"], c=colors[k], s=d["demand"])

        # plot centroids
        i = 0
        for c in self.centroid:
            plt.scatter(
                c[1],
                c[0],  # alpha=.5,
                s=500,
                c="gold",
                label="cluster ke-" + str(i),
                marker="*",
            )
            i += 1
        plt.xlabel("lat")  # first column of df
        plt.ylabel("long")  # second column of df
        plt.title(title)

        plt.legend()

        return plt.show(block=True)

    def visualize_maps(
        self, title="Plot of K Means Clustering Algorithm", alpha_col=None
    ):
        import plotly.express as px

        px.set_mapbox_access_token(MAPBOX_API_KEY)
        dframe = pd.DataFrame()
        dframe = dframe.assign(
            lat=self.lat_data,
            long=self.long_data,
            pred=self.classification,
            idx=[i for i in range(self.n_population)],
            demand=self.df["demand_target"],
            areaname=self.df["subdistrict_name"],
        )


        if alpha_col:
            dframe = dframe.assign(**{alpha_col: self.df[alpha_col]})

        if alpha_col:
            textdata = dframe[alpha_col].tolist()
            textdata = [str(a) for a in textdata]
            fig = px.scatter_mapbox(
                dframe,
                lat="lat",
                lon="long",
                color="pred",
                size_max=20,
                zoom=12,
                size="demand",
                text=textdata,
                title=title,
                hover_name="areaname",
            )
        else:
            fig = px.scatter_mapbox(
                dframe,
                lat="lat",
                lon="long",
                color="pred",
                size_max=20,
                zoom=12,
                size="demand",
                title=title,
                hover_name="areaname",
            )

        fig.add_trace(go.Scattermapbox(
            mode="markers",
            lat=[a[0] for a in self.centroid],
            lon=[a[1] for a in self.centroid],
            hovertext="centroid",
            text="Centroid",
            opacity=0.85,
            marker={'size':25, 'color' : "#000000", 'symbol' : 'star'},
        ))
        fig.show()

    def calculate_silhouette(self, prediction):
        result = []
        dataset = self.df.assign(prediction=prediction)
        for i in range(self.K):
            cluster_result = []

            # Set dataset between current cluster and neighbor cluster
            cluster_dataset_now = (dataset[dataset.prediction == i])[
                ["lat", "long", "prediction"]
            ]
            c_pointer = i + 1 if i < self.K - 1 else i - 1
            cluster_dataset_next = (dataset[dataset.prediction == c_pointer])[
                ["lat", "long", "prediction"]
            ]

            # Calculate silhoutte every point in cluster
            for _, c in cluster_dataset_now.iterrows():
                a_score = 0
                b_score = 0
                for idx, c_inner in cluster_dataset_now.iterrows():
                    if not (c_inner["lat"] == c["lat"] and c["long"] == c["long"]):
                        a_score += vincenty(
                            (c_inner.lat, c_inner.long), (c.lat, c.long)
                        )

                for idx, c_outer in cluster_dataset_next.iterrows():
                    b_score += vincenty((c_outer.lat, c_outer.long), (c.lat, c.long))

                s_score = (b_score - a_score) / max(a_score, b_score)
                cluster_result.append(s_score)

            result.append(cluster_result)

        return result

    def get_mean_cluster_distane(self, cluster):
        dataset = self.df.assign(prediction=self.predict())
        cluster_dataset_now = (dataset[dataset.prediction == cluster])[
            ["lat", "long", "prediction"]
        ]
        total_distance = 0
        for _, c in cluster_dataset_now.iterrows():
            total_distance += vincenty(self.centroid[cluster], (c.lat, c.long))

        return total_distance / len(cluster_dataset_now)


if __name__ == "__main__":
    flname = "dataset_cluster_warehouse_exp_2.csv"
    df = pd.read_csv(flname)
    data_lat = []
    data_long = []
    for idx, row in df.iterrows():
        point = row["lat_long"].split(",")
        lat = float(point[0])
        long = float(point[1])
        data_lat.append(lat)
        data_long.append(long)

    df = df.assign(lat=data_lat, long=data_long)
    dataset = df[
        [
            "city_name",
            "district_name",
            "avg_demand_baseline",
            "demand_target",
            "lat",
            "long",
            "subdistrict_name",
        ]
    ]
    rent_fee = [random.randint(1000, 2000) for _ in range(len(dataset))]
    dataset = dataset.assign(rent_fee=rent_fee)
    demand_target_data = dataset["demand_target"].tolist()
    km = KMeansAlgorithm(dataset, 6, "lat", "long", demand_target_data)
    km.fit(100)
    print(km.predict())
    # km.visualize_kmeans(figsize=(16, 9))

    scaled_rf = []

    km.visualize_maps(alpha_col="rent_fee")
