import sys

from tools import *
import numpy as np


class KMeans:
    def __init__(self, samples, labels, features, num_clusters, random_seed=2):
        self.samples = samples
        self.labels = labels
        self.features = features
        self.num_clusters = num_clusters
        # array storing k centers
        self.random_seed = random_seed
        # randomly generate k clusters' centers
        self.centers = np.random.RandomState(self.random_seed).randint(low=min(samples[:, 0]), high=max(samples[:, 1]),
                                                                       size=(num_clusters, features))
        # initialize label array for learning algorithm
        self.learning_labels = np.zeros(len(labels))

        self.images = []

    def clustering_algorithm(self, stopping_value):
        stop = False
        n = 0
        while not stop:
            distances = self.distance_to_clusters(self.samples)
            self.learning_labels = np.argmin(distances, axis=1)
            # calculate new centers
            new_centers = self.new_centers()
            # evaluate stop condition
            stop = self.evaluate_end(new_centers, stopping_value)
            self.centers = new_centers
            n += 1
            self.images.append(plot_result(self.samples, self.learning_labels, self.centers, n, True))

    def evaluate_end(self, new_centers, stopping_value):
        for center_idx in range(len(self.centers)):
            if np.sqrt(np.sum((new_centers[center_idx] - self.centers[center_idx]) ** 2)) > stopping_value:
                return False
        return True

    def new_centers(self):
        new_centers = []
        for center_idx in range(len(self.centers)):
            points_won = np.where(self.learning_labels == center_idx)
            if len(points_won[0]):
                cluster_points = self.samples[points_won]
                new_centers.append(np.mean(cluster_points, axis=0))
            else:
                # if no entry clustered, maintain old center
                new_centers.append(self.centers[center_idx])
        return new_centers

    def distance_to_clusters(self, array):
        distances = []
        for sample in array:
            distances.append(np.sqrt(np.sum((sample - self.centers) ** 2, axis=1)))
        return distances


def main():
    # argv = [number of samples, num_features,clusters in data, clusters to classify, epsilon]
    args = sys.argv
    n_samples = int(args[1])
    num_features = int(args[2])
    num_centers = int(args[3])
    clusters_classify = int(args[4])
    stopping_value = float(args[5])
    random_seed = int(args[6])

    dataset, labels, n_features = generate_input(n_samples, num_centers, num_features, random_seed)
    cluster = KMeans(dataset, labels, n_features, clusters_classify)
    cluster.clustering_algorithm(stopping_value)
    plot_result(cluster.samples, cluster.learning_labels, cluster.centers)
    generate_gif(cluster.images)


if __name__ == '__main__':
    main()
