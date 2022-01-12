import imageio
from sklearn import datasets
from os import listdir
from matplotlib import pyplot as plt
import numpy as np


def generate_input(num_samples, num_centers, n_features, random_seed, cluster_std=1):
    samples, labels = datasets.make_blobs(n_samples=num_samples, centers=num_centers,
                                          n_features=n_features, random_state=random_seed, cluster_std=cluster_std)

    plt.scatter(samples[:, 0], samples[:, 1], s=5)
    plt.show()
    plt.close()

    return samples, labels, n_features


def plot_result(samples, labels, centers, iteration=None, frame=False):
    for center_idx in range(len(centers)):
        cluster = samples[np.where(labels == center_idx)]
        plt.scatter(cluster[:, 0], cluster[:, 1], s=5)
    plt.scatter([center[0] for center in centers], [center[1] for center in centers], c='red', s=10)
    plt.suptitle("K-Means Clustering")
    title = f'Clusters: {len(centers)}, N Samples: {len(samples)}'
    if iteration:
        title += f', Iteration: {iteration}'
    plt.title(title)

    filename = ''
    if frame:
        filename = f'./experiment/frames/experiment_samples{len(samples)}_clusters{len(centers)}_{iteration}.png'
        plt.savefig(filename)
    else:
        filename = f'./experiment/experiment_samples{len(samples)}_clusters{len(centers)}.png'
        plt.savefig(filename)

    plt.close()

    return filename


def generate_gif(filename_list):
    """
    Creates a gif animation from a list of images
    :param filename_list: list of image filenames
    :return: saves a gif movie in root path
    """

    images = []
    for filename in filename_list:
        images.append(imageio.imread(filename))
    imageio.mimsave("./experiment/movie.gif", images)
