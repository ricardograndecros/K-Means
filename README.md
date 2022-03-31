# K-Means
Small project for playing around with k-means. Sample data is generated using gaussian blobs.  
All datasets generated are 2D for a more intuitive and comprehensible plot.

To run an experiment execute 
```
python3 k_means.py <number of samples> <num_features> <clusters in data> <clusters to classify> <epsilon>
```

The parameter epsilon is the threshold at which the network will stop training.  
For example, if epsilon = 0.01 the algorithm will stop running when the centroids'  
position changes in a value lower than epsilon. 

Executing an experiment will generate a series of plots that will be stored at [experiment](https://github.com/ricardograndecros/K-Means/tree/master/experiment)  
An animation of the learning process will be generated and stored as [movie.gif](https://github.com/ricardograndecros/K-Means/blob/master/experiment/movie.gif)  

![Example](https://github.com/ricardograndecros/K-Means/blob/master/experiment/movie.gif)
