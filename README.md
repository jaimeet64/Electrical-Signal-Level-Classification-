# Electrical-Signal-Level-Classification-
------------------------------------------------
Title: Signal data clustering using DBSCAN method.
Author: Jaimeet Vinayakbhau Patel
Date: 08/12/16
Description of DBSCAN:
    
DBSCAN is an acronym for Density-based spatial clustering of applications with noise.
It was developed by Martin Ester, Hans-Peter Kriegel, Jörg Sander and Xiaowei Xu in 1996.
Consider a set of points in some space to be clustered. For the purpose of DBSCAN clustering,
the points are classified as core points, (density-)reachable points and outliers, as follows:
1)A point p is a core point if at least minPts points are within distance ε of it (including p).
  Those points are said to be directly reachable from p.
  By definition, no points are directly reachable from a non-core point.
2)A point q is reachable from p if there is a path p1, ..., pn with p1 = p and pn = q,
  where each pi+1 is directly reachable from pi (all the points on the path must be core points,
  with the possible exception of q).
3)All points not reachable from any other point are outliers.

-->A cluster then satisfies two properties:

1)All points within the cluster are mutually density-connected.
2)If a point is density-reachable from any point of the cluster, it is part of the cluster as well.

Description of the code.

Below mentioned code used the dbscan algorithm to determine optimized value of Maximum radius and
Minimum radius in order to classify two state. This code averages the stable state i.e High or low
and keep the spike content in the signal.
