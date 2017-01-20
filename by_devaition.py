from math import sqrt
import numpy as np
import pickle
import pandas
import matplotlib.pyplot as plt
from pylab import *

feature = pickle.load(open("C:\\Users\\Jaimeet Patel\\Downloads\\features_test.pkl", 'rb'))
p = feature.p.values
num_samples=len(p[7])
b=np.array(p[101])
a=b.astype(np.int64)  ##convert from float to integer.

"""Calculate mean and std deviation from the input list."""
n = float(len(a))
mean = sum(a) / n
    
stdev = sqrt(sum((a-mean)*(a-mean))/n)

a1=[]    
cluster = []
for i in a:
    if len(cluster) <= 1:    # the first two values are going directly in
        cluster.append(i)
        continue

    if abs(mean - i) > 7 * stdev:    # check the "distance"
        print(cluster)
        a1.append([cluster])   # will return a huge set of values
        cluster[:] = []    # reset cluster to the empty list
    cluster.append(i)
    
v=a1.append(cluster)           # yield the last cluster)
print(cluster)    

plt.plot(v) #you can change index to see different signal
plt.show()
    



