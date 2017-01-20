from math import sqrt
import numpy as np
import pickle
import pandas
import matplotlib.pyplot as plt
from pylab import *

feature = pickle.load(open("C:\\Users\\Jaimeet Patel\\Downloads\\features_test.pkl", 'rb'))
p = feature.p.values
t=int(input('Enter the index of signal data:'))
num_samples=len(p[t])
a=np.array(p[t])
b=a.astype(np.int64)  ##convert from float to integer.
c=[x - b[i - 1] for i, x in enumerate(b)][1:]
d=map(abs, c)
print(d)
def stat(lst):
    """Calculate mean and std deviation from the input list."""
    n = float(len(lst))
    mean = sum(lst) / n
    diff_lst = [i-mean for i in lst]
    sq_diff_lst = [ i*i for i in diff_lst]
    sum_of_all_square_of_diff = sum(sq_diff_lst)
    stdev = sqrt(sum_of_all_square_of_diff/len(lst))

    return mean, stdev

def parse(lst, n):
    cluster = []
    output = []
    for i in lst:
        if len(cluster) <= 1:    # the first two values are going directly in
            cluster.append(i)
            continue

        mean,stdev = stat(cluster)
        if abs(mean - i) > n * stdev:    # check the "distance"
            
            output.append(cluster.copy())
            
            cluster[:] = []    # reset cluster to the empty list

        cluster.append(i)
    output.append(cluster.copy())
    return output           # yield the last cluster
v=a.copy()
m=a.copy()
length=0

for n in range(9,4,-1):   # 
    j_list=parse(m,n)
	
    #print(n)
    print(len(j_list))
    if (len(j_list)>1):
        j_mean_list=[mean(i) for i in j_list]    # Finding mean of the classified list.
        print(j_mean_list)
        v[length+4:length+len(j_list[0])-4]=j_mean_list[0]
        ##print(v)
        length=length+len(j_list[0])   # Finding length of the list
        
        
        transition=c=[x -j_mean_list[i - 1] for i, x in enumerate(j_mean_list)][1:]
        print(transition)

        m=[]    # reset a to the empty list
        m=a[length:len(a)-1]
        #print(m)
    
v[length+4:length+len(j_list[0])-4]=j_mean_list[1]
       
#fig = plt.figure()

plt.figure(1)
plt.subplot(121)
plt.plot(b)

plt.subplot(122)
plt.plot(v)
plt.show()
