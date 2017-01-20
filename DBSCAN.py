
##Importing important libraries

from math import sqrt
import numpy as np
import pickle
import pandas
import matplotlib.pyplot as plt
from pylab import *
from sklearn.cluster import DBSCAN

## Load Signal data from Pickle file
feature = pickle.load(open("C:\\Users\\Jaimeet Patel\\Downloads\\features_test.pkl", 'rb'))
p = feature.p.values
t=int(input('Enter the index of signal data:'))
a=np.array(p[t])
num_samples=len(a)    ## Length of the signal
b=a.astype(np.int32)  ##convert from float to integer.
c=np.zeros(len(a))
d=c.astype(np.int32)
X=list(zip(b,d))

##Desnisty based Clustering technique to classify 2 levels, 'High and Low'
##and extracting skipe of an eletrical signal from the data.
n_clusters=0
## Initialzation of 2 important parameter
##1)min_samples = Minimum number of data samples in a cluster.
##2)eps= Minimum Radius.
min_samples=20
eps=8
h=int(round(0.03*num_samples))
print('h',h)

## Obtaining Optimum value for both parameters:
while(n_clusters<=2) :
    db = DBSCAN(eps, min_samples).fit(X)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_.tolist()

    outliner_count=labels.count(-1)
    
    z=np.convolve([-1,-1,-1],labels).tolist()
    
    spike_count=z.count(3)+2
    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    if(eps>1):    
        eps=eps-0.1
    elif(min_samples>2):
        min_samples=min_samples-1
    
    print('min_samp',min_samples)    
    print('n_clusters',n_clusters)
    print('eps',eps)
    print('-----------------------------------------')
    
## Minimum length of the spike is 3 
if(spike_count>2 and spike_count<7):
    print('Spike is present in the signal of length',spike_count)
elif(spike_count>7 and n_clusters==1):
    print('Signal is confused to be spike')
else:
    print('Spike is not present in the signal')
    
if(min_samples<20):
    min_samples=min_samples+2
    eps=eps+0.1
    
db = DBSCAN(eps, min_samples).fit(X)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
high=np.zeros(len(b))
low=np.zeros(len(b))
spike=np.zeros(len(b))

for i in range(0,len(a)):
    if labels[i]==1:
        high[i]=b[i]
    elif labels[i]==0:
        low[i]=b[i]  
    if labels[i]<0 and labels[i-1]<0 and labels[i-2]<0:
        spike[i-2]=b[i-2]
        spike[i-1]=b[i-1]
        spike[i]=b[i]
    
    elif (labels[i-1]==0 and labels[i]==-1 and labels[i+1]==0):
        low[i]=b[i]
    elif (labels[i-1]==0 and labels[i]==-1 and labels[i+1]==-1 and labels[i+2]==0):
        low[i]=b[i]
    elif (labels[i-2]==0 and labels[i-1]==-1 and labels[i]==-1 and labels[i+1]==0):
        low[i]=b[i]
    elif (labels[i-1]==1 and labels[i]==-1 and labels[i+1]==1):
        high[i]=b[i]
    elif (labels[i-1]==1 and labels[i]==-1 and labels[i+1]==-1 and labels[i+2]==1):
        high[i]=b[i]
    elif (labels[i-2]==1 and labels[i-1]==-1 and labels[i]==-1 and labels[i+1]==1):
        high[i]=b[i]

## Obtaining mean value of classified list
        
H_list=high.tolist()
len_H=len(high)-H_list.count(0)

L_list=low.tolist()
len_L=len(low)-L_list.count(0)

spike_list=spike.tolist()
len_spike=len(spike)-spike_list.count(0)

H_mean=sum(high)/len_H
L_mean=sum(low)/len_L
spike_mean=sum(spike)/len_spike

##Replacing original value with the mean
for i in range(0,len(a)):
    if high[i]>0:
        high[i]=H_mean
    if low[i]>0:
        low[i]=L_mean
    if (spike[i]>0 and spike_count>5 and n_clusters==1):
        spike[i]=spike_mean
        
##Construncting Signal
Final=high+low+spike

print('--------------------------------------------------------------------------')
print('Labels with optimised value Minimum numper of points and maximum radius ',labels )
print('--------------------------------------------------------------------------')

# ##Plot of original data and estimate data.
# plt.plot(Final,'r') #you can change index to see different signal
# plt.plot(a,'b') #you can change index to see different signal
# plt.show()


#fig = plt.figure()

plt.figure(1)
plt.subplot(121)
plt.plot(a)

plt.subplot(122)
plt.plot(Final)
plt.show()

## Gives the range of zeros
def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    print(ranges[:][1])
    return ranges
