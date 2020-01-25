import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import glob 
from scipy.signal import find_peaks, peak_prominences 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering 
from pylab import * 
from obspy.signal.detrend import polynomial 
from dtaidistance import dtw

filename = glob.glob("*.csv")
N_files = len(filename) 
cell_data=np.zeros((300,300))
k=0
p=0
rowsum=0
columnsum=0

for f in range(0,N_files):
    data=pd.read_csv(filename[f])
    converted_data=data.as_matrix()
    (m,n)=np.shape(converted_data)
    columnsum=columnsum+n
    for something in range(0,n):
         temp =converted_data[0:200,something]
         cell_data[0:200,k] = temp
         k=k+1
    

final_cell_data=np.zeros((200,columnsum))

final_cell_data[0:200,0:columnsum]=cell_data[0:200,0:columnsum]

for i in range(0,columnsum):  #Detrending the signals
    polynomial(final_cell_data[:,i],order=4,plot=False)
    
    
[m,n] = np.shape(final_cell_data)

distance = np.zeros((n,n))

#for i in range(0,n):
    #final_cell_data[:,i] = final_cell_data[:,i]/np.max(final_cell_data[:,i])
    #final_cell_data[:,i] = (final_cell_data[:,i]-np.min(final_cell_data[:,i]))/(np.max(final_cell_data[:,i])-np.min(final_cell_data[:,i]))

for i in range(0,n):
    for j in range(0,n):
        distance[i,j] = dtw.distance(final_cell_data[:,i],final_cell_data[:,j])


nclusters = [3,4,5,6,7,8,9]
for k in nclusters:        
    cluster = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='average')  
    cluster.fit_predict(distance)
    c2 = cluster.labels_
    
    tempj=0
    sum1=np.zeros((m,1))
    error=np.zeros((m,1))
    for temp in range(0,k):
        temp_cluster=np.where(c2==k)
        temp_size=np.size(temp_cluster)
        for tempj in range(0,temp_size):
            temp_cell = temp_cluster[0][tempj]
            sum1=sum1+final_cell_data[:,temp_cell]
        centroid=sum1/temp_size
        for tempj in range(0,temp_size):
            temp_cell = temp_cluster[0][tempj]
            error=error + (final_cell_data[:,temp_cell]-centroid)*2
        avg_error=np.sum(error)
            
            
        


for temp in range(0,6):
    temp_cluster=np.where(c2==temp)
    temp_size=np.size(temp_cluster)
    print('Plotting New Cluster')
    print(temp+1)
    
    tempj=0;            
    for i in range(1,2*temp_size,2):
        plt.subplot(temp_size,2,i)
        #plt.title('Cell')
        temp_cell=temp_cluster[0][tempj]
        plt.plot(final_cell_data[:,temp_cell])
        plt.axis('on')
        plt.xticks([])
        plt.yticks([])
        tempj=tempj+1
    plt.show()





