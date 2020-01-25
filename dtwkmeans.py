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
from tslearn.clustering import TimeSeriesKMeans
from tslearn.generators import random_walks
import dtw1
from tslearn.metrics import dtw



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
         temp = converted_data[0:200,something]
         cell_data[0:200,k] = temp
         k=k+1
    

final_cell_data=np.zeros((200,columnsum))

final_cell_data[0:200,0:columnsum]=cell_data[0:200,0:columnsum]

for i in range(0,columnsum):  #Detrending the signals
    polynomial(final_cell_data[:,i],order=4,plot=False)
        

[rows,columns] = np.shape(final_cell_data)

#dtw1.metric(final_cell_data[0],final_cell_data[7])

#
for i in range(0,columns):
    final_cell_data[:,i] = final_cell_data[:,i]/np.max(final_cell_data[:,i])
  

num = np.linspace(2,10,9)  
add = np.zeros((rows,1))
temp_error=0
error=[]
for i in num:
    

    km = TimeSeriesKMeans(n_clusters=int(i),metric="dtw",max_iter=5,random_state=0).fit(np.transpose(final_cell_data))
        
    labels = km.labels_
    labels = np.transpose(labels)
        
    for temp in range(0,int(i)):
        temp_cluster=np.where(labels==temp)
        temp_size=np.size(temp_cluster)
            
            
        print('Plotting New Cluster')
        print(temp+1)
        tempj=0
    
            
        for t in range(0,temp_size):
            temp_cell=temp_cluster[0][tempj]
            add = np.add(add,final_cell_data[:,temp_cell])/temp_size
                
            tempj=tempj+1
            
        t_add = np.transpose(add[0,:])
    
            
        tempj=0
        for t in range(0,temp_size):
            temp_cell=temp_cluster[0][tempj]
            temp_error = temp_error + dtw.distance(final_cell_data[:,temp_cell],km.cluster_centers_[:,int(i)])
            tempj=tempj+1
                
            
    error.append(temp_error)
            
            
plt.plot(num,error,'o-')
plt.show() 
            
#for i in range(1,2*temp_size,2):
#    plt.subplot(temp_size,2,i)
#            #plt.title('Cell')
#    temp_cell=temp_cluster[0][tempj]
#    add = add + final_cell_data[:,temp_cell]
#    avg = add/temp_size
#    plt.plot(final_cell_data[:,temp_cell])
#    plt.axis('on')
#    plt.xticks([])
#    plt.yticks([])
#    tempj=tempj+1
    
#plt.show()


    
    
    

    
    




