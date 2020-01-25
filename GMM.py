import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import glob 
from scipy.signal import find_peaks, peak_prominences 
from mpl_toolkits.mplot3d import Axes3D

from pylab import * 
from obspy.signal.detrend import polynomial 
from sklearn.mixture import GaussianMixture 

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
    polynomial(final_cell_data[:,i],order=3,plot=False)
        

# Feature Extraction 

# Amplitude 

amplitude=np.zeros((1,columnsum))

for i in range(0,columnsum):
    amplitude[:,i]=max(final_cell_data[:,i])
    
# Number of Peaks and Inter-Spike Distance  : 
    
npeaks=np.zeros((1,columnsum))
interspike=np.zeros((1,columnsum))

for i in range(0,columnsum):
    temp_peaks,_=find_peaks(final_cell_data[:,i])
    prominences = peak_prominences(final_cell_data[:,i], temp_peaks)[0]
    temp=len(temp_peaks)
    for k in range(0,temp):
        peaks, _ = find_peaks(final_cell_data[:,i],prominence=(prominences[k]/5,None))
#    plt.plot(final_cell_data[:,i])
#    plt.plot(peaks,final_cell_data[peaks,i],"x")
#    plt.show()
    npeaks[0,i]=len(peaks)
    for p in range(0,len(peaks)):
             k=final_cell_data[:,i]
             interlist=[]
             temp_inter_spike_interval=np.sqrt((k[p+1]-k[p])**2)
             interlist.append(temp_inter_spike_interval)
             interspike[0,i]=np.median(interlist)
             

# Gaussian Mixture Part : 
gmm = GaussianMixture(n_components = 5) # Mixture of 5 Gaussians 

#Binding all the features and Clustering : 
features=np.concatenate((amplitude.T,npeaks.T,interspike.T),axis=1)

norm_npeaks=npeaks/np.max(npeaks)
norm_amplitude=amplitude/np.max(amplitude)
norm_interspike=interspike/np.max(interspike)

norm_features=np.concatenate((norm_amplitude.T,norm_npeaks.T,norm_interspike.T),axis=1)

gmm.fit(norm_features)

labels = gmm.predict(norm_features) 
#norm_features['labels']= labels 
d0 = np.asarray(np.where(labels==0),dtype='float')
d1 = np.asarray(np.where(labels==1),dtype='float') 
d2 = np.asarray(np.where(labels==2),dtype='float')
d3 = np.asarray(np.where(labels==3),dtype='float')
d4 = np.asarray(np.where(labels==4),dtype='float')

#fig = plt.figure(2)
#ax = fig.add_subplot(111, projection='3d')

#ax.scatter(norm_features[:,0],norm_features[:,1],norm_features[:,2],s=30,c=labels,cmap='rainbow',)  
#plt.show()

plt.scatter(norm_features[:,0],norm_features[:,1],s=30,c=labels,cmap='rainbow',)
plt.show()

d0size = np.size(d0)
Cluster0=np.zeros((d0size,3))
k=0;
for i in range(0,d0size):
    temp_p=np.asarray(d0[0][i],dtype='int')
    Cluster0[k,0:3]=norm_features[temp_p,0:3]
    k=k+1
    
d1size = np.size(d1)
Cluster1=np.zeros((d1size,3))
k=0;
for i in range(0,d1size):
    temp_p=np.asarray(d1[0][i],dtype='int')
    Cluster1[k,0:3]=norm_features[temp_p,0:3]
    k=k+1
    
d2size = np.size(d2)
Cluster2=np.zeros((d2size,3))
k=0;
for i in range(0,d2size):
    temp_p=np.asarray(d2[0][i],dtype='int')
    Cluster2[k,0:3]=norm_features[temp_p,0:3]
    k=k+1
    
d3size = np.size(d3)
Cluster3=np.zeros((d3size,3))
k=0;
for i in range(0,d3size):
    temp_p=np.asarray(d3[0][i],dtype='int')
    Cluster3[k,0:3]=norm_features[temp_p,0:3]
    k=k+1
    
d4size = np.size(d4)
Cluster4=np.zeros((d4size,3))
k=0;
for i in range(0,d4size):
    temp_p=np.asarray(d4[0][i],dtype='int')
    Cluster4[k,0:3]=norm_features[temp_p,0:3]
    k=k+1
    
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Cluster0[:,0], Cluster0[:,1], Cluster0[:,2],s=40, c ='red') 
ax.scatter(Cluster1[:,0], Cluster1[:,1], Cluster1[:,2],s=40, c ='yellow') 
ax.scatter(Cluster2[:,0], Cluster2[:,1], Cluster2[:,2],s=40, c ='green')
ax.scatter(Cluster3[:,0], Cluster3[:,1], Cluster3[:,2],s=40, c ='blue')
ax.scatter(Cluster4[:,0], Cluster4[:,1], Cluster4[:,2],s=40, c ='black')


# Mapping Back 

# Cluster 1

for temp in range(0,5):
    temp_cluster=np.where(labels==temp)
    temp_size=np.size(temp_cluster)
    print('Plotting New Cluster')
    print(temp+1)
    
    tempj=0;            
    for i in range(1,2*temp_size,2):
        plt.subplot(temp_size,2,i)
        #plt.title('Cell')
        temp_cell=temp_cluster[0][tempj]
        plt.plot(final_cell_data[:,temp_cell])
        plt.xticks([])
        plt.yticks([])
        
        #for temp2 in range(np.size(temp_cluster)):
        #q=temp_cluster[0][temp2]
        temp_peaks, _ = find_peaks(final_cell_data[:,temp_cell])
        prominences = peak_prominences(final_cell_data[:,temp_cell], temp_peaks)[0]
        temp1=len(temp_peaks)
        for k in range(0,temp1):
            peaks2, _ = find_peaks(final_cell_data[:,temp_cell],prominence=(prominences[k]/5,None))
        
        plt.subplot(temp_size,2,i+1)
        #plt.title('Raster Plot')
        plt.eventplot(peaks2)
        plt.xticks([])
        plt.yticks([])
        tempj=tempj+1;
        
    
    plt.show()
  


#
#
#cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')  
#cluster.fit_predict(norm_features)
#
#fig = plt.figure(2)
#ax = fig.add_subplot(111, projection='3d')
#
#ax.scatter(features[:,0],features[:,1],features[:,2],s=30,c=cluster.labels_,cmap='rainbow',)  
#c=cluster.labels_
#plt.show()
#
#
## Mapping back clustered data to original cell data 
#
#for temp in range(0,5):
#    temp_cluster=np.where(c==temp)
#    temp_size=np.size(temp_cluster)
#    print('Plotting New Cluster')
#    print(temp+1)
#    
#    tempj=0;            
#    for i in range(1,2*temp_size,2):
#        plt.subplot(temp_size,2,i)
#        plt.title('Cell')
#        temp_cell=temp_cluster[0][tempj]
#        plt.plot(final_cell_data[:,temp_cell])
#        
#        #for temp2 in range(np.size(temp_cluster)):
#        #q=temp_cluster[0][temp2]
#        temp_peaks, _ = find_peaks(final_cell_data[:,temp_cell])
#        prominences = peak_prominences(final_cell_data[:,temp_cell], temp_peaks)[0]
#        temp1=len(temp_peaks)
#        for k in range(0,temp1):
#            peaks2, _ = find_peaks(final_cell_data[:,temp_cell],prominence=(prominences[k]/5,None))
#        
#        plt.subplot(temp_size,2,i+1)
#        plt.title('Raster Plot')
#        plt.eventplot(peaks2)
#        tempj=tempj+1;
#    plt.show()
#



             



       
    
    
    

    
    
    

