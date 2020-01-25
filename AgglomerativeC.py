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
             

#  Binding all the features and Clustering : 
features=np.concatenate((amplitude.T,npeaks.T,interspike.T),axis=1)

norm_npeaks=npeaks/np.max(npeaks)
norm_amplitude=amplitude/np.max(amplitude)
norm_interspike=interspike/np.max(interspike)

norm_features=np.concatenate((norm_amplitude.T,norm_npeaks.T,norm_interspike.T),axis=1)



cluster = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')  
cluster.fit_predict(norm_features)

fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')

ax.scatter(norm_features[:,0],norm_features[:,1],norm_features[:,2],s=30,c=cluster.labels_,cmap='rainbow',)  
plt.show()
#plt.scatter(norm_features[:,0],norm_features[:,1],s=30,c=cluster.labels_,cmap='rainbow',)
#plt.show()
c=cluster.labels_

plt.show()


# Mapping back clustered data to original cell data 

for temp in range(0,6):
    temp_cluster=np.where(c==temp)
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
        plt.axis('on')
        plt.xticks([])
        plt.yticks([])
        tempj=tempj+1;
    plt.show()




             



       
    
    
    

    
    
    

