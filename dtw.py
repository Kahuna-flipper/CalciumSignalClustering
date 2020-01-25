import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

S1 = [1,4,5,7,3,8,6]
S2 = [4,2,9,3,5,7,8]


def Distance(S1,S2):
    m = np.size(S1)
    n = np.size(S2)
    
    distances = np.zeros((m,n))
    
    for i in range(0,m):
        for j in range(0,n):
            distances[i,j]=(S1[i]-S2[j])**2
        
def distance_cost_plot(distances):
    im = plt.imshow(distances, interpolation='nearest', cmap='Reds') 
    plt.gca().invert_yaxis()
    plt.xlabel("S1")
    plt.ylabel("S2")
    plt.grid()
    plt.colorbar();
    
distance_cost_plot(distances)
plt.show()

cost = np.zeros((m,n))

for i in range(0,n):
    cost[0,i] = distances[0,i] + cost[0,i-1]
    
for i in range(0,n):
    cost[i,0] = distances[i,0] + cost[i,0]
    

for i in range(1, m):
    for j in range(1, n):
        cost[i, j] = min(cost[i-1, j-1], cost[i-1, j], cost[i, j-1]) + distances[i, j]
        

distance_cost_plot(cost)

plt.show()

def path_cost(x, y,cost, distances):
    path = [[m-1, n-1]]
    i = m-1
    j = n-1
    while i>0 and j>0:
        if i==0:
            j = j - 1
        elif j==0:
            i = i - 1
        else:
            if cost[i-1, j] == min(cost[i-1, j-1], cost[i-1, j], cost[i, j-1]):
                i = i - 1
            elif cost[i, j-1] == min(cost[i-1, j-1], cost[i-1, j], cost[i, j-1]):
                j = j-1
            else:
                i = i - 1
                j= j- 1
        path.append([j, i])
    path.append([0,0])
    for [y, x] in path:
        cost = cost +distances[x, y]
    return path, cost
    
    
    path_x = [point[0] for point in path]
    path_y = [point[1] for point in path]
    distance_cost_plot(cost)
    plt.plot(path_x, path_y)
    plt.show()


plt.plot(S1, 'bo-' ,label='S1')
plt.plot(S2, 'g^-', label = 'S2')
plt.legend();
paths = path_cost(S1, S2, cost, distances)[0]
for [map_x, map_y] in paths:
    print(map_x, S1[map_x], ":", map_y, S2[map_y])
    
    plt.plot([map_x, map_y], [S1[map_x], S2[map_y]], 'r')
    plt.legend(loc="upper left")
plt.show()

plt.plot(S1,label='S1')
plt.plot(S2,label='S2')
plt.legend(loc="upper left")
plt.show()