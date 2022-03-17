---
layout: post
title: Spectral Clustering!
---

In this blog post we will be exploring spectral clustering. Spectral clustering is a way of clustering points that don't show up in normal blob clusters. Unlike an algorithm like k-means which tries to find a center point for each cluster, spectral clustering looks at how close each point in the cluster is to all the points around it and sees if those points belong to the same cluster. In this way the fuction essentially trys to minimise the number of points that are close to each other and do not have the same classification. Lets get started with our normal imports an generate two blob classes on a graph using sklearns make_blobs utility. 


```python
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
```


```python
n = 200
np.random.seed(1111)
X, y = datasets.make_blobs(n_samples=n, shuffle=True, random_state=None, centers = 2, cluster_std = 2.0)
plt.scatter(X[:,0], X[:,1])
```




    <matplotlib.collections.PathCollection at 0x7fc02a164a00>




![output_3_1.png]({{ site.baseurl }}/images/output_3_1.png)    

    


The first method we will try using is K-means which attemps to find the center of each 'blob' and classify the points based on distance to that center point. 


```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters = 2)
km.fit(X)

plt.scatter(X[:,0], X[:,1], c = km.predict(X))
```




    <matplotlib.collections.PathCollection at 0x7fc02ebcd280>




    
![output_5_1.png]({{ site.baseurl }}/images/output_5_1.png)    
    


As we can see, K-means did a good job seperating our two blobs. However what happens if we try this same experiment using half moon clusters? 


```python
np.random.seed(1234)
n = 200
X, y = datasets.make_moons(n_samples=n, shuffle=True, noise=0.05, random_state=None)
plt.scatter(X[:,0], X[:,1])
```




    <matplotlib.collections.PathCollection at 0x7fc02eceffa0>




    
![output_7_1.png]({{ site.baseurl }}/images/output_7_1.png)    
    


In this sample there are still two clusters, but they dont have a clear 'center' which means that K-means might have a more difficult time. 


```python
km = KMeans(n_clusters = 2)
km.fit(X)
plt.scatter(X[:,0], X[:,1], c = km.predict(X))
```




    <matplotlib.collections.PathCollection at 0x7fc02edc2f40>




    
![output_9_1.png]({{ site.baseurl }}/images/output_9_1.png)
    


As we can see, K-means did not do a good job seperating our half moon shaped groups. In this case we need a new approach. We will use spectral clustering for this task.
## Similarity Matrix
In order to build our spectral clustering algorithm we need to first construct a similarity matrix for our dataset. The similarity matrix is a matrix the represents which points in the dataset are close together. To do this we write a function called make_sim_matrix that takes in our dataset and an epsilon value which represents how close points need to be to be considered 'close' by the function and returns a matrix A, of 1s and 0s. The point A[i][z] represents if points X[i] and X[z] are 'close' to each other. A 0 means they are not close and a 1 means that they are. 


```python
from sklearn.metrics.pairwise import euclidean_distances
def make_sim_matrix(X,epsilon):

    #set the epsilon
    epsilon = epsilon

    #use the euclidean_distances function to compute a matrix of the pairwise distances
    distance_matrix=euclidean_distances(X,X)

    #convert the values of that matrix from distance to 1s and 0s by compairing the distance to epsilon
    A = np.where(distance_matrix<epsilon, 1, 0)

    #fill the matrix diagonal with 0s 
    np.fill_diagonal(A,0)
    return A
A= make_sim_matrix(X,0.4)
A
```




    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 1, 0],
           ...,
           [0, 0, 0, ..., 0, 1, 1],
           [0, 0, 1, ..., 1, 0, 1],
           [0, 0, 0, ..., 1, 1, 0]])



## Binary Norm Cut Objective
Now that we have the similarity matrix the next step is to compute the binary norm cut objective. If we assign each point to a cluster C0 or C1, the binary norm cut objective is a measure of how 'good' this particular clustering of the data is. This is based on two things, how many entries in our similatiry matrix relate points in C0 to points in C1 and how 'big' each cluster is relative to the dataset as a whole. We call the former the 'cut term' and the later the 'volume term'. A low binary norm cut objective is good and means that we have a good partition of the data. Looking at both terms, this means that to get a good binary norm cut objective we need:
1. The clusterings to not have a lot of overlap.
2. Neither cluster to be very small compaired to the other.

First we compute the cut term by seeing how many points in C0 are close to points in C1.


```python
def cut(A,y):
    total_incorrect=0
    n=y.size
    #itterate through each entry of A
    for i in range(n):
        for j in range(n):
            #if that entry has a 1 meaning the points are close
            if(A[i][j]==1):
                
                #if i and j do are not part of the same group, add 1 to total incorrect
                if(y[i]!=y[j]):
                    total_incorrect+=1
    #return the total
    return total_incorrect/2
                
```

Now we check our function by running it with the true labels and with random labels. As we can see the real labels generate much lower scores, a good sign that our function works as intended.


```python
fake_labels=np.random.choice([1,0], 200)

real_cut = cut(A,y)
fake_cut= cut(A, fake_labels)
print("real: " + str(real_cut))
print("fake: " + str(fake_cut))
```

    real: 13.0
    fake: 1150.0


Now we much compute the volume term. To do this we first calculate the 'degree' of each point or the number of points each point is 'close' to. Then we add up the degrees of points in C0 to get the volume of C0 and the same for C1. Finally we put the volume and cut terms together using the formula:
$$N_{\mathbf{A}}(C_0, C_1)\equiv \mathbf{cut}(C_0, C_1)\left(\frac{1}{\mathbf{vol}(C_0)} + \frac{1}{\mathbf{vol}(C_1)}\right)\;.$$
to get the binary normcut objective.


```python
def compute_degree(A):
    #create a base array with the same legth as the number of points
    base_array=np.zeros(A.shape[0])
    #sum our similarity matrix to get an array containing the number of points each point is 'close' to
    base_array=np.sum(A, axis=1)
    return base_array
    

def vols(A,y):
    #compute the degree
    degree=compute_degree(A)
    #compute the volumes by summing the degree for each cluster
    vol0=degree[y==0].sum()
    vol1=degree[y==1].sum()
    return vol0, vol1



def normcut(A, y):
    #compute vol and cut terms using the defined functions
    v0, v1= vols(A,y)
    cut_val=cut(A,y)
    #return the normcut based on the given function
    return cut_val*((1/v0)+(1/v1))
```


```python

```

We can now compare normcut scores for the real and fake labels on our dataset.


```python
print(normcut(A,y))
print(normcut(A, fake_labels))
```

    0.011518412331615225
    1.0240023597759158


As you can see the normcut objective is much much higher when using the fake labels. This makes sense since our clusers would largly be wrong if we simply assigned values randomly.

Now we could try to brute force guess an array of labels that minimizes the normcut objective but this is very computationally expensive. Rather, lets define a vector Z where:
$$
z_i = 
\begin{cases}
    \frac{1}{\mathbf{vol}(C_0)} &\quad \text{if } y_i = 0 \\ 
    -\frac{1}{\mathbf{vol}(C_1)} &\quad \text{if } y_i = 1 \\ 
\end{cases}
$$

By doing this we end up with the values of y implicit in z and can use z to define our normcut objective using the formula:

$$\mathbf{N}_{\mathbf{A}}(C_0, C_1) = \frac{\mathbf{z}^T (\mathbf{D} - \mathbf{A})\mathbf{z}}{\mathbf{z}^T\mathbf{D}\mathbf{z}}\;,$$


D in this case is a matrix of zeros where the diagonal is the degree of each point in X.
Lets define a function to compute z.



```python
def transform(A,y):
    #get vols
    v0,v1 = vols(A,y)
    #define the cases and use np. where to construct z
    case0 = (1/v0) 
    case1=(-1/v1)
    Z = np.where(y==0, case0, case1)
    return Z

```


```python
#construct z
z= transform(A,y)
#comput the degree of A make a 200 by 200 matrix D
D_val= compute_degree(A)
D=np.zeros((200,200))
#use D_val to fill the diagonal on D
np.fill_diagonal(D,D_val)
D

```




    array([[15.,  0.,  0., ...,  0.,  0.,  0.],
           [ 0., 25.,  0., ...,  0.,  0.,  0.],
           [ 0.,  0., 24., ...,  0.,  0.,  0.],
           ...,
           [ 0.,  0.,  0., ..., 26.,  0.,  0.],
           [ 0.,  0.,  0., ...,  0., 28.,  0.],
           [ 0.,  0.,  0., ...,  0.,  0., 26.]])




```python
#define a function to compute the matrix product
def matrix_product(z,D,A):
    prod=((z@(D-A)@z)/(z@D@z))
    return prod

#check that matrix_product and normcut return the same thing
print(np.isclose(matrix_product(z,D,A),normcut(A,y)))

#check that z contains roughly as many positive as negative entries
print(np.isclose(np.ones(n)@D@z,0))
```

    True
    True


Now we know that we can minimize the function $$ R_\mathbf{A}(\mathbf{z})\equiv \frac{\mathbf{z}^T (\mathbf{D} - \mathbf{A})\mathbf{z}}{\mathbf{z}^T\mathbf{D}\mathbf{z}} $$ in order to end up with an optimal split of our classes. However we need to do this with respect to the condition $\mathbf{z}^T\mathbf{D}\mathbb{1} = 0$. The function orth below bakes this condition into the function which we can then minimize using the minimize function from scipy. The vector that is returned from this minimization contains all the information for our class split since any negative datapoints belong to one class and any possitive points belong to another (recall how we defined z using the sign as an indicator of class assignment).


```python
def orth(u, v):
    return (u @ v) / (v @ v) * v

e = np.ones(n) 

d = D @ e

def orth_obj(z):
    z_o = z - orth(z, d)
    return (z_o @ (D - A) @ z_o)/(z_o @ D @ z_o)
```


```python
from scipy.optimize import minimize

#use scipy minimize to minimize orth_obj with respect to z
z_min=minimize(orth_obj,z, jac='2-point')
```


```python
#create a scatter of the points using the minimized result array for coloring based on if they are negative
#or positive
plt.scatter(X[:,0], X[:,1], c = z_min.x < 0)
```




    <matplotlib.collections.PathCollection at 0x7fc02f08cee0>




    
![output_28_1.png]({{ site.baseurl }}/images/output_28_1.png)    
    


It worked! However this minimization is very clow so we need a new trick. We can use the Rayleigh-Ritz Theorem which tells us that to minimize z we can  fine the smallest eigenvalue of:
$$ (\mathbf{D} - \mathbf{A}) \mathbf{z} = \lambda \mathbf{D}\mathbf{z}\;, \quad \mathbf{z}^T\mathbf{D}\mathbb{1} = 0$$
To solve this we can compute the Laplacian matrix of our similarity matrix and then take the second smallest eigenvalue which coresponds to the z which we want.


```python
from scipy.sparse.linalg import eigsh

#compute the inverse of D with np.linalg.inv
inv_D=np.linalg.inv(D)

#compute the matrix L
L=inv_D@(D-A)

#use eigsh with matrix L to find the eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(L) 
```


```python
print(eigenvalues)

#seperate the 2nd column, corisponding to our eigenvector with the 
#second lowest eigenvalue
z_eig=eigenvectors[:, 1]
```

    [-6.24500451e-17  5.38441205e-03  3.56083377e-02  3.20091857e-02
      1.11548322e-01  1.27235106e-01  2.51774053e-01  2.85918254e-01
      4.07425093e-01  4.24605995e-01  5.94580400e-01  5.99457688e-01
      7.20463457e-01  7.63327594e-01  1.27955514e+00  1.27385633e+00
      1.26111355e+00  1.25858658e+00  1.24441877e+00  1.23560889e+00
      1.22931709e+00  1.22230534e+00  8.66003737e-01  1.20162741e+00
      1.18962569e+00  1.17848089e+00  1.17137178e+00  8.91636987e-01
      8.95544608e-01  1.15463072e+00  1.14861569e+00  1.14219005e+00
      9.07192465e-01  9.12675323e-01  9.17978437e-01  9.30655907e-01
      9.37226660e-01  9.36340763e-01  9.41883243e-01  9.47233250e-01
      9.56726160e-01  9.60888022e-01  9.68921758e-01  9.67681657e-01
      1.13775488e+00  1.13443015e+00  1.12986833e+00  1.12694700e+00
      1.12349822e+00  1.11924533e+00  1.11746817e+00  1.11362530e+00
      1.10897299e+00  1.11147334e+00  1.11199019e+00  9.72943231e-01
      9.74995956e-01  9.76089669e-01  9.77092940e-01  9.81864284e-01
      9.81127330e-01  1.10663306e+00  1.10557702e+00  1.10425902e+00
      1.09881785e+00  1.09791211e+00  1.09772237e+00  1.09570625e+00
      1.09320347e+00  9.84971190e-01  9.87477828e-01  9.90484824e-01
      9.93434258e-01  9.94549331e-01  9.93797811e-01  9.95572599e-01
      1.08991268e+00  1.08909925e+00  1.08774022e+00  1.08649696e+00
      1.08555583e+00  1.08367352e+00  1.08426120e+00  1.07837562e+00
      1.08012029e+00  1.08184242e+00  1.08154391e+00  1.07733897e+00
      1.07691783e+00  9.97733846e-01  9.98774291e-01  9.98690921e-01
      1.07543721e+00  9.99836819e-01  1.00234700e+00  1.07176337e+00
      1.07227523e+00  1.07313497e+00  1.07500346e+00  1.00320359e+00
      1.06827555e+00  1.06787699e+00  1.07068524e+00  1.07132677e+00
      1.07004538e+00  1.00550385e+00  1.00600900e+00  1.00732950e+00
      1.00942353e+00  1.06009719e+00  1.06079635e+00  1.06417969e+00
      1.06339756e+00  1.06388736e+00  1.06391112e+00  1.01129468e+00
      1.06666667e+00  1.01195324e+00  1.05467252e+00  1.05365769e+00
      1.02234142e+00  1.02458608e+00  1.02545362e+00  1.02603429e+00
      1.02860869e+00  1.03096170e+00  1.05277870e+00  1.05147393e+00
      1.01361598e+00  1.01445967e+00  1.01811823e+00  1.01773882e+00
      1.02002478e+00  1.03321506e+00  1.03003617e+00  1.03049248e+00
      1.03601575e+00  1.03693048e+00  1.01269212e+00  1.01511142e+00
      1.01652296e+00  1.01578183e+00  1.01633060e+00  1.01552663e+00
      1.03855908e+00  1.04685655e+00  1.04755066e+00  1.04585159e+00
      1.04027661e+00  1.04119472e+00  1.04632889e+00  1.04243124e+00
      1.04202282e+00  1.04372928e+00  1.04347476e+00  1.06250000e+00
      1.05555556e+00  1.04166495e+00  1.04165124e+00  1.03846154e+00
      1.04761905e+00  1.04545455e+00  1.04000000e+00  1.04166667e+00
      1.04347826e+00  1.04347826e+00  1.06666667e+00  1.06250000e+00
      1.04761905e+00  1.03846154e+00  1.04545455e+00  1.04000000e+00
      1.04166667e+00  1.04166667e+00  1.06250000e+00  1.03846154e+00
      1.04761905e+00  1.04545455e+00  1.04000000e+00  1.06666667e+00
      1.06666667e+00  1.06666667e+00  1.04347826e+00  1.04347826e+00
      1.04347826e+00  1.04761905e+00  1.03846154e+00  1.03846154e+00
      1.04545455e+00  1.04545455e+00  1.04000000e+00  1.04000000e+00
      1.04347826e+00  1.04166667e+00  1.04166667e+00  1.04166667e+00
      1.04166667e+00  1.04347826e+00  1.04347826e+00  1.04347826e+00]



```python
#create a scatter of the points using the minimized result array for coloring
plt.scatter(X[:,0], X[:,1], c = z_eig < 0)
```




    <matplotlib.collections.PathCollection at 0x7fc02ec442b0>




    
![output_32_1.png]({{ site.baseurl }}/images/output_32_1.png)    
    


Now we can put everything we've done into one function which will compute the spectral cluster of our points. 

## Part G

Synthesize your results from the previous parts. In particular, write a function called `spectral_clustering(X, epsilon)` which takes in the input data `X` (in the same format as Part A) and the distance threshold `epsilon` and performs spectral clustering, returning an array of binary labels indicating whether data point `i` is in group `0` or group `1`. Demonstrate your function using the supplied data from the beginning of the problem. 

#### Notes

Despite the fact that this has been a long journey, the final function should be quite short. You should definitely aim to keep your solution under 10, very compact lines. 

**In this part only, please supply an informative docstring!** 

#### Outline

Given data, you need to: 

1. Construct the similarity matrix. 
2. Construct the Laplacian matrix. 
3. Compute the eigenvector with second-smallest eigenvalue of the Laplacian matrix. 
4. Return labels based on this eigenvector. 


```python
def spectral_clustering(X, epsilon):
    '''
    This Function takes in a dataset X and a distance epsilon and returns the predicted catigory
    of each datapoint in the set. The epsilon value is used to set how close each point needs to 
    be to another for the algorithm to consider them 'close'. 
    
    @params
    X-(array): an array of points that the user wishs to classify
    epsilon(float): a number representing what euclidien distance is 'close'
    
    @return
    z_eig(vector): a vector of 1s and 0s representing the classification of each point
    '''
    #construct the similarity matrix
    A=make_sim_matrix(X,epsilon)
    
    #compute the degree of A
    D_val= compute_degree(A)
    
    #create the matrix of zeros and populate the diagonal with the degree values 
    D=np.zeros((X.shape[0],X.shape[0]))
    np.fill_diagonal(D,D_val)
    
    #compute the inverse of D with np.linalg.inv
    inv_D=np.linalg.inv(D)

    #compute the matrix L
    L=inv_D@(D-A)

    #use eigsh with matrix L to find the eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(L) 
    #take the eigenvector assosiated with the second smallest eigenvalue
    z_eig=eigenvectors[:, 1]
    
    #return a vector of bools of which points are greater than 0
    return (z_eig < 0)
    
```


```python
result =spectral_clustering(X,0.4)

plt.scatter(X[:,0], X[:,1], c = result)
```




    <matplotlib.collections.PathCollection at 0x7fc0309fb370>




    
![output_36_1.png]({{ site.baseurl }}/images/output_36_1.png)    
    


## Testing

As we can see from the experiments below, as we increase the noise of our data, the datasets start to mix and the spectral clustering algorithm starts being unable to differentiate the half moons. However in most cases it still does a better job than an algorithm like k-means would.


```python
X2, y2 = datasets.make_moons(n_samples=1000, shuffle=True, noise=0.1, random_state=None)
result2 =spectral_clustering(X2,0.4)
plt.scatter(X2[:,0], X2[:,1], c=result2)
```




    <matplotlib.collections.PathCollection at 0x7fc0309268b0>




    
![output_38_1.png]({{ site.baseurl }}/images/output_38_1.png)    
    



```python
X3, y3 = datasets.make_moons(n_samples=1000, shuffle=True, noise=0.12, random_state=None)
result3 =spectral_clustering(X3,0.4)
plt.scatter(X3[:,0], X3[:,1], c=result3)
```




    <matplotlib.collections.PathCollection at 0x7fc018b704c0>




    
![output_39_1.png]({{ site.baseurl }}/images/output_39_1.png)    
    



```python
X4, y4 = datasets.make_moons(n_samples=1000, shuffle=True, noise=0.15, random_state=None)
result4 =spectral_clustering(X4,0.4)
plt.scatter(X4[:,0], X4[:,1], c=result4)
```




    <matplotlib.collections.PathCollection at 0x7fc0194521f0>




    
![output_40_1.png]({{ site.baseurl }}/images/output_40_1.png)    
    


Now we'll try it on a bullseye dataset. 
We will run a series of experiments trying to use our specral clustering function to classify the points of the bullseye dataset. Each experiment will use a different epsilon value and we'll see what works best.


```python
X, y = datasets.make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=None, factor = 0.4)
result =spectral_clustering(X,.1)
plt.scatter(X[:,0], X[:,1], c=result)
```




    <matplotlib.collections.PathCollection at 0x7fc01946e4c0>




    
![output_44_1.png]({{ site.baseurl }}/images/output_44_1.png)    
    



```python
X, y = datasets.make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=None, factor = 0.4)
result =spectral_clustering(X,.3)
plt.scatter(X[:,0], X[:,1], c=result)
```




    <matplotlib.collections.PathCollection at 0x7fc01a7483d0>




    
![output_45_1.png]({{ site.baseurl }}/images/output_45_1.png)    
    



```python
X, y = datasets.make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=None, factor = 0.4)
result =spectral_clustering(X,.4)
plt.scatter(X[:,0], X[:,1], c=result)
```




    <matplotlib.collections.PathCollection at 0x7fc01b947880>




    
![output_46_1.png]({{ site.baseurl }}/images/output_46_1.png)    
    



```python
X, y = datasets.make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=None, factor = 0.4)
result =spectral_clustering(X,.4)
plt.scatter(X[:,0], X[:,1], c=result)
```




    <matplotlib.collections.PathCollection at 0x7fc01ba694c0>




    
![output_47_1.png]({{ site.baseurl }}/images/output_47_1.png)    



```python
X, y = datasets.make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=None, factor = 0.4)
result =spectral_clustering(X,.5)
plt.scatter(X[:,0], X[:,1], c=result)
```




    <matplotlib.collections.PathCollection at 0x7fc01bae0100>




    
![output_48_1.png]({{ site.baseurl }}/images/output_48_1.png)    
    



```python
X, y = datasets.make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=None, factor = 0.4)
result =spectral_clustering(X,.6)
plt.scatter(X[:,0], X[:,1], c=result)
```




    <matplotlib.collections.PathCollection at 0x7fc01bc0ad00>



![output_49_1.png]({{ site.baseurl }}/images/output_49_1.png)    



```python
X, y = datasets.make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=None, factor = 0.4)
result =spectral_clustering(X,.8)
plt.scatter(X[:,0], X[:,1], c=result)
```




    <matplotlib.collections.PathCollection at 0x7fc01bcf9970>




    
![output_50_1.png]({{ site.baseurl }}/images/output_50_1.png)    
    


## Conclusion

On this specific dataset we can see that a epsilon values between .03 and .05 produce the best results and allows us to properly classify each point. Smaller values break up the rings into multiple groups and larger values produce results similar to k-means.


```python

```
