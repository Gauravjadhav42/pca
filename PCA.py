#!/usr/bin/env python
# coding: utf-8

# ### Load Data

# In[1]:


from sklearn import datasets


# In[2]:


iris = datasets.load_iris()


# In[3]:


X = iris.data
y = iris.target


# In[4]:


X.shape


# In[5]:


iris.feature_names


# In[6]:


iris.target_names


# ### Visualize Data

# In[7]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# Lets create a 3D Graph

# In[8]:


fig = plt.figure(1, figsize=(10, 8))

plt.clf()

ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()

for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(X[y == label, 0].mean(),
              X[y == label, 1].mean() + 1.5,
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Change the order of labels, so that they match
y_clr = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_clr, cmap=cm.get_cmap("nipy_spectral"))

plt.show()


# ### Building a Classifier

# In[9]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# Split Train and Test Data

# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, 
                                                    stratify=y, 
                                                    random_state=42)


# Build a Decision Tree

# In[11]:


model = DecisionTreeClassifier(max_depth=2, random_state=42)
model.fit(X_train, y_train)


# Check Accuracy

# In[12]:


model.score(X_test, y_test)


# ### Using PCA

# In[13]:


from sklearn.decomposition import PCA


# Centering the Data

# In[14]:


X_centered = X - X.mean(axis=0)


# PCA with 2 components

# In[15]:


pca = PCA(n_components=2)
pca.fit(X_centered)


# Get new dimensions

# In[16]:


X_pca = pca.transform(X_centered)


# In[17]:


X_pca.shape


# Plotting Iris data using 2 PCs

# In[18]:


fig = plt.figure(1, figsize=(10, 8))

plt.plot(X_pca[y == 0, 0], X_pca[y == 0, 1], 'bo', label='Setosa')
plt.plot(X_pca[y == 1, 0], X_pca[y == 1, 1], 'go', label='Versicolour')
plt.plot(X_pca[y == 2, 0], X_pca[y == 2, 1], 'ro', label='Virginica')
plt.legend(loc=0)

plt.show()


# ### Exploring PCA 

# Check EigenVectors or PC 1/2

# In[19]:


pca.components_


# In[20]:


pca.explained_variance_


# In[21]:


pca.explained_variance_ratio_


# ### Building Classifier using PCA features

# In[22]:


model = DecisionTreeClassifier(max_depth=2, random_state=42)
model.fit(pca.transform(X_train), y_train)


# In[23]:


model.score(pca.transform(X_test), y_test)


# In[ ]:


INFERENCE :- PCA techique  Help to improve the Accuracy of model by selecting important features.

