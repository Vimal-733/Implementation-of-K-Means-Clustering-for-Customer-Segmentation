# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step1: Import the necessary packages using import statement. 

Step2: Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

Step3: Import KMeans and use for loop to cluster the data.

Step4: Predict the cluster and plot data graphs. Step5: Print the outputs and end the program
## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed byA.VIMAL 
RegisterNumber: 212224240183
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("Mall_Customers (1).csv")

# Basic info
print(data.head())
print(data.info())
print(data.isnull().sum())

# Selecting features (Annual Income, Spending Score)
X = data.iloc[:, 3:]

# -----------------------------
# Elbow Method (WCSS)
# -----------------------------
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# -----------------------------
# Final KMeans Model
# -----------------------------
km = KMeans(n_clusters=5, init="k-means++", n_init=10, random_state=42)
y_pred = km.fit_predict(X)

# Add cluster to dataset
data["cluster"] = y_pred

# -----------------------------
# Split clusters
# -----------------------------
df0 = data[data["cluster"] == 0]
df1 = data[data["cluster"] == 1]
df2 = data[data["cluster"] == 2]
df3 = data[data["cluster"] == 3]
df4 = data[data["cluster"] == 4]

# -----------------------------
# Plot clusters
# -----------------------------
plt.scatter(df0["Annual Income (k$)"], df0["Spending Score (1-100)"], c="red", label="Cluster 0")
plt.scatter(df1["Annual Income (k$)"], df1["Spending Score (1-100)"], c="blue", label="Cluster 1")
plt.scatter(df2["Annual Income (k$)"], df2["Spending Score (1-100)"], c="green", label="Cluster 2")
plt.scatter(df3["Annual Income (k$)"], df3["Spending Score (1-100)"], c="cyan", label="Cluster 3")
plt.scatter(df4["Annual Income (k$)"], df4["Spending Score (1-100)"], c="orange", label="Cluster 4")

# Plot centroids
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            c="black", marker="X", s=200, label="Centroids")

plt.legend()
plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()
```

## Output:

<img width="996" height="738" alt="image" src="https://github.com/user-attachments/assets/942d346d-6d2d-4934-a361-7c2b5d81b3b5" />

<img width="897" height="673" alt="image" src="https://github.com/user-attachments/assets/59a86627-e950-40bf-a667-fb7494b34f9f" />

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
