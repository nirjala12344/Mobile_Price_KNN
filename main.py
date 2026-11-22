
# MOBILE PRICE RANGE CLASSIFICATION USING KNN
# Dataset Analysis

import pandas as pd
import matplotlib.pyplot as plt

# Read dataset
data = pd.read_csv("dataset/train.csv")

# Display first 5 records
print("\nFIRST FIVE RECORDS")
print(data.head())

# Dataset size
print("\nDATASET SHAPE")
print(data.shape)

# Dataset information
print("\nDATASET INFORMATION")
data.info()

# Check missing values
print("\nMISSING VALUES")
print(data.isnull().sum())

# Price category count
print("\nPRICE RANGE COUNTS")
print(data["price_range"].value_counts())

# Plot price categories
data["price_range"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Mobile Price Range Distribution")
plt.xlabel("Price Range")
plt.ylabel("Number of Mobiles")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print("\nDataset analysis completed!")
