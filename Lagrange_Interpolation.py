#SyedMuhammadBilal | Github Link: https://github.com/BilalZahid0
from math import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("Interpolation_Dataset.csv")
df = df.iloc[:, 1:]
df.replace("?", pd.NA, inplace=True)
df.rename(columns={"Height (in feet)": "Height"}, inplace=True)
df1 = df.dropna().copy()
days = df1["Day"]
heights = df1["Height"]
x = np.array(days)
y = np.array(heights)
x = x.astype(float)
y = y.astype(float)
indic = df[df.isna().any(axis=1)].index
test = df.iloc[indic].copy()
x_new = test["Day"].values
import numpy as np

# Find the Lagrange polynomial through the points (x, y) and return its value at t.
def lagrange(x, y, t):

    if len(x) != len(y):
        raise ValueError("The arrays x and y must have the same length.")

    # Initialize the polynomial
    p = 0

    # Loop over the points
    for i in range(len(x)):
        # Get the current point
        xi, yi = x[i], y[i]

        # Initialize the term
        term = yi

        # Loop over the other points
        for j in range(len(x)):
            # Skip the current point
            if i == j:
                continue

            # Multiply the term by the appropriate factor
            term *= (t - x[j]) / (xi - x[j])

        # Add the term to the polynomial
        p += term

    return p

error_=[100 for i in range(6)]
prev_=[0 for i in range(6)]
esti=[]
for i in range(10,60):
    if i%10==0 or i==59:
        print(f"FOR {i}th interpolation result is:")
        for value in x_new:

            print("\nValue at", value, "is", round(lagrange(x[0:i], y[0:i], value), 2))
            esti.append(lagrange(x[0:i], y[0:i], value))
        for j in range(0,6):         
            error_.append(abs(esti[j]-prev_[j])/abs(esti[j])*100)
        prev_=esti.copy()
        print(error_)
        plt.figure(figsize=(12, 8))
        plt.plot(x[0:i], y[0:i], 'bo')
        plt.title(f"Actual Values for {i}th interpolation")
        plt.show()

        plt.plot(x_new, esti, "bo")
        plt.xlabel('Day')
        plt.ylabel('Height')
        plt.title(f"Predicted Values for {i}th interpolation")
        plt.show()