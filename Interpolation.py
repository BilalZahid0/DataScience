#SyedMuhammadBilal | Github Link: https://github.com/BilalZahid0

#Libraries used:
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Reading, Replacing and Renaming the columns of the dataset:
df = pd.read_csv("Interpolation_Dataset.csv")
df = df.iloc[:, 1:]
df.replace("?", pd.NA, inplace=True)
df.rename(columns={"Height (in feet)": "Height"}, inplace=True)


# Calculating the divided differences table:
def divided_diff(x, y):
    n = len(y)
    coef = np.zeros([n, n])
    # the first column is y
    coef[:, 0] = y
    
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])
    return coef

# Extract days and values columns
days = df['Day'].values
values = pd.to_numeric(df['Height'], errors='coerce').astype(float)


# Find indices of missing values
missing_indices = np.isnan(values)


# Polynomial Interpolation for missing values
degree = 2
coefficients = np.polyfit(days[~missing_indices], values[~missing_indices], degree)
poly_interp = np.poly1d(coefficients)
values_interp_poly = poly_interp(days[missing_indices])

# Replace missing values with polynomial interpolated values
values[missing_indices] = values_interp_poly

values_interp_poly = np.interp(np.where(missing_indices)[0], np.where(~missing_indices)[0], values[~missing_indices])


# Replace missing values with polynomial interpolated values
values[missing_indices] = values_interp_poly


# Plot original 
plt.figure(figsize=(10, 6))
plt.scatter(days, values, label='Original Data')
plt.title('Polynomial Interpolation of Missing Values')
plt.xlabel('Days')
plt.ylabel('Height (in feet)')
plt.legend()
plt.show()


# Evaluating the newton polynomial at x:
def newton_poly(coef, x_data, x):
    n = len(x_data) - 1 
    p = coef[n]
    for k in range(1, n + 1):
        p = coef[n - k] + (x - x_data[n - k]) * p
    return p

df1 = df.dropna().copy()
days = df1["Day"]
heights = df1["Height"]
x = np.array(days)
y = np.array(heights)

indic = df[df.isna().any(axis=1)].index
test = df.iloc[indic].copy()
x_new = test["Day"].values


# Evaluating with Newton's interpolation
for i in range(10,60):
    if i%10==0 or i==59:
        # Divided difference coefficients
        a_s = divided_diff(x[0:i], y[0:i])[0, :]
        print(f"Values at {i}th interpolation :{newton_poly(a_s, x[0:i], x_new)} ")
        y_new_newton = newton_poly(a_s, x[0:i], x_new)

        plt.figure(figsize=(8, 8))
        plt.plot(x[0:i], y[0:i], "bo")
        plt.title(f"Actual Values for {i}th interpolation")
        print(y_new_newton)
        plt.show()
        
        plt.figure(figsize=(8, 8))
        plt.plot(x_new, y_new_newton, "bo")
        plt.xlabel('Day')
        plt.ylabel('Height')
        plt.title(f"Predicted Values for {i}th interpolation")
        plt.show()