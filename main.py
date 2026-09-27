import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("Salary_dataset.csv")
df.head()

X = df["YearsExperience"].to_numpy()
Y = df["Salary"]

X_norm = (X - np.mean(X)) / (np.std(X))
Y_norm = (Y - np.mean(Y)) / (np.std(Y))
print(X_norm.shape)
print(Y_norm.shape)


def cost(Y, Y_hat):
  return (1/len(Y))*np.sum((Y_hat-Y)**2)

def dCdm(X, Y, Y_hat):
  return (2/len(Y))*np.sum((Y_hat-Y)*X)

def dCdb(Y, Y_hat):
  return (2/len(Y))*np.sum((Y_hat-Y))


def gradient_descent(X, Y, m, b, learning_rate, epoch):
    for i in range(epoch):
        Y_hat = m * X + b
        m = m - learning_rate * dCdm(X, Y, Y_hat)
        b = b - learning_rate * dCdb(Y, Y_hat)

        C = cost(Y, Y_hat)
        print(f"Cost in epoch {i} is {C}")

    return m, b


m = 0 ; b = 0
m,b = gradient_descent(X_norm, Y_norm, m, b, learning_rate=0.01, epoch=1000)
print(f"The final values are {m} and {b}")

def predict(m,b, X):
  return m*X + b

Y_hat = predict(m,b,X_norm)
cost  = cost(Y_norm, Y_hat)

# Y_pred_original = Y_hat*np.std(Y) + np.mean(Y)
# Y_pred_original
#
# df = pd.DataFrame({
#     "Actual": Y,
#     "Predicted": Y_pred_original,
#     "Difference": Y - Y_pred_original
# })

plt.scatter(X_norm, Y_norm, marker="x", c="r")
plt.plot(X_norm, Y_hat, c="b")
plt.show()