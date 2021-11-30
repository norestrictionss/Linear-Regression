import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    data = pd.read_csv('data.csv')
    height = data.iloc[:, 1].values
    age = data.iloc[:, -2].values
    heights = pd.DataFrame(data=height, index=range(22), columns=["boy"])
    ages = pd.DataFrame(data=age, index=range(22), columns=["yaş"])

    x_train, x_test, y_train, y_test = train_test_split(ages, heights, test_size=0.33, random_state=0)
    sc = StandardScaler()

    x_train = x_train.sort_index()
    y_train = y_train.sort_index()
    x_test = x_test.sort_index()
    y_test = y_test.sort_index()
    x_train = x_train.sort_index()
    y_train = y_train.sort_index()
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    prediction = lr.predict(x_test)

    plt.plot(x_test, y_test)
    plt.plot(x_test, prediction)
    plt.xlabel("Age")
    plt.ylabel("Height in centimeters")
    plt.show()
