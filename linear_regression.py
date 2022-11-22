import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

print("The length of the data set is:", len(x))

plt.plot(x, y, "b.")
plt.xlabel("Equipment affected(u/1000)")
plt.ylabel("Incident cost (u/10000)")
plt.show()

data = {'n_equipment_affected': x.flatten(), 'cost': y.flatten()}
df = pd.DataFrame(data)
print(df.head(10))

df['n_equipment_affected'] = df['n_equipment_affected'] * 1000
df['n_equipment_affected'] = df['n_equipment_affected'].astype('int')
df['cost'] = df['cost'] * 10000
df['cost'] = df['cost'].astype('int')
print(df.head(10))

plt.plot(df['n_equipment_affected'], df['cost'], "b.")
plt.xlabel("Equipment affected")
plt.ylabel("Incident cost")
plt.show()

lin_reg = LinearRegression()
lin_reg.fit(df['n_equipment_affected'].values.reshape(-1, 1),
            df['cost'].values)

print(lin_reg.intercept_)
print(lin_reg.coef_)

x_min_max = np.array([[df["n_equipment_affected"].min()],
                     [df["n_equipment_affected"].max()]])
y_train_pred = lin_reg.predict(x_min_max)

plt.plot(x_min_max, y_train_pred, "g-")
plt.plot(df['n_equipment_affected'], df['cost'], "b.")
plt.xlabel("Equipment affected")
plt.ylabel("Incident cost")
plt.show()

x_new = np.array([[1200]])
predicted_cost = lin_reg.predict(x_new)

print("The incident cost would be:", int(predicted_cost[0]), "$")

plt.plot(df['n_equipment_affected'], df['cost'], "b.")
plt.plot(x_min_max, y_train_pred, "g-")
plt.plot(x_new, predicted_cost, "rx")
plt.xlabel("Equipment affected")
plt.ylabel("Incident cost")
plt.show()
