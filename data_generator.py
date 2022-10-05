import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from plot import line_plot

np.random.seed(123)

num_data = 360


# generate x
x = [datetime(2021, 1, 1)]
for i in range(num_data-1): 
    x_new = x[-1] + timedelta(days=1)
    x.append(x_new)
x = np.array(x)

# generate y
value_range = (3,8)

values = [2.5]
random_speed = 1
for i in range(num_data-1):
    value = values[-1]
    if value > value_range[1]+np.random.uniform(-2, 2):
        value = value_range[0]+np.random.uniform(-1, 2)
        random_speed = np.random.uniform(0.2, 1.5)
    random_speed *= 0.99
    value += round(np.random.uniform(-0.1, 2)*random_speed, 2)
    values.append(value)
    
# plot
print("plot")
line_plot(range(num_data), values, x_name="time", y_name="value", save_path="./plot/fake_data.png")

# save
data = np.stack((x,values),1)
data = pd.DataFrame(data, columns=["x", "y"])

data.to_csv("./raw_data/fake_data.csv", index=False)