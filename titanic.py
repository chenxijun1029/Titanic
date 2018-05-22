import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("train.csv")
survived = data['Survived']
age = data['Age']
plt.scatter(age, survived)


