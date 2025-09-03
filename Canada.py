import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

csv_path = "C:/Users/HP 745 G6/Desktop/ML/Linear Regression/Canada_per_capita_income.csv"
# Load the dataset from the file path
df = pd.read_csv(csv_path)
plt.xlabel('Year')
plt.ylabel('Per Capita Income (US$)')
plt.scatter(df['year'], df['per_capita_income'], color='red', marker='+')

reg = linear_model.LinearRegression()
reg.fit(df[['year']], df['per_capita_income'])   # X=year, y=income

plt.plot(df['year'], reg.predict(df[['year']]), color='blue')
plt.show()

#predict income for 2020
prediction = reg.predict(pd.DataFrame({'year': [2020]}))
print("Predicted per capita income for 2020:", prediction[0])