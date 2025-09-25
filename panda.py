import pandas as pd


from sklearn.linear_model import LinearRegression

# Load a CSV file
df = pd.read_csv("students.csv")

# See first 5 rows
print(df.head())

# Average score
print(df["score"].mean())
