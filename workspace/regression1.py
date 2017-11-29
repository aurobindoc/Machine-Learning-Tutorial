import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
import pickle

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
warnings.simplefilter("ignore", category=DeprecationWarning)

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume', ]]

df['HL_Percentage'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['Change_Percentage'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_Percentage', 'Change_Percentage', 'Adj. Volume']]

# Column to be forecast
forecast_col = 'Adj. Close'

# Fill missing values with outside data
df.fillna(-999999, inplace=True)

# Number of prediction
forecast_out = int(math.ceil(0.01 * len(df)))

print(forecast_out)
# Label defination
df['Label'] = df[forecast_col].shift(-forecast_out)

# Feature = X and Label = y
X = np.array(df.drop(['Label'], 1))

# Scale X such that it has 0 mean and 1 std. dev
X = preprocessing.scale(X)

X_lately = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace=True)

y = np.array(df['Label'])

# Create training and testing Data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# Classifier defination
clf = LinearRegression()
clf.fit(X_train, y_train)

# Dump Classifier into pickle
with open("linear_regression.pickle", "wb") as f:
    pickle.dump(clf, f)

# Load from Pickle
clf = pickle.load(open("linear_regression.pickle", "rb"))
# Test the data
accuracy = clf.score(X_test, y_test)

# Get prediction for next <forecast_out> days
forecast_set = np.array(clf.predict(X_lately))
print(forecast_set)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
print(last_date)
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

print("=========================================")
print("=========================================")

# Plot a graph
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
