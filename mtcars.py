# import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns

# load data
data = pd.read_csv('inflasi1.csv')
data.head()

# statistik deskripsi data inflasi
data.describe()

# grafik timeseries inflasi selama 201 bulan terakhir
data.plot()
pyplot.show()

# box and whisker plots
data.plot(kind='box', subplots=True, layout=(1,1), sharex=False, sharey=False)
plt.show()

# grafik autokorelasi
pd.plotting.autocorrelation_plot(data)


from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
# Analisis ARIMA (Autoregressive Integrated Moving Average)
# fit model
model = ARIMA(data, order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())


from sklearn.metrics import mean_squared_error
# prediction
# train data
X = data.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
