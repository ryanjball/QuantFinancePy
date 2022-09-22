import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
# import data
def get_data(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start, end)
    exportDF = stockData.head(20).to_csv('export.csv')
    #print(stockData.columns)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    #print(returns)
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix
stockList = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO']

stocks = [stock + '.AX' for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)
meanReturns, covMatrix = get_data(stocks, startDate, endDate)
weights = np.random.random(len(meanReturns))
#print('weights',weights)
weights /= np.sum(weights)
#print('weights',weights)

mc_sims = 100
T = 100

meanM = np.full(shape=(T, len(weights)), fill_value = meanReturns)
meanM = meanM.T
print(meanM)

portfolio_sims = np.full(shape=(T,mc_sims),fill_value=0.0)
print(portfolio_sims)

initialPortfolio = 10000

#DAILY RETURNS
#portfolio is for each day, accumalated product
for m in range(0, mc_sims):
    Z = np.random.normal(size = (T,len(weights)))
    L = np.linalg.cholesky(covMatrix)
    dailyReturns = meanM + np.inner(L,Z)
    portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initialPortfolio

plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('MC simulation of a stock portfolio')
plt.show()
print('done')

