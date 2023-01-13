# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 14:08:21 2022

@author: rahel dzmitry
"""
#Подключение всех необходимых для обработки данных и отладки модели библиотек
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
from itertools import product
import numpy as np

def invboxcox(y,lmbda):
   if lmbda == 0:
      return(np.exp(y))
   else:
      return(np.exp(np.log(lmbda*y+1)/lmbda))
  
#Построение графика изменения среднемесячной номинальной заработной платы  
df = pd.read_csv('C:\Python27\WAG_C_M.csv',';', index_col=['month'], parse_dates=['month'], dayfirst=True)

plt.figure(figsize=(15,7))
df.WAG_C_M.plot()
plt.ylabel(u'Средняя номинальная заработная плата')
plt.xlabel(u'Период наблюдения')
 
print ('1. Минимальная среднемесячная номинальная заработная плата равная {} рублей зафиксирована {}'.format(df[df['WAG_C_M'] == df['WAG_C_M'].min()].values[0, 0],df[df['WAG_C_M'] == df['WAG_C_M'].min()].index[0].date()))
print ('2. Максимальная среднемесячная номинальная заработная плата равная {} рублей зафиксирована {}'.format(df[df['WAG_C_M'] == df['WAG_C_M'].max()].values[0, 0],df[df['WAG_C_M'] == df['WAG_C_M'].max()].index[0].date()))
print ("3. Критерий Дики-Фуллера: p=%f" % sm.tsa.stattools.adfuller(df.WAG_C_M)[1])

#Выделение тренда и сезонности в нашем ряду
plt.figure(figsize=(15,10))
sm.tsa.seasonal_decompose(df.WAG_C_M).plot()
plt.show()

#На основании построенных графиков:
#Четко прослеживается тренд: плавное долгосрочное увеличение уровня.
#Четко прослеживается сезонность: циклическое изменение уровня с постоянным периодом.
#Дисперсия значений в начале значительно меньше дисперсии значений в конце.

#Для построения модели, обощающей временной ряд требуется обеспечить стационарность:
#Стабилизировать дисперсию в течении всего временного ряда.
#Провести дифференцирование.

#Выполнение необходимых преобразований и визуализация временного ряда
df['WAG_C_M_boxcox'], lmbda = stats.boxcox(df.WAG_C_M)
plt.figure(figsize=(15,7))
df.WAG_C_M_boxcox.plot()
plt.ylabel(u'Преобразованная средняя номинальная заработная плата')
plt.xlabel(u'Период наблюдения')
plt.show()

print ("Критерий Дики-Фуллера: p=%f" % sm.tsa.stattools.adfuller(df.WAG_C_M_boxcox)[1])

#Результат сезонного дифференцирования:
df['WAG_C_M_boxcox_diff'] = df.WAG_C_M_boxcox - df.WAG_C_M_boxcox.shift(12)
plt.figure(figsize=(15,10))
sm.tsa.seasonal_decompose(df.WAG_C_M_boxcox_diff[12:]).plot()
plt.show()

print ("Критерий Дики-Фуллера: p=%f" % sm.tsa.stattools.adfuller(df.WAG_C_M_boxcox_diff[12:])[1])

#Влияние тренда и сезонных факторов стало малым. 
df['WAG_C_M_boxcox_diff12and2'] = df.WAG_C_M_boxcox_diff - df.WAG_C_M_boxcox_diff.shift(1)
plt.figure(figsize=(15,10))
sm.tsa.seasonal_decompose(df.WAG_C_M_boxcox_diff12and2[13:]).plot() 
plt.show()

print ("Критерий Дики-Фуллера: p=%f" % sm.tsa.stattools.adfuller(df.WAG_C_M_boxcox_diff12and2[13:])[1])

#Снизилось влияние тренда, увеличилось влияние сезонных факторов, но при этом оно тоже является малым.
#С учетом этого построение необходимых коррелогамм.
plt.figure(figsize=(15,8))
ax = plt.subplot(211)
sm.graphics.tsa.plot_acf(df.WAG_C_M_boxcox_diff12and2[13:].values.squeeze(), lags=48, ax=ax)
plt.show()
ax = plt.subplot(212)
sm.graphics.tsa.plot_pacf(df.WAG_C_M_boxcox_diff12and2[13:].values.squeeze(), lags=48, ax=ax)
plt.show()

#Выбор оптимальных значений на основании минимума критерия Акаике.
ps = range(0, 2)
d=1
qs = range(0, 2)
Ps = range(0, 4)
D=1
Qs = range(0, 1)

parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
print ("Всего комбинаций параметров:", len(parameters_list))

#Обучение моделей для созданных комбинаций с учетом расчетного времени.
import time
start = time.time()

results = []
best_aic = float("inf")
warnings.filterwarnings('ignore')

for param in parameters_list:
    try:
        model=sm.tsa.statespace.SARIMAX(df.WAG_C_M_boxcox, order=(param[0], d, param[1]), 
                                        seasonal_order=(param[2], D, param[3], 12)).fit(disp=-1)
    #параметры, на которых модель не обучается и переходим к следующему набору
    except ValueError:
        print ('wrong parameters:', param)
        continue
    aic = model.aic
    #сохраняем лучшую модель, aic, параметры
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param, model.aic])
    
warnings.filterwarnings('default')

end = time.time()
print("wall time:", end - start)

#Вывод лучших результатов.
result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic']
print (result_table.sort_values(by = 'aic', ascending=True).head())

#Вывод параметров лучшей модели.
print (best_model.summary())

#Анализ остатков полученной модели.
plt.figure(figsize=(15,8))
plt.subplot(211)
best_model.resid[13:].plot()
plt.ylabel(u'Остатки')

ax = plt.subplot(212)
sm.graphics.tsa.plot_acf(best_model.resid[13:].values.squeeze(), lags=48, ax=ax)

print ("Критерий Стьюдента: p=%f" % stats.ttest_1samp(best_model.resid[13:], 0)[1])
print ("Критерий Дики-Фуллера: p=%f" % sm.tsa.stattools.adfuller(best_model.resid[13:])[1])

#Остатки несмещены (на основании критерия Стьюдента) стационарны (на основании критерия Дики-Фуллера), неавтокоррелированы (критерий Льюнга-Бокса и подтверждается коррелограммой)
warnings.filterwarnings('ignore')
df['model'] = invboxcox(best_model.fittedvalues, lmbda)
plt.figure(figsize=(15,7))
plot1 = df.WAG_C_M.plot()
plot2 = df.model[13:].plot(color='r')
plt.ylabel(u'Средняя номинальная заработная плата')
plt.xlabel(u'Период наблюдения')
plt.legend( [u'Фактические данные', u'Смоделированые данные'], loc=1, ncol = 2, prop={'size':10})
plt.show()
warnings.filterwarnings('default')
#На основании графика необходимо отметить, что модель описывает ряд данных достаточно точно.

#Построение графика средней номинальной заработной платы и ее прогноза на два следующих года
import datetime as dt
from dateutil.relativedelta import relativedelta
df2 = df[['WAG_C_M']]
date_list = [dt.datetime.strptime("2016-08-01", "%Y-%m-%d") + relativedelta(months=x) for x in range(0,36)]
future = pd.DataFrame(index=date_list, columns= df2.columns)
df2 = pd.concat([df2, future])
df2['forecast'] = invboxcox(best_model.predict(start=283, end=320), lmbda)

plt.figure(figsize=(15,7))
df2.WAG_C_M.plot()
df2.forecast.plot(color='r')
plt.ylabel(u'Средняя номинальная заработная плата, руб.')
plt.title(u'Средняя номинальная заработная плата и ее прогноз на следующие 2 года')
plt.show()
