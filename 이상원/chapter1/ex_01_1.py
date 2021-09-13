import numpy as np
import pandas as pd
import sklearn.linear_model

from pkg import *

oecd_bli = pd.read_csv('../handson-ml2/datasets/lifesat/oecd_bli_2015.csv',
                       thousands=',')
gdp_per_capita = pd.read_csv('../handson-ml2/datasets/lifesat/gdp_per_capita.csv',
                             thousands=',', delimiter='\t', encoding='latin1',
                             na_values='n/a')

# 데이터 준비
country_status = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_status["GDP per capita"]]
y = np.c_[country_status["Life satisfaction"]]

# 데이터 시각화
country_status.plot(kind='scatter', x='GDP per capita', y='Life satisfaction')
plt.show()

# 선형 모델 선택
model = sklearn.linear_model.LinearRegression()

# 모델 훈련
model.fit(X, y)

# 키프로스에 대한 예측
X_new = [[22587]]
print(model.predict(X_new))
