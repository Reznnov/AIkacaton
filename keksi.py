import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Загрузка данных
df = pd.read_csv('/home/reznnov/rabota/assets/config/ML_MWW_R2.csv')

# Разделение данных на признаки и целевые переменные
X = df[['Time']]
y_admixture = df['Admixture']
y_presdrop = df['PresDrop']

# Разделение данных на тренировочный и тестовый наборы
X_train, X_test, y_admixture_train, y_admixture_test, y_presdrop_train, y_presdrop_test = train_test_split(X, y_admixture, y_presdrop, test_size=0.2, random_state=42)

# Создание и обучение модели для Admixture
model_admixture = LinearRegression()
model_admixture.fit(X_train, y_admixture_train)

# Создание и обучение модели для PresDrop
model_presdrop = LinearRegression()
model_presdrop.fit(X_train, y_presdrop_train)

# Прогнозирование значений Admixture и PresDrop
y_admixture_pred = model_admixture.predict(X_test)
y_presdrop_pred = model_presdrop.predict(X_test)

# Оценка моделей
mse_admixture = mean_squared_error(y_admixture_test, y_admixture_pred)
mse_presdrop = mean_squared_error(y_presdrop_test, y_presdrop_pred)

print(f'Mean Squared Error (Admixture): {mse_admixture}')
print(f'Mean Squared Error (PresDrop): {mse_presdrop}')

# Визуализация результатов (пример)
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(X_test['Time'].values, y_admixture_test.values, label='Actual Admixture')
plt.plot(X_test['Time'].values, y_admixture_pred, label='Predicted Admixture')
plt.legend()
plt.title('Admixture Prediction')

plt.subplot(2, 1, 2)
plt.plot(X_test['Time'].values, y_presdrop_test.values, label='Actual PresDrop')
plt.plot(X_test['Time'].values, y_presdrop_pred, label='Predicted PresDrop')
plt.legend()
plt.title('PresDrop Prediction')

plt.tight_layout()
plt.show()