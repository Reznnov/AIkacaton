import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
train_df = pd.read_csv('/home/reznnov/rabota/assets/config/DataKTGA4.csv')

# Добавление столбца 'result' с начальным значением 0
train_df['result'] = 0

# Применение условий
condition1 = (train_df['Feed1'].between(1, 3.5)) | (train_df['Feed2'].between(1, 3.5))
condition2 = (train_df['BenzeneConv'] > 100)
condition3 = (train_df['Selectivity'] > 100)

# Применение условий и обновление 'result'
train_df.loc[condition1 | condition2 | condition3, 'result'] = 0

# Вывод строк, для которых 'result' был обновлен
result_updated_rows = train_df[(condition1 | condition2 | condition3) & (train_df['result'] == 0)]

# Переводим столбцы времени, PresDrop и Selectivity в массивы NumPy
time = train_df['Time'].to_numpy()
presdrop = train_df['PresDrop'].to_numpy()
selectivity = train_df['Selectivity'].to_numpy()

# Построение графика PresDrop и Selectivity от времени
fig, ax1 = plt.subplots(figsize=(12, 6))

# Переводим столбцы времени, PresDrop и Selectivity в массивы NumPy
time = train_df['Time'].to_numpy()
presdrop = train_df['PresDrop'].to_numpy()
selectivity = train_df['Selectivity'].to_numpy()

# Построение графика PresDrop и Selectivity от времени
fig, ax1 = plt.subplots(figsize=(12, 6))

# Ось Y для PresDrop
ax1.set_xlabel('Время')
ax1.set_ylabel('Конверсия', color='tab:blue')
ax1.plot(time, presdrop, marker='o', linestyle='-', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Создаем вторую ось Y для Selectivity
ax2 = ax1.twinx()
ax2.set_ylabel('Selectivity', color='tab:red')
ax2.plot(time, selectivity, marker='x', linestyle='--', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Добавляем заголовок и сетку
plt.title('Зависимость Конверсии и Selectivity от времени')
plt.grid(True)

# Отобразим график
plt.show()

# Вывод результата
print(train_df)

