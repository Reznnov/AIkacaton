import pandas as pd

# Загрузка данных
train_df = pd.read_csv('/home/reznnov/rabota/assets/config/DataKTGA4.csv')

selected_rows = train_df[train_df['Selectivity'] > 100]


pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

pd.set_option('display.max_colwidth', None)



# Вывод первых нескольких строк отфильтрованных данных
print(selected_rows.head())
