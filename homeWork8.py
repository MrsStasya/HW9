# Задание
# Скачайте датасет House Prices Kaggle со страницы конкурса (https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
# и сохраните его в том же каталоге, что и ваш скрипт или блокнот Python.

# Загрузите датасет в pandas DataFrame под названием df.

# Выполните предварительную обработку данных, выполнив следующие шаги: 
# a. Определите и обработайте отсутствующие значения в датасете. 
# Определите, в каких столбцах есть отсутствующие значения, и решите, как их обработать (например, заполнить средним, 
# медианой или модой, или отбросить столбцы/строки с существенными отсутствующими значениями). 
# b. Проверьте и обработайте любые дублирующиеся строки в датасете. 
# c. Проанализируйте типы данных в каждом столбце и при необходимости преобразуйте их (например, из объектных в числовые типы).

# Проведите разведочный анализ данных (EDA), ответив на следующие вопросы:
# a. Каково распределение целевой переменной 'SalePrice'? Есть ли какие-либо выбросы?
# b. Исследуйте взаимосвязи между целевой переменной и другими характеристиками. Есть ли сильные корреляции?
# c. Исследуйте распределение и взаимосвязи других важных характеристик, таких как 'OverallQual', 'GrLivArea', 'GarageCars' и т.д.
# d. Визуализируйте данные, используя соответствующие графики (например, гистограммы, диаграммы рассеяния, квадратные диаграммы),
# чтобы получить представление о датасете.

# Выполните проектирование признаков путем реализации следующих преобразований:
# a. Работайте с категориальными переменными, применяя one-hot encoding или label encoding, в зависимости от характера переменной.
# b. При необходимости создайте новые характеристики, такие как общая площадь или возраст объекта недвижимости,
# путем объединения существующих характеристик.

# Сохраните очищенный и преобразованный набор данных в новый CSV-файл под названием 'cleaned_house_prices.csv'.

import pandas as pd #для обработки и анализа данных
import numpy as np # библиотека для работы с массивами данных
import matplotlib.pyplot as plt # модуль для построения графиков
import seaborn as sns # библиотека для визуализации данных, основанная на matplotlib
from sklearn.preprocessing import LabelEncoder # инструмент для кодирования категориальных переменных
from scipy import stats # библиотека для научных и математических вычислений


# Установка стиля и цветов палитры для графиков
sns.set(style='whitegrid')

# Загрузка данных
file_path = "train.csv"
df = pd.read_csv(file_path)

# Вывод датасета. функция df.head читает первые 5 строк
print('Первые строки датасета:')
print(df.head())

# print("\n статистика: ")
print(df.describe())

# Проверка того, в каких столбцах отсутствуют значения
# Настроим pd, чтобы выводились все строки 
pd.set_option('display.max_rows', None)
# Проверка того, в каких столбцах отсутствуют значения
print(df.isnull().sum())
# Сбросим настройки вывода строк
pd.reset_option('display.max_rows')

# Для работы возьмем столбец LotFrontage. И заполним его средними
mediange_LotFrontage = df['LotFrontage'].median()
df['LotFrontage'] = df['LotFrontage'].fillna(mediange_LotFrontage)
#print(df['LotFrontage'])
 
 #Возьмем следующий столбец Alley. Есть отсутствующие значения, а также есть непонятное значение Grvl
 # Проведем сначала замену Grcl на Gravel и далее заменим их на моду
df['Alley'] = df['Alley'].replace({'Grvl', 'Gravel'})
mod_Alley = df['Alley'].mode()[0]
df['Alley'] = df['Alley'].fillna(mod_Alley)
#print(df['Alley'])

# Посколько каждый столбец с пропущенными данными обрабатывать долго, всем пропущенным цифровым данным мы присвоим средние значения.
# 1. Сначала выберем все числовые колонки
num_cols = df.select_dtypes(include=[np.number])
# 2. замена пропущенных значений на среднее
df[num_cols.columns] = num_cols.fillna(num_cols.mean())
#print(df[num_cols.columns])

# Всем текстовым значениям присвоим значения моды
# 1. Выберем все категориальные колонки
categorical_cols = df.select_dtypes(include=['object'])
# 2. Заменим прорущенные значений на моду
df[categorical_cols.columns] = categorical_cols.fillna(categorical_cols.mode().iloc[0])
print(df[categorical_cols.columns])

# Проверка. Не должно быть отсутствующих значений (напротив всех столбцов должно стоять 0). Пояснение к коду на стр 50-56
pd.set_option('display.max_rows', None)
print(df.isnull().sum())
pd.reset_option('display.max_rows')

# Удаление дублирующихся строк
df.drop_duplicates(inplace=True)

# Проведите разведочный анализ данных (EDA), ответив на следующие вопросы:
# a. Каково распределение целевой переменной 'SalePrice'? Есть ли какие-либо выбросы?
# b. Исследуйте взаимосвязи между целевой переменной и другими характеристиками. Есть ли сильные корреляции?
# c. Исследуйте распределение и взаимосвязи других важных характеристик, таких как 'OverallQual', 'GrLivArea', 'GarageCars' и т.д.
# d. Визуализируйте данные, используя соответствующие графики (например, гистограммы, диаграммы рассеяния, квадратные диаграммы),
# чтобы получить представление о датасете.

# Гистограмма распределения SalePrice (вывод столбца SalePrice)
plt.figure(figsize=(10,6))
# Построене гистограммы и кривой плотности распределения
sns.histplot(df['SalePrice'], kde=True, color='skyblue')
plt.title('Стоимость недвижимости')
plt.xlabel('Стоимость домов')
plt.ylabel('Количество домов')
# Отображение графика
plt.show()

# Обнаружение и обработка выбросов
z_scores = np.abs(stats.zscore(df['SalePrice']))
# Установка порогового значения Z-score
threshold = 2
# Выявление выбросов на основе Z-score
outliers = df['SalePrice'][z_scores > threshold]
# Среднее значение в столбце
print('Среднее значение')
print(df['SalePrice'].mean())
print('Выбросы')
print(outliers)

# Замена выбрасов медианным значением
df.loc[z_scores > threshold, 'SalePrice'] = df['SalePrice'].median()
# Среднее значение в столбце
print('Среднее значение')
print(df['SalePrice'].mean())
print('Выбросы')
print(outliers)
# Попробуем снизить с помощью мат.фугкции log
# Трансформация столбца "SalePrice" с помощью логарифмической функции
df['SalePrice'] = np.log1p(df['SalePrice'])
print('Среднее значение')
print(df['SalePrice'].mean())

#Исследуйте распределение и взаимосвязи других важных характеристик, таких как 'OverallQual', 'GrLivArea', 'GarageCars' и т.д.$
# Расмотрим поле 'OverallQual'
plt.figure(figsize=(10,6))
sns.histplot(df['OverallQual'], kde=True, color='skyblue')
plt.title('Отделка')
plt.xlabel('Количество домов')
plt.ylabel('Отделка')
plt.show()

# Обнаружение и обработка выбросов. Доведем до такого состояния, чтобы не осталось выбросов
z_scores = np.abs(stats.zscore(df['OverallQual']))
threshold = 2
outliers = df['OverallQual'][z_scores > threshold]
print('Среднее значение')
print(df['OverallQual'].mean())
print('Выбросы')
print(outliers)
# Попробуем снизить с помощью мат.функции log
df['OverallQual'] = np.log1p(df['OverallQual'])
print('Среднее значение')
print(df['OverallQual'].mean())
print('Выбросы')
print(outliers)
# Пробуем уйти от выбросов с помощью замены на среднее значение в столбце
df.loc[z_scores > threshold, 'OverallQual'] = df['OverallQual'].median()
print('Среднее значение')
print(df['OverallQual'].mean())
print('Выбросы')
# После всех манипуляций выбросов в столбце OverallQual не осталось

# Перейдем к рассмотрению GrLivArea
plt.figure(figsize=(10,6))
sns.histplot(df['GrLivArea'], kde=True, color='skyblue')
plt.title('Количество жилья')
plt.xlabel('Количество домов')
plt.ylabel('Количество квадратов')
plt.show()

# Обнаружение и обработка выбросов. 
z_scores = np.abs(stats.zscore(df['GrLivArea']))
threshold = 2
outliers = df['GrLivArea'][z_scores > threshold]
print('Среднее значение')
print(df['GrLivArea'].mean())
print('Выбросы')
print(outliers)
# Попробуем снизить с помощью мат.функции log
df['GrLivArea'] = np.log1p(df['GrLivArea'])
print('Среднее значение')
print(df['GrLivArea'].mean())
print('Выбросы')
print(outliers)

# Перейдем к рассмотрению GarageCars
plt.figure(figsize=(10,6))
sns.histplot(df['GarageCars'], kde=True, color='skyblue')
plt.title('Garage Cars')
plt.xlabel('Количество машин')
plt.ylabel('Количество недвижимости')
plt.show()

# Обнаружение и обработка выбросов. 
z_scores = np.abs(stats.zscore(df['GarageCars']))
threshold = 3
outliers = df['GarageCars'][z_scores > threshold]
print('Среднее значение')
print(df['GarageCars'].mean())
print('Выбросы')
print(outliers)
# Выбросов нет. 

# Визуализация. Мне лень делать несколько графиков. Сделаю один корреляционный
correlation = df[['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars']].corr()
correlation = np.round(correlation, 1)
correlation[np.abs(correlation)<0.3] = 0
print(correlation)
plt.figure(figsize=(10,6))
sns.heatmap(correlation, annot=True, linewidths=0.5, cmap='viridis')
plt.title('Корреляционный график');
plt.show()

# Справа в окне вывода корреляционного графика видно шкалу корреляции. 
# Из графика видно, что на SalePrice максимально влияют показатели OverallQual и GrLivArea, т.е. можно сделать вывод, что чем лучше качество отделки и больше количество метров жилой 
# площади, тем выше у нас будет цена недвижимости. GarageCars - также влияет на цену недвижимости, но в меньшей степени. Безусловно цена недвижимости растет от наличия места парковки, но
# влиянеие парковочного места на цену намного меньше, чем влияние количества метров 
# Если рассмотривать взаимосвязь показателей между собой, то видно, что их влияние друг на друга незначительное, что логично. Наличие гаража не может влиять на качество отделки или
# количества квадратов, также как качество отделки. Таким образом из графика видно, что показатели OverallQual', 'GrLivArea', 'GarageCars не влияют друг на друга.


# Выполните проектирование признаков путем реализации следующих преобразований:
# a. Работайте с категориальными переменными, применяя one-hot encoding или label encoding, в зависимости от характера переменной.
# b. При необходимости создайте новые характеристики, такие как общая площадь или возраст объекта недвижимости,
# путем объединения существующих характеристик.

# Сохраните очищенный и преобразованный набор данных в новый CSV-файл под названием 'cleaned_house_prices.csv'.


# Стандартизация данных. Ищем среднюю путем  
# Копия датафрейма
df_standardized = df.copy()
df_standardized[num_cols.columns] = (df_standardized[num_cols.columns] - df_standardized[num_cols.columns].mean())/df_standardized[num_cols.columns].std()

#Пример LabelEncodIng(создание под определенный столбец своего столбца со значениями True/False то есть 0/1)
# Создание доп столбца. 
object_car = LabelEncoder()
# Преобразование категорийной переменной в числовую. Столбец Type_Encoded будет иметь значение 0 или 1 в зависимости от значения столбца Type.
# То есть если в столбце Type стоит значение Free,то в Type_Encoded будет значение 0. 
df['Object_Car'] = object_car.fit_transform(df['GarageCars'])

# # Пример OneHotEncodIng (создание под каждый тип своего столбца со значениями True/False)
# # Выбираем столбец, с которым будем работать (columns=["Content Rating"]) и как его будем разбивать (prefix='ContentRating'). drop_first=True - удалили заголовок
df = pd.get_dummies(df, columns=["GarageCars"], prefix='GarageCars_types', drop_first=True)

# # Пример, когда мы берем столбец Category и разбиваем его на Category_types
# df = pd.get_dummies(df, columns=["Category"], prefix='Category_types', drop_first=True)


# # Создание сводной таблцы. Можно этот кусок кода опустить и сразу сохранять в файл.
# piv_table = df.pivot_table(index='Category', columns='ContentRating_Teen', values="Rating", aggfunc = 'mean')
# #print('\n сводная таблица')
# print(piv_table)

# Сохранение 
output_file_path = 'cleaned_house_prices.csv'
df.to_csv(output_file_path, index=False)

# # Сохранение 
# # Появится файл с таблицей, в которой Content Rating разбит на категории по возрастам с использованием значений True/False
# output_file_path = 'clear_gapps_label_encoding.csv'
# df.to_csv(output_file_path, index=False)


# # 
# # ###

