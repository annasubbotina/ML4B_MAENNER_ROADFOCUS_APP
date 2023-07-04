#importing necessery libraries for future analysis of the dataset
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as plt
import seaborn as sns
#!pip install tensorflow shap
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

#using pandas library and 'read_json' function to read file as file 
bus_1= pd.read_json("Bus1_Bauvereinstr.-Technische Hochschule-Dürrenhof.23-05-23_18-29-17.json")
bus_2= pd.read_json("Bus2_Dürrenhof-Stephanstr.23-05-23_18-31-17.json")
bus_3= pd.read_json("bus-_Gleitwitzertr._-_Langwasser_Mitte-2023-05-24_12-06-59 2.json")
bus_4= pd.read_json("Bus_Wöhrd-Bauvereinstr.23-05-23_18-28-17.json")
bus_5= pd.read_json("Bus_Rathenauplatz-Harmoniestr.23-05-23_18-25-46.json")
bus_6= pd.read_json("Bus_Laufer Tor-Rathenauplatz.23-05-23_18-24-45.json")
bus_7= pd.read_json("Bus_Harmoniestr.-Wöhrd.23-05-23_18-27-14.json")

bus_df = [bus_1, bus_2, bus_3, bus_4, bus_5, bus_6, bus_7]
bus_df  = pd.concat(bus_df,ignore_index=True)

# Проверка наличия пропущенных значений в DataFrame
print(bus_df.isnull().sum())

bus_df = bus_df.dropna(subset=['time', 'seconds_elapsed'])
bus_df['z'].fillna(bus_df['z'].mean(), inplace=True)
bus_df['y'].fillna(bus_df['y'].mean(), inplace=True)
bus_df['x'].fillna(bus_df['x'].mean(), inplace=True)

#Drop collumns
bus_df = bus_df.drop(['version','device name','recording time','platform','appVersion','device id', 'sensors','sampleRateMs','qz','qy','qx','qw','roll','pitch','yaw','relativeAltitude','pressure', 'altitudeAboveMeanSeaLevel'], axis=1)
bus_df = bus_df.drop(['bearingAccuracy', 'speedAccuracy', 'verticalAccuracy', 'horizontalAccuracy', 'speed', 'bearing', 'altitude', 'longitude', 'latitude'], axis=1)

print(bus_df.isnull().sum())

# Начальное значение скорости
initial_velocity = 0.0

# Преобразование столбцов в числовой формат
bus_df['seconds_elapsed'] = pd.to_numeric(bus_df['seconds_elapsed'], errors='coerce')
bus_df['z'] = pd.to_numeric(bus_df['z'], errors='coerce')

# Рассчитываем скорость на основе ускорения с помощью метода численного интегрирования (например, метод трапеций)
bus_df['calculated_speed'] = initial_velocity + bus_df['seconds_elapsed'].diff() * bus_df['z']
bus_df.loc[:, 'calculated_speed'].fillna(bus_df['calculated_speed'].mean(), inplace=True)


# В результате получим столбец 'calculated_speed', который содержит скорость транспортного средства на основе ускорения

print(bus_df['calculated_speed'])

print(bus_df.isnull().sum())

# Пороговое значение для определения движения
threshold_acceleration = 0.1

# Создаем новый столбец, где будем хранить информацию о движении (True - движение, False - остановка)
bus_df['moving'] = bus_df[['x', 'y', 'z']].apply(lambda row: any(abs(row) > threshold_acceleration), axis=1)

# В результате получим столбец 'moving', который содержит True или False в зависимости от состояния движения
print(bus_df['moving'])

# Создаем новый столбец, где будем хранить информацию о движении (1 - движение, 0 - остановка)
bus_df['moving_label'] = bus_df['moving'].astype(int)

# Выбираем признаки (features) для обучения модели
features = ['x', 'y', 'z', 'calculated_speed']
X = bus_df[features]
y = bus_df['moving_label']

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем и обучаем модель Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Предсказываем состояние движения на тестовой выборке
y_pred = clf.predict(X_test)

# Оцениваем точность модели
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Теперь, используя обученную модель, вы можете предсказывать состояние движения на новых данных.
new_data = pd.DataFrame({'x': [0.1], 'y': [0.2], 'z': [0.3], 'calculated_speed': [10.0]})
prediction = clf.predict(new_data)
print("Prediction:", prediction)

