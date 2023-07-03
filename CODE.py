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
#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score
#import tensorflow as tf
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler

#using pandas library and 'read_json' function to read file as file 
bus_1= pd.read_json("G:\Мой диск\ML4B\Bus\Bus_Bauvereinstr.-Technische Hochschule-Dürrenhof.23-05-23_18-29-17.json")
bus_2= pd.read_json("G:\Мой диск\ML4B\Bus\Bus_Harmoniestr.-Wöhrd.23-05-23_18-27-14.json")
bus_3= pd.read_json("G:\Мой диск\ML4B\Bus\Bus_Laufer Tor-Rathenauplatz.23-05-23_18-24-45.json")
bus_4= pd.read_json("G:\Мой диск\ML4B\Bus\Bus_Rathenauplatz-Harmoniestr.23-05-23_18-25-46.json")
bus_5= pd.read_json("G:\Мой диск\ML4B\Bus\Bus_Wöhrd-Bauvereinstr.23-05-23_18-28-17.json")
bus_6= pd.read_json("Bus_Laufer Tor-Rathenauplatz.23-05-23_18-24-45.json")
bus_7= pd.read_json("Bus_Harmoniestr.-Wöhrd.23-05-23_18-27-14.json")
fahrrad_1 = pd.read_json("Fahrraddata1.json")
fahrrad_2 = pd.read_json("Fahrraddata2.json")
fahrrad_3 = pd.read_json("Fahrraddata3.json")
fahrrad_4 = pd.read_json("Fahrraddata4.json")
fahrrad_5 = pd.read_json("Fahrraddata6.json")

#Create dataframes for each type of transport
bf_df = [bus_1, bus_2, bus_3, bus_4, bus_5, bus_6, bus_7, fahrrad_1, fahrrad_2, fahrrad_3, fahrrad_4, fahrrad_5]
bf_df  = pd.concat(bf_df,ignore_index=True)

# Проверка наличия пропущенных значений в DataFrame
print(bf_df.isnull().sum())

bf_df = bf_df.dropna(subset=['time', 'seconds_elapsed'])
bf_df['z'].fillna(bf_df['z'].mean(), inplace=True)
bf_df['y'].fillna(bf_df['y'].mean(), inplace=True)
bf_df['x'].fillna(bf_df['x'].mean(), inplace=True)

#Drop collumns
bf_df = bf_df.drop(['version','device name','recording time','platform','appVersion','device id', 'sensors','sampleRateMs','qz','qy','qx','qw','roll','pitch','yaw','relativeAltitude','pressure', 'altitudeAboveMeanSeaLevel'], axis=1)
bf_df = bf_df.drop(['bearingAccuracy', 'speedAccuracy', 'verticalAccuracy', 'horizontalAccuracy', 'speed', 'bearing', 'altitude', 'longitude', 'latitude'], axis=1)
#fahrrad_df = fahrrad_df.drop(['version','device name','recording time','platform','appVersion','device id', 'sensors','sampleRateMs','qz','qy','qx','qw','roll','pitch','yaw','relativeAltitude','pressure', 'altitudeAboveMeanSeaLevel'], axis=1)
#fahrrad_df

# Начальное значение скорости
initial_velocity = 0.0

# Преобразование столбцов в числовой формат
bf_df['seconds_elapsed'] = pd.to_numeric(bf_df['seconds_elapsed'], errors='coerce')
bf_df['z'] = pd.to_numeric(bf_df['z'], errors='coerce')

# Рассчитываем скорость на основе ускорения с помощью метода численного интегрирования (например, метод трапеций)
bf_df['calculated_speed'] = initial_velocity + bf_df['seconds_elapsed'].diff() * bf_df['z']
bf_df['calculated_speed'].fillna(bf_df['calculated_speed'].mean(), inplace=True)

# В результате получим столбец 'calculated_speed', который содержит скорость транспортного средства на основе ускорения

print(bf_df['calculated_speed'])

print(bf_df.isnull().sum())

# Преобразование значения времени из наносекунд в datetime
bf_df['time'] = pd.to_datetime(bf_df['time'], unit='ns')

# Преобразование времени в 24-часовой формат
bf_df['time'] = bf_df['time'].dt.strftime('%H:%M:%S')
time_data = bf_df['time']
print(time_data)

# Пороговое значение для определения движения
threshold_acceleration = 0.1

# Создаем новый столбец, где будем хранить информацию о движении (True - движение, False - остановка)
bf_df['moving'] = bf_df[['x', 'y', 'z']].apply(lambda row: any(abs(row) > threshold_acceleration), axis=1)

# В результате получим столбец 'moving', который содержит True или False в зависимости от состояния движения
print(bf_df['moving'])

# Создаем новый столбец, где будем хранить информацию о движении (1 - движение, 0 - остановка)
bf_df['moving_label'] = bf_df['moving'].astype(int)

# Выбираем признаки (features) для обучения модели
features = ['x', 'y', 'z', 'calculated_speed']
X = bf_df[features]
y = bf_df['moving_label']

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

# Streamlit приложение
def main():
    st.title("RoadFocus")
    st.write("Прогнозирует состояние движения на основе данных ускорения")
    # Добавьте интерактивные элементы (например, текстовые поля или слайдеры), чтобы пользователь мог вводить данные
    x_input = st.number_input("Введите значение x:")
    y_input = st.number_input("Введите значение y:")
    z_input = st.number_input("Введите значение z:")
    speed_input = st.number_input("Введите значение скорости:")
    # Используйте обученную модель для предсказания состояния движения на основе введенных пользователем данных
    new_data = pd.DataFrame({'x': [x_input], 'y': [y_input], 'z': [z_input], 'calculated_speed': [speed_input]})
    prediction = clf.predict(new_data)
    # Отображение результата предсказания
    if prediction[0] == 1:
        st.write("Текущее состояние: Движение")
    else:
        st.write("Текущее состояние: Остановка")
# Run the Streamlit app
if __name__ == '__main__':
    main()