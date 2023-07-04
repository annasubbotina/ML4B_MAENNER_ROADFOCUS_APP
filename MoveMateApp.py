import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
from telegram import Bot, Update, ReplyKeyboardMarkup
from telegram.ext import Updater, CommandHandler
import requests

def load_data():
    bus_1 = pd.read_json("Bus1_Bauvereinstr.-Technische Hochschule-Dürrenhof.23-05-23_18-29-17.json")
    bus_2 = pd.read_json("Bus2_Dürrenhof-Stephanstr.23-05-23_18-31-17.json")
    # Add other datasets as needed
    bus_df = pd.concat([bus_1, bus_2], ignore_index=True)
    return bus_df

def preprocess_data(df):
    # Preprocess data here (remove null values, drop unnecessary columns, calculate speed, etc.)
    df = df.dropna(subset=['time', 'seconds_elapsed'])
    df['z'].fillna(df['z'].mean(), inplace=True)
    df['y'].fillna(df['y'].mean(), inplace=True)
    df['x'].fillna(df['x'].mean(), inplace=True)

    initial_velocity = 0.0
    df['seconds_elapsed'] = pd.to_numeric(df['seconds_elapsed'], errors='coerce')
    df['z'] = pd.to_numeric(df['z'], errors='coerce')
    df['calculated_speed'] = initial_velocity + df['seconds_elapsed'].diff() * df['z']
    df.loc[:, 'calculated_speed'].fillna(df['calculated_speed'].mean(), inplace=True)


    threshold_acceleration = 0.1
    df['moving'] = df[['x', 'y', 'z']].apply(lambda row: any(abs(row) > threshold_acceleration), axis=1)
    df['moving_label'] = df['moving'].astype(int)

    return df

def train_model(df):
    features = ['x', 'y', 'z', 'calculated_speed']
    X = df[features]
    y = df['moving_label']

    # Train the model
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    return clf

def predict_movement(model, new_data):
    prediction = model.predict(new_data)
    return prediction

def predict_transportation(model, new_data):
    prediction = model.predict(new_data)
    transportation_modes = {
        0: "Автомобиль",
        1: "Автобус",
        2: "Метро",
        3: "Велосипед"
    }
    transportation_mode = transportation_modes.get(prediction[0], "Неизвестно")
    return transportation_mode


def load_data():
    response = requests.get(url, timeout=10)  # Установка времени ожидания в 10 секунд

    data = response.json()
    df = pd.DataFrame(data)
    return df

def main():
    st.set_page_config(page_title="MoveMate", page_icon=":oncoming_automobile:", layout="wide", initial_sidebar_state="collapsed")

    st.title("MoveMate")
    st.header("Halte alle auf dem Laufenden")
    st.write("Um die Funktionen von MoveMate zu nutzen, schauen Sie bitte die Instruktion im linken Menü an!")
    
    st.write("Nach dem Lesen der Instruktion und der Konfiguration von SensorLoggerApp und MoveMateBot in Telegram können wir jetzt loslegen!")
    #st.write("Fügen Sie das kopierte http aus der SensorLogApp hier ein:")
    # Виджет для ввода HTTP-адреса
    #url = st.text_input("Введите HTTP-адрес для получения данных")

    # Load data from SensorLogApp
    data = load_data()

    # Preprocess the data
    data = preprocess_data(data)

    # Train the model
    model = train_model(data)

    # Display a sample of the data
    st.subheader("Пример данных:")
    st.write(data.head())

    st.subheader("Предсказание способа передвижения:")
    
    # Collect input data from the user
    x = st.number_input("Введите значение x:", value=0.1)
    y = st.number_input("Введите значение y:", value=0.2)
    z = st.number_input("Введите значение z:", value=0.3)
    calculated_speed = st.number_input("Введите рассчитанную скорость:", value=10.0)

    # Create a DataFrame with the user's input
    new_data = pd.DataFrame({'x': [x], 'y': [y], 'z': [z], 'calculated_speed': [calculated_speed]})

    # Make the prediction
    movement_prediction = predict_movement(model, new_data)
    transportation = predict_transportation(model, new_data)

    # Display the result
    st.write(f"Движение: {bool(movement_prediction[0])}")
    st.write(f"Способ передвижения пользователя: {transportation}")

if __name__ == "__main__":
    url = "http://192.168.1.99:8000/data"  # Здесь указываете нужный URL
    main()
