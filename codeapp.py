import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
from telegram import Bot, Update, ReplyKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import requests


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


def handle_message(bot, update):
    message = update.message.text

    if message.startswith('/start'):
        bot.send_message(chat_id=update.message.chat_id, text="Привет! Я бот MoveMate. Отправь мне свои данные о передвижении, и я скажу тебе, на чем ты едешь!")

    elif message.startswith('/predict'):
        user_id = update.message.from_user.id
        df = load_user_data(user_id)
        if df is not None:
            model = train_model(df)
            prediction = predict_transportation(model, df)
            bot.send_message(chat_id=update.message.chat_id, text=f"Ты едешь на: {prediction}")
        else:
            bot.send_message(chat_id=update.message.chat_id, text="У меня нет данных о твоем передвижении.")

    else:
        bot.send_message(chat_id=update.message.chat_id, text="Извините, я не понимаю ваш запрос.")


def load_user_data(user_id):
    # Load user data from file or database based on user_id
    # Return the data as a DataFrame
    # This is just a placeholder function, you should implement your own logic here
    # For example, you can store user data in separate files based on user_id
    data_file = f"{user_id}.json"
    try:
        df = pd.read_json(data_file)
        return df
    except FileNotFoundError:
        return None


def main():
    st.title("MoveMate - Предсказание способа передвижения")
    st.write("Введите данные о вашем передвижении и мы предскажем, на чем вы едете!")

    file = st.file_uploader("Загрузите файл с данными", type="json")
    if file is not None:
        try:
            df = pd.read_json(file)
            df = preprocess_data(df)
            model = train_model(df)
            st.write("Данные успешно обработаны и модель обучена.")
        except pd.errors.JSONDecodeError:
            st.write("Ошибка: Неверный формат файла. Пожалуйста, загрузите JSON-файл.")

    x = st.number_input("Введите значение x", value=0.0)
    y = st.number_input("Введите значение y", value=0.0)
    z = st.number_input("Введите значение z", value=0.0)
    speed = st.number_input("Введите значение скорости", value=0.0)

    new_data = pd.DataFrame({'x': [x], 'y': [y], 'z': [z], 'calculated_speed': [speed]})
    prediction = predict_transportation(model, new_data)
    st.write("Результат предсказания:", prediction)


if __name__ == "__main__":
    # Set up Telegram bot
    token = "6318451790:AAF_qeJbT98s9L6V0hs6lAsxxycVg5W0y8k"
    updater = Updater(token, use_context=True)
    dispatcher = updater.dispatcher
    dispatcher.add_handler(MessageHandler(Filters.text, handle_message))

    updater.start_polling()
    st.write("Telegram бот запущен. Ожидание сообщений...")

    main()
