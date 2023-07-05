import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

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

def handle_message(update, context):
    bot = context.bot
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

def main():
    # Set up Telegram bot
    token = "YOUR_TELEGRAM_BOT_TOKEN"
    updater = Updater(token, use_context=True)
    dispatcher = updater.dispatcher
    dispatcher.add_handler(MessageHandler(Filters.text, handle_message))

    updater.start_polling()
    st.write("Telegram бот запущен. Ожидание сообщений...")

if __name__ == "__main__":
    main()
