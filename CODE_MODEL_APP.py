import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import streamlit as st


def preprocess_data(df):
    # Добавьте здесь код предварительной обработки данных, если необходимо
    return df


def train_model(df):
    features = ['x', 'y', 'z', 'calculated_speed']
    X = df[features]
    y = df['moving_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    return clf


def predict_movement(model, df):
    features = ['x', 'y', 'z', 'calculated_speed']
    X = df[features]
    prediction = model.predict(X)
    return prediction


def predict_transportation(model, new_data):
    features = ['x', 'y', 'z', 'calculated_speed']
    X = new_data[features]
    prediction = model.predict(X)
    transportation_modes = {
        0: "Автомобиль",
        1: "Автобус",
        2: "Метро",
        3: "Велосипед"
    }
    transportation_mode = transportation_modes.get(prediction[0], "Неизвестно")
    return transportation_mode


def main():
    st.set_page_config(page_title="MoveMate", page_icon=":oncoming_automobile:", layout="wide", initial_sidebar_state="collapsed")
    st.title("MoveMate")
    st.header("Halte alle auf dem Laufenden")
    
    uploaded_file = st.file_uploader("Datei hochladen", type="json")
    
    if uploaded_file is not None:
        df = pd.read_json(uploaded_file)
        df = preprocess_data(df)
        model = train_model(df)
        
        movement_prediction = predict_movement(model, df)
        transportation = predict_transportation(model, df)
    
        st.write(f"Aktivität: {bool(movement_prediction[0])}")
        st.write(f"Tranport type: {transportation}")
    else:
        st.write("Laden Sie bitte Datei.json hoch!")


if __name__ == "__main__":
    main()



"""def main():
    st.set_page_config(page_title="MoveMate", page_icon=":oncoming_automobile:", layout="wide", initial_sidebar_state="collapsed")

    st.title("MoveMate")
    st.header("Halte alle auf dem Laufenden")
    st.write("Um die Funktionen von MoveMate zu nutzen, schauen Sie bitte die Instruktion im linken Menü an!")
    
    st.write("Nach dem Lesen der Instruktion und der Konfiguration von SensorLoggerApp und MoveMateBot in Telegram können wir jetzt loslegen!")
    
    uploaded_file = st.file_uploader("Datei hochladen", type="json")
    if uploaded_file is not None:
        df = pd.read_json(uploaded_file)
        df = preprocess_data(df)
        model = train_model(df)

        st.subheader("Пример данных:")
        st.write(df.head())

        st.subheader("Предсказание способа передвижения:")
        x = st.number_input("Введите значение x", value=0.0)
        y = st.number_input("Введите значение y", value=0.0)
        z = st.number_input("Введите значение z", value=0.0)
        speed = st.number_input("Введите значение скорости", value=0.0)

        new_data = pd.DataFrame({'x': [x], 'y': [y], 'z': [z], 'calculated_speed': [speed]})
        prediction = predict_transportation(model, new_data)
        st.write("Результат предсказания:", prediction)


if __name__ == "__main__":
    main()"""