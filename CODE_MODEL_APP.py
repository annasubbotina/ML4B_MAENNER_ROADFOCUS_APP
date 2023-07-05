import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

def load_data():
    bus_1 = pd.read_json("Bus1_Bauvereinstr.-Technische Hochschule-Dürrenhof.23-05-23_18-29-17.json")
    bus_2 = pd.read_json("Bus2_Dürrenhof-Stephanstr.23-05-23_18-31-17.json")
    fahrrad_1 = pd.read.json("Fahrraddata2.json")
    fahrrad_2 = pd.read.json("Fahrraddata3.json")
    ubahn_1 = pd.read.json("uBahn_Kaulbachplatz_-_Hbf-2023-05-24_06-09-08.json")
    ubahn_2 = pd.read.json("Ubahn_Wöhrder Wiese-Hbf-Opernhaus.24-05-23_13-33-39.json")
    # Add other datasets as needed
    bf_df = pd.concat([bus_1, bus_2, fahrrad_1, fahrrad_2, ubahn_1, ubahn_2], ignore_index=True)
    return bf_df

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
        0: "Auto",
        1: "Bus",
        2: "U-Bahn",
        3: "Fahrrad"
    }
    transportation_mode = transportation_modes.get(prediction[0], "Неизвестно")
    return transportation_mode


def main():
    st.set_page_config(page_title="MoveMate", page_icon=":oncoming_automobile:", layout="wide", initial_sidebar_state="collapsed")

    #logo_image = "<img src='logo.jpg' style='float: right;'>"
    #st.markdown(logo_image, unsafe_allow_html=True)

    # Загрузка и отображение изображения по ссылке
    image_url = "https://drive.google.com/file/d/1weZ5iaPsOa8tjGEm-9v6UopAzigm9h2e/view?usp=sharing"
    st.image(image_url, caption="Загруженное", use_column_width=True)

    st.title("MoveMate")
    st.header("Halte alle auf dem Laufenden")

    uploaded_file = st.file_uploader("Datei hochladen", type="json")

    if uploaded_file is not None:
        user_df = pd.read_json(uploaded_file)
        user_df = preprocess_data(user_df) 
        model = train_model(user_df)
        transportation = predict_transportation(model, user_df)
        
        st.write(f"Tranport type: {transportation}")
    else:
        st.write("Laden Sie bitte Datei.json hoch!")

if __name__ == "__main__":
    main()



#def main():
    #st.set_page_config(page_title="MoveMate", page_icon=":oncoming_automobile:", layout="wide", initial_sidebar_state="collapsed")

    #st.title("MoveMate")
    ##st.header("Halte alle auf dem Laufenden")
    #st.write("Um die Funktionen von MoveMate zu nutzen, schauen Sie bitte die Instruktion im linken Menü an!")
    
    #st.write("Nach dem Lesen der Instruktion und der Konfiguration von SensorLoggerApp und MoveMateBot in Telegram können wir jetzt loslegen!")
    
    #uploaded_file = st.file_uploader("Datei hochladen", type="json")
    #if uploaded_file is not None:
        #df = pd.read_json(uploaded_file)
        #df = preprocess_data(df)
        #model = train_model(df)

        #st.subheader("Пример данных:")
        #st.write(df.head())

        #st.subheader("Предсказание способа передвижения:")
        #x = st.number_input("Введите значение x", value=0.0)
        #y = st.number_input("Введите значение y", value=0.0)
        #z = st.number_input("Введите значение z", value=0.0)
        #speed = st.number_input("Введите значение скорости", value=0.0)

        #new_data = pd.DataFrame({'x': [x], 'y': [y], 'z': [z], 'calculated_speed': [speed]})
        #prediction = predict_transportation(model, new_data)
        #st.write("Результат предсказания:", prediction)


#if __name__ == "__main__":
    #main()