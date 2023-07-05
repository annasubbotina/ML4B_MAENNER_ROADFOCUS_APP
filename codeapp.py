import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import streamlit as st


def main():
    st.set_page_config(page_title="MoveMate", page_icon=":oncoming_automobile:", layout="wide", initial_sidebar_state="collapsed")

    st.title("MoveMate")
    st.header("Halte alle auf dem Laufenden")
    st.write("Um die Funktionen von MoveMate zu nutzen, schauen Sie bitte die Instruktion im linken Menü an!")
    
    st.write("Nach dem Lesen der Instruktion und der Konfiguration von MoveMateBot in Telegram können wir jetzt loslegen!")
    
    uploaded_file = st.file_uploader("Datei hochladen", type="json")
    
    if uploaded_file is not None:
        df = pd.read_json(uploaded_file)
        df = preprocess_data(df)
        model = train_model(df)
        return model,df
    else:
        st.write("Laden Sie bitte Datei.json hoch!")

    movement_prediction = predict_movement(model, df)# Make the prediction
    transportation = predict_transportation(model, df)
    

    # Display the result
    st.write(f"Aktivität: {bool(movement_prediction[0])}")
    st.write(f"Tranport type: {transportation}")

if __name__ == "__main__":
    
    main()
