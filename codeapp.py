#importing necessary libraries for future analysis of the dataset
import folium
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import time
import numpy as np
import streamlit as st
from streamlit_folium import st_folium
import plotly.express as px
import torch
from PIL import Image
from streamlit_lottie import st_lottie
import requests

model_tree = pickle.load(open('Tree_Model.pkl', 'rb'))
model = torch.load(model_tree)

#Functions 

#Transform accelerometer and gyroscope data to one dataframe
def transform_data_acceleration(file):
    df = pd.read_json(file)
    acce = df[df['sensor'] == 'Accelerometer']
    acce.reset_index(drop=True, inplace=True)
    acce = acce.drop(columns=['seconds_elapsed','sensor', 'relativeAltitude', 'pressure', 'altitude', 'speedAccuracy', 'bearingAccuracy', 'latitude', 'altitudeAboveMeanSeaLevel', 'bearing', 'horizontalAccuracy', 'verticalAccuracy', 'longitude', 'speed', 'version', 'device name', 'recording time', 'platform', 'appVersion', 'device id', 'sensors', 'sampleRateMs', 'yaw', 'qx', 'qz', 'roll', 'qw', 'qy', 'pitch'])
    acce['Magnitude_acce'] = np.sqrt(acce["x"] ** 2 + acce["y"] ** 2 + acce["z"] ** 2)
    
    gyro = df[df['sensor'] == 'Gyroscope']
    gyro.reset_index(drop=True, inplace=True)
    gyro = gyro.drop(columns=['seconds_elapsed','sensor', 'relativeAltitude', 'pressure', 'altitude', 'speedAccuracy', 'bearingAccuracy', 'latitude', 'altitudeAboveMeanSeaLevel', 'bearing', 'horizontalAccuracy', 'verticalAccuracy', 'longitude', 'speed', 'version', 'device name', 'recording time', 'platform', 'appVersion', 'device id', 'sensors', 'sampleRateMs', 'yaw', 'qx', 'qz', 'roll', 'qw', 'qy', 'pitch'])
    
    for df in [gyro, acce]:
        df.index = pd.to_datetime(df['time'], unit='ns')
        df.drop(columns=['time'], inplace=True)
    
    df_new = acce.join(gyro, lsuffix='_acce', rsuffix='_gyro', how='outer').interpolate()
    return df_new

#Transform location from file
def transform_data_location(file, format):
    if format == 'json':
        df = pd.read_json(file)
    else:
        df = pd.read_csv(file)
    
    location = df[df['sensor'] == 'Location']
    location.reset_index(drop=True, inplace=True)
    location = location.drop(columns=['sensor', 'z', 'y', 'x', 'relativeAltitude', 'pressure', 'version', 'device name', 'recording time', 'platform', 'appVersion', 'device id', 'sensors', 'sampleRateMs', 'yaw', 'qx', 'qz', 'roll', 'qw', 'qy', 'pitch'])
    
    location.index = pd.to_datetime(location['time'], unit='ns')
    location.drop(columns=['time'], inplace=True)
    return location

#Cut data into windows of 5 seconds and calculate min, max, mean and std
def create_feature_df(df):
    min_values = df.resample('5s').min(numeric_only=True)
    max_values = df.resample('5s').max(numeric_only=True)
    mean_values = df.resample('5s').mean(numeric_only=True)
    std_values = df.resample('5s').std(numeric_only=True)
    columns_to_drop = df.columns.difference(['Magnitude_acce', 'x_acce', 'x_gyro', 'y_acce', 'y_gyro', 'z_acce', 'z_gyro', 'x', 'y', 'z'])
    
    for df in [min_values, max_values, mean_values, std_values]:
        df.drop(columns=columns_to_drop, inplace=True)
    
    feature_df = pd.merge(pd.merge(min_values, max_values, suffixes=('_min', '_max'), on='time'), pd.merge(mean_values, std_values, suffixes=('_mean', '_std'), on='time'), on='time')
    return feature_df

#Process data for prediction
def process_data_prediction(df):
    df = transform_data_acceleration(df)
    df_prep = create_feature_df(df)
    return df_prep

def process_data_location(df):
    df = transform_data_location(df)
    return df

#Map data 
def map_data(df):
    coords = [(row.latitude, row.longitude) for _, row in df.iterrows()]
    my_map = folium.Map(location=[df.latitude.mean(), df.longitude.mean()], zoom_start=16)
    folium.PolyLine(coords, color="blue", weight=5.0).add_to(my_map)
    return my_map

#read data from url
#@st.experimental_memo
#def get_data() -> pd.DataFrame:
#    url = "http://10.100.213.5:8000/data"
#    return pd.read_json(url)

#Streamlit App code below

def main():
    st.set_page_config(page_title="MoveMate", page_icon=":oncoming_automobile:", layout="wide", initial_sidebar_state="collapsed")

    def load_lottieurl(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    
    logo = Image.open("logo.jpg")

    st.sidebar.success("Menu")

    with st.container():
        text_column, image_column = st.columns((2,1))
        with text_column:
            st.title("MoveMate")
            st.header("Keep everyone on track")
            st.markdown("See our GitHub Repository ⇒ (https://github.com/or81ynez/MaennerML)")
        with image_column:
            st.image(logo)

    with st.container():
        st.write("---")
        st.write("With our MoveMateApp you will be able to determine the type of transport on which you move.")
        st.write("But in order to get started we advise you to read the Instructions section on the left in the menu!")
    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        lottie_coding = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_xbf1be8x.json")
        with left_column:
            st.write("We hope you managed to record your movement data. Let's try to determine the type of your transport!")
            
            uploaded_file = st.file_uploader("Drop it here ⇓", accept_multiple_files=False)

            if uploaded_file is not None:
                prediction_data = process_data_prediction(uploaded_file)
                location_data = process_data_location(uploaded_file)
                st.subheader("Your travel graph")
                map_data(location_data)
                tree_predictions = model.predict(prediction_data)
                st.caption("You are using " + str(tree_predictions) + "!")
            
            else:
                st.write("Upload a JSON file!")
        with right_column:
                st_lottie(lottie_coding, height=300, key="coding")
        

if __name__ == "__main__":
    main()
