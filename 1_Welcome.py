#importing necessery libraries for future analysis of the dataset
#import folium
import pandas as pd
#import matplotlib.pyplot as plt
#import matplotlib as plt
#import seaborn as sns
#import json
import numpy as np
import streamlit as st
#from streamlit_folium import st_folium
#import plotly.express as px
import torch

model_tree = 'Tree_Model.pkl'
model = torch.load(model_tree)

from collections import Counter
 
# Function to get unique values
 
 
def unique(list1):
 
    # Print directly by using * symbol
    print(*Counter(list1))
#Functions 

#Tranform accelerometer and gyroscope data to one dataframe
def transform_data_acceleration(df):   
    acce = df[df['sensor'] == 'Accelerometer']
    acce.reset_index(drop=True, inplace=True)   
    acce = acce.drop(columns =['seconds_elapsed','sensor', 'relativeAltitude', 'pressure', 'altitude', 'speedAccuracy', 'bearingAccuracy', 'latitude', 'altitudeAboveMeanSeaLevel', 'bearing', 'horizontalAccuracy', 'verticalAccuracy', 'longitude', 'speed', 'version', 'device name', 'recording time', 'platform', 'appVersion', 'device id', 'sensors', 'sampleRateMs', 'yaw', 'qx', 'qz', 'roll', 'qw', 'qy', 'pitch'])
    acce['Magnitude_acce'] = np.sqrt(acce["x"] ** 2 + acce["y"] ** 2 + acce["z"] ** 2)
    
    gyro = df[df['sensor'] == 'Gyroscope']
    gyro.reset_index(drop=True, inplace=True)   
    gyro = gyro.drop(columns = ['seconds_elapsed','sensor', 'relativeAltitude', 'pressure', 'altitude', 'speedAccuracy', 'bearingAccuracy', 'latitude', 'altitudeAboveMeanSeaLevel', 'bearing', 'horizontalAccuracy', 'verticalAccuracy', 'longitude', 'speed', 'version', 'device name', 'recording time', 'platform', 'appVersion', 'device id', 'sensors', 'sampleRateMs', 'yaw', 'qx', 'qz', 'roll', 'qw', 'qy', 'pitch'])

    for df in [gyro, acce]:
         df.index = pd.to_datetime(df['time'], unit = 'ns',errors='ignore')
         df.drop(columns=['time'], inplace=True)
    #df_new = pd.merge(loc, gyro, suffixes=('_loc', '_gyro'), on='time')
    df_new = acce.join(gyro, lsuffix = '_acce', rsuffix = '_gyro', how = 'outer').interpolate()
    
    min_values = df_new.resample('15s').min(numeric_only=True)
    max_values = df_new.resample('15s').max(numeric_only=True)
    mean_values = df_new.resample('15s').mean(numeric_only=True)
    std_values = df_new.resample('15s').std(numeric_only=True)

    columns_to_drop = df_new.columns.difference(['Magnitude_acce', 'x_gyro', 'y_gyro', 'z_gyro'])
    for df in [min_values, max_values, mean_values, std_values]:
        df.drop(columns=columns_to_drop, inplace=True)
    feature_df = pd.merge(pd.merge(min_values, max_values, suffixes = ('_min', '_max'), on = 'time'), pd.merge(mean_values, std_values, suffixes = ('_mean', '_std'), on = 'time'), on = 'time')
    prediction= model.predict(feature_df)
    result = unique(prediction)
    return result


#Tranform location from file
def transform_data_location(df):
    location = df[df['sensor'] == 'Location']
    location.reset_index(drop=True, inplace=True)
    location = location.drop(columns = ['sensor', 'z', 'y', 'x', 'relativeAltitude', 'pressure', 'version', 
                                        'device name', 'recording time', 'platform', 'appVersion', 'device id', 'sensors', 'sampleRateMs', 'yaw', 'qx', 'qz', 'roll', 'qw', 'qy', 'pitch'])
    #Speed using abs to positive
    
    location.index = pd.to_datetime(location['time'], unit = 'ns',errors='ignore')
    location.drop(columns=['time'], inplace=True)
    #location['Type'] = type
    return location
 
### Streamlit Area
### Page config
st.set_page_config(page_title="Mobility Classification App", page_icon=":oncoming_automobile:", layout="wide", initial_sidebar_state="collapsed")

st.header("Dein Standort wurde gefunden")
with st.container():
    st.write("---")
    #st.subheader("Du bist gerade auf...")
    st.write("Fügen Sie bitte ihre Datei hinzu")
    def main():
        file = st.file_uploader("Laden Sie eine JSON Datei hoch!", type = ['json'])
        if file is not None:
            uploaded_file = pd.read_json(file)
            if st.button("Classify me!"):
                prediction =  transform_data_acceleration(uploaded_file)
                location_data = transform_data_location(uploaded_file)

                st.subheader("Dein Fortbewegungsgraph")
                st.map(location_data)
            
                st.caption("Du bist gerade auf dem " + str(prediction) + "!")
                
    if __name__ == "__main__":
        main()
    st.write("---")
