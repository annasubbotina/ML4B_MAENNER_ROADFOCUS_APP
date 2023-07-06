import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
from PIL import Image
from streamlit_lottie import st_lottie
import requests

@st.cache_data

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
    transportation_mode = transportation_modes.get(prediction[0], "Unbekannt")
    return transportation_mode

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
            st.markdown("[See our GitHub Repository -->] (https://github.com/or81ynez/MaennerML)")
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
            uploaded_file = st.file_uploader("Datei hochladen", type="json")
            if uploaded_file is not None:
                user_df = pd.read_json(uploaded_file)
                user_df = preprocess_data(user_df) 
                model = train_model(user_df)
                transportation = predict_transportation(model, user_df)
                st.write(f"Tranport type: {transportation}")
            else:
                st.write("Laden Sie bitte Datei.json hoch!")
        with right_column:
                st_lottie(lottie_coding, height=300, key="coding")
        

if __name__ == "__main__":
    main()







#Tranform accelerometer and gyroscope data to one dataframe
def transform_data_acceleration(file, format):
    if format == 'json':
        df = pd.read_json(file)
    else:
        df = pd.read_csv(file)  
        
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
   
    #df_new = pd.merge(pd.merge(loc, gyro, suffixes=('_loc', '_gyro'), on='time'), acce, suffixes=('', '_acce'), on='time')
    #df_new['Type'] = type
    
    return df_new

#Tranform location from file
def transform_data_location(file, format):
    if format == 'json':
        df = pd.read_json(file)
    else:
        df = pd.read_csv(file)   

    location = df[df['sensor'] == 'Location']
    location.reset_index(drop=True, inplace=True)
    location = location.drop(columns = ['sensor', 'z', 'y', 'x', 'relativeAltitude', 'pressure', 'version', 
                                        'device name', 'recording time', 'platform', 'appVersion', 'device id', 'sensors', 'sampleRateMs', 'yaw', 'qx', 'qz', 'roll', 'qw', 'qy', 'pitch'])
    #Speed using abs to positive
    
    location.index = pd.to_datetime(location['time'], unit = 'ns',errors='ignore')
    location.drop(columns=['time'], inplace=True)
    #location['Type'] = type
    return location

# Transform accelleration and gyro data for .csv files
def transform_data_accelleration_csv(file, format):
    if format != "csv":
        print("Use other function for .json data")

    acce = pd.read_csv(file)
    acce.reset_index(drop=True, inplace=True) 
    acce['Magnitude_acce'] = np.sqrt(acce["x"] ** 2 + acce["y"] ** 2 + acce["z"] ** 2)
    acce.drop(["x", "y", "z","seconds_elapsed"], axis=1, inplace=True)
    for df in [acce]:
         df.index = pd.to_datetime(df['time'], unit = 'ns',errors='ignore')
         df.drop(columns=['time'], inplace=True)
    return acce

def transform_data_gyroscope_csv(file, format):
    if format != "csv":
        print("Use other function for .json data")
    gyro = pd.read_csv(file)
    gyro.reset_index(drop=True, inplace=True)
    gyro.drop(['seconds_elapsed'], axis=1, inplace=True)
    for df in [gyro]:
         df.index = pd.to_datetime(df['time'], unit = 'ns',errors='ignore')
         df.drop(columns=['time'], inplace=True)
    return gyro

def transform_data_accelleration_auto(file):
    df = pd.read_json(file)
    acce = df[df['sensor'] == 'Accelerometer']
    acce.reset_index(drop=True, inplace=True)   
    acce['Magnitude_acce'] = np.sqrt(acce["x"] ** 2 + acce["y"] ** 2 + acce["z"] ** 2)
    
    gyro = df[df['sensor'] == 'Gyroscope']
    gyro.reset_index(drop=True, inplace=True)   

    for df in [gyro, acce]:
         df.index = pd.to_datetime(df['time'], unit = 'ns',errors='ignore')
         df.drop(columns=['time','seconds_elapsed','sensor'], inplace=True)
    df_new = acce.join(gyro, lsuffix = '_acce', rsuffix = '_gyro', how = 'outer').interpolate()
    return df_new

#Cut data into windows of 5 seconds and calculate min, max, mean and std
def create_feature_df(df, type):   
    min_values = df.resample('15s').min(numeric_only=True)
    max_values = df.resample('15s').max(numeric_only=True)
    mean_values = df.resample('15s').mean(numeric_only=True)
    std_values = df.resample('15s').std(numeric_only=True)
    #columns_to_drop = df.columns.difference(['Magnitude_acce','speed','x_acce', 'x_gyro','y_acce', 'y_gyro', 'z_acce', 'z_gyro','x','y','z'])
    columns_to_drop = df.columns.difference(['Magnitude_acce', 'x_gyro', 'y_gyro', 'z_gyro'])
    for df in [min_values, max_values, mean_values, std_values]:
        df.drop(columns=columns_to_drop, inplace=True)
    feature_df = pd.merge(pd.merge(min_values, max_values, suffixes = ('_min', '_max'), on = 'time'), pd.merge(mean_values, std_values, suffixes = ('_mean', '_std'), on = 'time'), on = 'time')
    feature_df['Type'] = type

    return feature_df

#Combine windows data into one DataFrame (only in case there are more than one df)
def combine_into_df(dfs, type):
    combined_df = pd.concat([create_feature_df(df, type) for df in dfs])  # Apply cut_into_window to each DataFrame and concatenate them
    #combined_df.reset_index(drop=True, inplace=True)  # Reset the index of the combined DataFrame
    return combined_df

#Combine windows data into one DataFrame (only in case there are more than one df)
def combine_with_loc(dfs_acce, dfs_loc):
    df_new = dfs_acce.join(dfs_loc, lsuffix = '', rsuffix = '_loc', how = 'outer').interpolate()
    return df_new

def map_data(df):
    coords = [(row.latitude, row.longitude) for _, row in df.iterrows()]
    my_map = folium.Map(location=[df.latitude.mean(), df.longitude.mean()], zoom_start=16)
    folium.PolyLine(coords, color="blue", weight=5.0).add_to(my_map)
    return my_map

map_data(ubahn1_loc)