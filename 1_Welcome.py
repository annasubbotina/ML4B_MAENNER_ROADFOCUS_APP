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
import json
import numpy as np
#from sklearn.metrics import mean_squared_error
import numpy as np 
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import folium
import graphviz

@st.cache_data

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

#Read date for ubahn
ubahn1_df = transform_data_acceleration('Data/Ubahn/Json/ubahn1.json','json')
ubahn2_df = transform_data_acceleration('Data/Ubahn/Json/ubahn2.json','json')
ubahn3_df = transform_data_acceleration('Data/Ubahn/Json/ubahn3.json','json')
ubahn4_df = transform_data_acceleration('Data/Ubahn/Json/ubahn4.json','json')
ubahn5_df = transform_data_acceleration('Data/Ubahn/Json/ubahn5.json','json')
ubahn6_df = transform_data_acceleration('Data/Ubahn/Json/ubahn6.json','json')
ubahn7_df = transform_data_acceleration('Data/Ubahn/Json/ubahn7.json','json')

ubahn1_loc = transform_data_location('Data/Ubahn/Json/ubahn1.json','json')
ubahn2_loc = transform_data_location('Data/Ubahn/Json/ubahn2.json','json')
ubahn3_loc = transform_data_location('Data/Ubahn/Json/ubahn3.json','json')
ubahn4_loc = transform_data_location('Data/Ubahn/Json/ubahn4.json','json')
ubahn5_loc = transform_data_location('Data/Ubahn/Json/ubahn5.json','json')
ubahn6_loc = transform_data_location('Data/Ubahn/Json/ubahn6.json','json')
ubahn7_loc = transform_data_location('Data/Ubahn/Json/ubahn7.json','json')

dfs = [ubahn1_df, ubahn2_df, ubahn3_df, ubahn4_df, ubahn5_df, ubahn6_df, ubahn7_df]
ubahn_df = combine_into_df(dfs, 'ubahn')
ubahn_df.head()

ubahn_df.isnull().sum() 

#Read data for fahrrad
fahrrad2_df = transform_data_acceleration('Data/Fahrrad/Fahrrad2.json','json')
fahrrad3_df = transform_data_acceleration('Data/Fahrrad/Fahrrad3.json','json')
fahrrad2_df.head()

fahrrad2_df.head()

dfs = [fahrrad2_df, fahrrad3_df]
fahrrad_df = combine_into_df(dfs, 'fahrrad')
fahrrad_df.head()

fahrrad_df.isnull().sum()

def transform_data_acce_special(file, format):
    if format == 'json':
        acce = pd.read_json(file)
    elif format == 'csv':
        acce = pd.read_csv(file)
    acce.reset_index(drop=True, inplace=True)   
    #acce = acce.drop(columns =['sensor', 'relativeAltitude', 'pressure', 'altitude', 'speedAccuracy', 'bearingAccuracy', 'latitude', 'altitudeAboveMeanSeaLevel', 'bearing', 'horizontalAccuracy', 'verticalAccuracy', 'longitude', 'speed', 'version', 'device name', 'recording time', 'platform', 'appVersion', 'device id', 'sensors', 'sampleRateMs', 'yaw', 'qx', 'qz', 'roll', 'qw', 'qy', 'pitch'])
    acce['Magnitude_acce'] = np.sqrt(acce["x"] ** 2 + acce["y"] ** 2 + acce["z"] ** 2)
    acce.index = pd.to_datetime(acce['time'], unit = 'ns',errors='ignore')
    acce.drop(columns=['time','seconds_elapsed'], inplace=True)
    return acce

auto1 = transform_data_acceleration('Data/Auto/auto1_prep.json','json')
auto1.head()

auto2 = transform_data_accelleration_auto('Data/Auto/auto2_prep.json')
auto3 = transform_data_accelleration_auto('Data/Auto/auto3_prep.json')
auto4 = transform_data_accelleration_auto('Data/Auto/auto4_prep.json')
auto5 = transform_data_accelleration_auto('Data/Auto/auto5_prep.json')

dfs = [auto1, auto2, auto3, auto4, auto5]
auto_df = combine_into_df(dfs, 'auto')
auto_df.head()

auto_df.isnull().sum()

#Tranform accelerometer and gyroscope data to one dataframe
def transform_data_acceleration_bus(file, format):
    if format == 'json':
        df = pd.read_json(file)
    else:
        df = pd.read_csv(file)  
        
    acce = df[df['sensor'] == 'Accelerometer']
    acce.reset_index(drop=True, inplace=True)   
    acce = acce.drop(columns =['seconds_elapsed','sensor', 'altitude', 'speedAccuracy', 'bearingAccuracy', 'latitude',  'bearing', 'horizontalAccuracy', 'verticalAccuracy', 'longitude', 'speed', 'version', 'device name', 'recording time', 'platform', 'appVersion', 'device id', 'sensors', 'sampleRateMs', 'yaw', 'qx', 'qz', 'roll', 'qw', 'qy', 'pitch'])
    acce['Magnitude_acce'] = np.sqrt(acce["x"] ** 2 + acce["y"] ** 2 + acce["z"] ** 2)
    
    gyro = df[df['sensor'] == 'Gyroscope']
    gyro.reset_index(drop=True, inplace=True)   
    gyro = gyro.drop(columns = ['seconds_elapsed','sensor', 'altitude', 'speedAccuracy', 'bearingAccuracy', 'latitude',  'bearing', 'horizontalAccuracy', 'verticalAccuracy', 'longitude', 'speed', 'version', 'device name', 'recording time', 'platform', 'appVersion', 'device id', 'sensors', 'sampleRateMs', 'yaw', 'qx', 'qz', 'roll', 'qw', 'qy', 'pitch'])
    

    for df in [gyro, acce]:
         df.index = pd.to_datetime(df['time'], unit = 'ns',errors='ignore')
         df.drop(columns=['time'], inplace=True)
    #df_new = pd.merge(loc, gyro, suffixes=('_loc', '_gyro'), on='time')
    df_new = acce.join(gyro, lsuffix = '_acce', rsuffix = '_gyro', how = 'outer').interpolate()
   
    #df_new = pd.merge(pd.merge(loc, gyro, suffixes=('_loc', '_gyro'), on='time'), acce, suffixes=('', '_acce'), on='time')
    #df_new['Type'] = type
    
    return df_new

bus1 = transform_data_acceleration_bus('Data/Bus/Bus_Bauvereinstr.-Technische Hochschule-Dürrenhof.23-05-23_18-29-17.json', 'json')
bus2 = transform_data_acceleration_bus('Data/Bus/Bus_Harmoniestr.-Wöhrd.23-05-23_18-27-14.json','json')
bus3 = transform_data_acceleration_bus('Data/Bus/Bus_Laufer Tor-Rathenauplatz.23-05-23_18-24-45.json','json')
bus4 = transform_data_acceleration_bus('Data/Bus/Bus_Rathenauplatz-Harmoniestr.23-05-23_18-25-46.json','json')
bus5 = transform_data_acceleration_bus('Data/Bus/Bus_Wöhrd-Bauvereinstr.23-05-23_18-28-17.json','json')
bus6 = transform_data_acceleration_bus('Data/Bus/bus-_Gleitwitzertr._-_Langwasser_Mitte-2023-05-24_12-06-59 2.json','json')
bus7 = transform_data_acceleration_bus('Data/Bus/Bus2_Dürrenhof-Stephanstr.23-05-23_18-31-17.json','json')

dfs = [bus1, bus2, bus3, bus4, bus5, bus6, bus7]
bus_df = combine_into_df(dfs, 'bus')
bus_df.head()

bus_df.isnull().sum()

df_completed = pd.concat([ubahn_df, fahrrad_df, bus_df, auto_df])

df_completed.isnull().sum()

df_completed.head(2)

#separating labels and predictors
X = df_completed.drop('Type',axis=1)
y = df_completed['Type'].values

#splitting train (75%) and test set (25%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state=42)

#Gini index as our attribute selection method for the training of decision tree classifier with Sklearn function DecisionTreeClassifier()
clf_model = DecisionTreeClassifier(criterion="gini", random_state=42,max_depth=5, min_samples_leaf=5)   
clf_model.fit(X_train,y_train)

#Define target and features using
target = list(df_completed['Type'].unique())
feature_names = list(X.columns)
print(target)
print(feature_names)

#Train model using Decision Tree Classifier
classifier_decision_tree = tree.DecisionTreeClassifier()
classifier_decision_tree.fit(X_train, y_train)
tree_predictions = classifier_decision_tree.predict(X_train)
y_predict = clf_model.predict(X_test)

#Calculate accuracy
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
acc = accuracy_score(y_test,y_predict)
print("Accuracy of Decision Tree Classifier: " + str(acc))
#print(classification_report(y_test, y_predict))


dot_data = tree.export_graphviz(clf_model,
                                out_file=None, 
                      feature_names=feature_names,  
                      class_names=target,  
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data)  

graph


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
            
            uploaded_file = st.file_uploader("Drop it here ⇓", type=["json"])

            if uploaded_file is not None:
                user_df = pd.read_json(uploaded_file)
                user_df = preprocess_data(user_df) 
                model = train_model(user_df)
                transportation = predict_transportation(model, user_df)
                st.write("You are using " + str(transportation) + "!")

            #if uploaded_file is not None:
                #prediction_data = process_data_prediction(uploaded_file)
                #location_data = process_data_location(uploaded_file)
                #st.subheader("Your travel graph")
                #map_data(user_df)
                #tree_predictions = model.predict(prediction_data)
                #st.caption("You are using " + str(tree_predictions) + "!")
            
            else:
                st.write("Upload a JSON file!")
        with right_column:
                st_lottie(lottie_coding, height=300, key="coding")
        

if __name__ == "__main__":
    main()