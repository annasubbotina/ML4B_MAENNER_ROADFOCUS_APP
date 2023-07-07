import streamlit as st
from streamlit_lottie import st_lottie
import requests
from PIL import Image


st.set_page_config(page_title="MoveMate", page_icon=":oncoming_automobile:", layout="wide", initial_sidebar_state="collapsed")


st.header("Instructions") 

def load_lottieurl(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
left_column, right_column = st.columns(2)

lottie_coding = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_TDvLW1etRl.json")

SensorLogo = Image.open("SensorLogo.png")
Scr1 = Image.open("Screenshot_20230706-233406_Sensor Logger.jpg")
Scr2 = Image.open("Screenshot_20230706-233419_Sensor Logger.jpg")
Scr3 = Image.open("Screenshot_20230706-233438_Sensor Logger.jpg")
Scr4 = Image.open("Screenshot_20230707-032717_Sensor Logger.jpg")
Scr5 = Image.open("Screenshot_20230707-033306_Sensor Logger.jpg")
Scr6 = Image.open("Screenshot_20230707-032741_Sensor Logger.jpg")
Scr7 = Image.open("Screenshot_20230707-032753_Sensor Logger.jpg")

with st.container():
    st.write("---")
    image_column, text_column = st.columns((1,2))
    with image_column:
        st.image(SensorLogo)
    with text_column:
        st.write("In order to upload your movement data, you can use the Sensor Logger App. ")
        st.write("Type to download Sensor Logger App")
        st.markdown("Download Sensor Logger App for Android (https://play.google.com/store/apps/details?id=com.kelvin.sensorapp&hl=en_US&pli=1)")
        st.markdown("Download Sensor Logger App for Apple (https://apps.apple.com/us/app/sensor-logger/id1531582925)")
        
    st.write("---")
    image_column, text_column = st.columns((1,2))
    with image_column:
        st.image(Scr1)
    with text_column:
        st.write("After installing the application open it and familiarize yourself with its interface.") 

    st.subheader("For the correct recording of your json file and upload it to our application, follow these steps:")
with st.container():
    text_column, image_column = st.columns((2,1))
    with image_column:
         st.image(Scr2)
    with text_column:
          st.write("Turn on all the necessary Sensors as shown in the screenshot")
with st.container():
    image_column, text_column  = st.columns((1,2))
    with image_column:
          st.image(Scr3)
    with text_column:
        st.write("Turn off all  unnecessary Sensors as shown in the screenshot")
with st.container():
    text_column, image_column = st.columns((2,1))
    with image_column:
         st.image(Scr4)
    with text_column:
          st.write("Click on the Start Recording button and record your movement")
          st.write("After recording click the button End recording")
with st.container():
    image_column, text_column  = st.columns((1,2))
    with image_column:
          st.image(Scr5)
    with text_column:
        st.write("Click on the Recordings icon. Here you can view the records you have made about your movement")
        st.write("Click on your entry as shown in the screenshot")
with st.container():
    text_column, image_column = st.columns((2,1))
    with image_column:
         st.image(Scr6)
    with text_column:
          st.write("You will see a record with sensor data")
          st.write("Click the Export button to export the file to your device")
with st.container():
    image_column, text_column  = st.columns((1,2))
    with image_column:
          st.image(Scr7)
    with text_column:
        st.write("We're almost there!")
        st.write("Just click the Export Recording button and download your json file to a convenient location for you")
st.write("---")
st.subheader("Now you can upload your file directly to our app for classifying your vehicle!")

with left_column:
    st_lottie(lottie_coding, height=300, key="coding")
        