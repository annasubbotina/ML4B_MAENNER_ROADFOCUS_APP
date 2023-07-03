import requests 
import json 
import streamlit as st
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Mobility Classification App", page_icon=":oncoming_automobile:", layout="wide", initial_sidebar_state="collapsed")

# Weitere Codezeilen...

st.header("Unsere Motivation")


st.header("Unsere Motivation") 
st.write("Unser täglicher Wahnsinn beginnt bereits, wenn wir das Haus verlassen und uns fragen, wie wir am schnellsten von Punkt A nach Punkt B gelangen können - mit dem Auto, dem Fahrrad oder den öffentlichen Verkehrsmitteln?")
st.write("Jedes dieser Transportmittel hat seine Vor- und Nachteile. Wenn es einen Stau gibt, kann das Autofahren möglicherweise zeitaufwändig sein, während das Fahrradfahren bei anstrengenden Strecken möglicherweise nicht die beste Wahl ist und die Verbindungen mit den öffentlichen Verkehrsmitteln möglicherweise nicht optimal für das gewünschte Ziel sind.")
st.write("Genau hier setzt unsere Idee an: Wir haben eine innovative Fahrzeugerkennungs-App entwickelt, die automatisch erkennt, mit welchem Transportmittel du unterwegs bist und welche Strecken du häufig befährst")
st.write("Mit diesen Informationen kann dir unsere App die beste Route empfehlen und sogar sagen, welches Verkehrsmittel für dich heute die beste Wahl ist.")
st.write("Unsere App nutzt fortschrittliche Technologie, um deine bevorzugten Verkehrsmittel zu analysieren und deine üblichen Routen zu erkennen.")
st.write("Basierend auf Echtzeitverkehrsdaten und Informationen über Straßenzustände kann sie dir die schnellste und bequemste Route für deinen täglichen Weg zur Arbeit, zur Uni oder zu anderen Zielen anzeigen.")
st.write("gal, ob du Zeit sparen möchtest, dich umweltfreundlicher fortbewegen willst oder einfach nur den stressfreisten Weg finden möchtest, unsere Fahrzeugerkennungs-App steht dir zur Seite, um die beste Entscheidung zu treffen und deinen Alltag zu erleichtern.")

import requests
import json

def load_lottieurl(url: str):
    response = requests.get(url)
    response.raise_for_status()  # Überprüfen, ob die Anfrage erfolgreich war

    return response.json()

lottie_url = "https://assets6.lottiefiles.com/packages/lf20_znsmxbjo.json"
lottie_content = load_lottieurl(lottie_url)

def load_lottieurl(url: str): 
    response = requests.get(url)
    response.raise_for_status() # überprüfen, ob die Anfrage erfolgreich war 
    
    return response.json()

lottie_url = "https://assets1.lottiefiles.com/private_files/lf30_rixr9r00.json"
lottie_content = load_lottieurl(lottie_url)

def load_lottieur(url: str): 
    response = requests.get(url)
    response.raise_for_status() # überprüfen, ob die anfrage erfolgreich war 
    
    return response.json()

lottie_url = " https://assets1.lottiefiles.com/packages/lf20_hwumabdt.json"
lottie_content = load_lottieur(lottie_url) 