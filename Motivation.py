#!/usr/bin/env python
# coding: utf-8

# In[9]:

#!/usr/bin/env python
# coding: utf-8

# In[9]:


from audioop import reverse
from turtle import width
import requests 
import json 
import streamlit as st
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Mobility Classification App", page_icon=":oncoming_automobile:", layout="wide", initial_sidebar_state="collapsed")

# Weitere Codezeilen...


st.header("Unsere Motivation") 
st.write("Unser täglicher Wahnsinn beginnt bereits, wenn wir das Haus verlassen und uns fragen, wie wir am schnellsten von Punkt A nach Punkt B gelangen können - mit dem Auto, dem Fahrrad oder den öffentlichen Verkehrsmitteln? Jedes dieser Transportmittel hat seine Vor- und Nachteile. Wenn es einen Stau gibt, kann das Autofahren möglicherweise zeitaufwändig sein, während das Fahrradfahren bei anstrengenden Strecken möglicherweise nicht die beste Wahl ist und die Verbindungen mit den öffentlichen Verkehrsmitteln möglicherweise nicht optimal für das gewünschte Ziel sind. Genau hier setzt unsere Idee an: Wir haben eine innovative Fahrzeugerkennungs-App entwickelt, die automatisch erkennt, mit welchem Transportmittel du unterwegs bist und welche Strecken du häufig befährst. Mit diesen Informationen kann dir unsere App die beste Route empfehlen und sogar sagen, welches Verkehrsmittel für dich heute die beste Wahl ist. Unsere App nutzt fortschrittliche Technologie, um deine bevorzugten Verkehrsmittel zu analysieren und deine üblichen Routen zu erkennen. Basierend auf Echtzeitverkehrsdaten und Informationen über Straßenzustände kann sie dir die schnellste und bequemste Route für deinen täglichen Weg zur Arbeit, zur Uni oder zu anderen Zielen anzeigen.Egal, ob du Zeit sparen möchtest, dich umweltfreundlicher fortbewegen willst oder einfach nur den stressfreisten Weg finden möchtest, unsere Fahrzeugerkennungs-App steht dir zur Seite, um die beste Entscheidung zu treffen und deinen Alltag zu erleichtern.")


# Lade die Lottie-Animation von der Webadresse
url = "https://assets5.lottiefiles.com/private_files/lf30_x9f70hcu.json"
response = requests.get(url)
animation = response.json()

# Zeige die Lottie-Animation an
col1, col2,col3=st.columns([1,1,1])
with col2 : 
    st_lottie(animation, speed=1, reverse=False, loop=True, width=350, height=350)

st.header("Zeitlicher Ablauf der Entwicklug")

st.write("Recherche und Datenauswahl: Wir führten umfangreiche Recherchen durch, um herauszufinden, welche Daten sich am besten für unseren Zweck eignen. Wir untersuchten verschiedene Datenquellen, wie beispielsweise Sensoren oder APIs, um genaue Informationen über das aktuell genutzte Transportmittel zu erhalten.")
st.write("Datensammlung: Die Sammlung ausreichender Daten erwies sich als eine Herausforderung. Wir benötigten eine große Menge an qualitativ hochwertigen Daten, um unsere App effektiv zu trainieren. Dabei stellten wir fest, dass wir zu Beginn fehlerhafte Daten aufgenommen hatten. Dies erforderte, dass wir die Daten löschen und den Sammelprozess erneut starten mussten, um genaue und zuverlässige Informationen zu erhalten.")
st.write("Datenbereinigung und -verarbeitung: Nachdem wir die gesammelten Daten hatten, mussten wir sie bereinigen und aufbereiten, um sie für die spätere Verarbeitung optimal zu nutzen. Dies beinhaltete das Entfernen von Rauschen, Ausreißern oder inkonsistenten Datenpunkten. Wir verwendeten verschiedene Datenverarbeitungstechniken und Algorithmen, um sicherzustellen, dass die Daten qualitativ hochwertig und repräsentativ waren.")
st.write("Modellbildung und Algorithmusauswahl: Ein weiterer wichtiger Schritt bestand darin, ein Modell zu entwickeln, das in der Lage ist, das aktuelle Transportmittel zu erkennen. Wir haben verschiedene Algorithmen und Machine-Learning-Techniken evaluiert, um die beste Leistung und Genauigkeit zu erzielen. Dies erforderte eine sorgfältige Auswahl und Anpassung des Modells an unsere spezifischen Anforderungen.")
st.write("Validierung und Optimierung: Nach der Modellbildung haben wir das System umfangreich validiert, um sicherzustellen, dass es zuverlässig und genau funktioniert. Dies beinhaltete die Verwendung von Testdaten und die Überprüfung der Ergebnisse im Vergleich zu den tatsächlichen Transportmitteln. Wir haben das Modell iterativ optimiert und verbessert, um eine höhere Genauigkeit und Robustheit zu erreichen.")
st.write("Während des gesamten Entwicklungsprozesses waren wir mit Herausforderungen konfrontiert, die es erforderten, kreative Lösungen zu finden. Dazu gehörten auch die Zusammenarbeit im Team, das Überwinden technischer Hindernisse und das kontinuierliche Feedback von Testnutzern.")

#hochladen der bild datei 
url = "https://assets1.lottiefiles.com/packages/lf20_8cxcnczq.json"
response = requests.get(url)
prozess = response.json()

# zeige die Animation an 

col4, col5, col6 = st.columns([1, 1, 1])

with col5: 
    st.lottie(prozess, speed=1, reverse=False, loop=True, width=350, height=350)


st.header("Wir Blicken in die Zukunft")

# Raumschiff bild 

url="https://assets7.lottiefiles.com/packages/lf20_ry2uhako.json"
response = requests.get(url)
raumschiff = response.json()

# Anzeige anitmation 

col7,col8,col9= st.columns([1,1,1])

with col8: 
    st.lottie(raumschiff, speed=1, reverse=False, loop=True, width=400, height=400)

    st.write("3. Des weiteren kann die App die beste Route sowohl für Autofahrten als auch für Fahrradfahrten berechnen wodurch sie sich als praktischer Begleiter im Straßenverkehr erweist. ")

with col7: 
    st.write("1.Eine Möglichkeit besteht darin, den eigenen Status während der Autofahrt automatisch an Kontakte zu übermitteln, um sicherzustellen, dass man während der Fahrt nicht abgelenkt wird und keine Anrufe entgegennimmt. ")


    with col9: 
        st.write("2. Auch bei der Nutzung öffentlicher Verkehrsmittel erweist sich die App nützlich, indem sie die besten Verbindungsmöglichkeiten und Routenempfehlungen bietet.")

        import streamlit as st
