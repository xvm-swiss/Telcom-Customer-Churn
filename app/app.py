import os
import numpy as np
import pandas as pd
import streamlit as st # streamlit run app/app.py
import joblib
import plotly.express as px

model = joblib.load( 'data/artifacts/knn_pipeline.pkl')
st.set_page_config( 'Telecom Customer', ':book:', 'wide')


st.markdown("""
    <style>
    .main-title { 
        font-size: calc(20px + 2vw) !important; 
        color: orange !important; 
        font-family: 'Times New Roman', Times, serif !important; 
        text-align: center !important; 
        margin-bottom: 20px;
    }
    </style>
    <h1 class="main-title">Telecom Costomer Churn </h1>
    """, unsafe_allow_html=True)



# Chat boot Gemini

from google import genai
import os

# 1. API KEY laden (Streamlit sucht lokal in .streamlit/secrets.toml)
# Wenn du es auf GitHub/Streamlit Cloud hochlädst, musst du den Key dort in den Settings eintragen!
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("Fehler: GEMINI_API_KEY nicht in den Secrets gefunden!")
    st.stop()

client = genai.Client(api_key=api_key)

st.header('Our First Chatbot')

# System Instruction definieren
system_prompt = """You are a helpful AI assistant for a Parkinson's Disease (PD) Detection web application...
(Dein restlicher Text hier)"""

user_input = st.text_area('Enter your text')

if st.button('Send'):
    if user_input:
        response = client.models.generate_content(
            model="gemini-2.0-flash", # Tipp: Nutze die aktuelle stabile Version
            contents=user_input,
            config={
                'system_instruction': system_prompt
            }
        )
        st.write(response.text)
    else:
        st.warning("Please enter some text first.")


st.divider() # Erzeugt eine saubere, graue Trennlinie



df = pd.read_csv( 'data/preprocessing data/preprocessing_data.csv')







# Part 1
box_11 , box_12, box_13, box_14, box_15 = st.columns( 5 )

box_1, = st.columns( 1 )

st.divider() # Erzeugt eine saubere, graue Trennlinie

# Part 2
box_16a, box_16, box_17, box_18, box_19 = st.columns(5)

box_2, = st.columns( 1 )

st.divider() # Erzeugt eine saubere, graue Trennlinie

# Part 4
box_27, box_28, box_29, box_30, box_31 = st.columns(5)

box_4, = st.columns( 1 )

st.divider() # Erzeugt eine saubere, graue Trennlinie

# Part 3
box_21, box_22, box_23, box_24, box_25 = st.columns(5)

box_3, = st.columns( 1 )

st.divider() # Erzeugt eine saubere, graue Trennlinie


# Row 1
genders = box_11.multiselect( 'Gender:',
                          options= df["gender"].unique(),
                          default= df["gender"].unique())

onlineSecuritys = box_16a.multiselect( 'Online Security:',
                                    options= df["OnlineSecurity"].unique(),
                                    default= df["OnlineSecurity"].unique() )

paymentMethods = box_21.multiselect( 'Payment Method:',
                                options= df["PaymentMethod"].unique(),
                                 default= df["PaymentMethod"].unique())

churns = box_27.multiselect( 'Churn:',
                                 options= df["Churn"].unique(),
                                default=df["Churn"].unique())
# Filter
filtered_df = df.query(
    'gender in @genders and '
    'OnlineSecurity in @onlineSecuritys and '
    'PaymentMethod in @paymentMethods and '
    'Churn in @churns'
)



# Kpi 1
female_sum =  (filtered_df["gender"] == 'Female').sum()
male_sum =  (filtered_df["gender"] == 'Male').sum()
totall_sum = filtered_df.shape[0]

# Columns 1
box_13.markdown(f'<h3>Male: <br> {male_sum}</h3>', unsafe_allow_html=True)
box_14.markdown(f'<h3>Female: <br> {female_sum}</h3>', unsafe_allow_html= True)
box_15.markdown(f'<h3>Totall: <br> {totall_sum}</h3>', unsafe_allow_html=True)



# Kpi 2
no_sum = filtered_df[ df["OnlineSecurity"]== 'No'].value_counts().sum()
yes_sum = filtered_df[  df["OnlineSecurity"]== 'Yes'].value_counts().sum()
no_internet_sum = filtered_df[ df["OnlineSecurity"]== 'No internet service'].value_counts().sum()

# Columns 2
box_17.markdown(f'<h3>No Security: <br> {no_sum}</h3>', unsafe_allow_html= True)
box_18.markdown(f'<h3>Yes Security: <br> {yes_sum}</h3>', unsafe_allow_html=True)
box_19.markdown(f'<h3>No Internet: <br> {no_internet_sum}</h3>', unsafe_allow_html=True)


# Kpi 3
Electronic_check = filtered_df[ df["PaymentMethod"]== 'Electronic check'].value_counts().sum()
Mailed_check = filtered_df[  df["PaymentMethod"]== 'Mailed check'].value_counts().sum()
Bank_transfer = filtered_df[ df["PaymentMethod"]== 'Bank transfer (automatic)'].value_counts().sum()
Credit_card = filtered_df[ df["PaymentMethod"]== 'Credit card (automatic)'].value_counts().sum()

# Columns 3
box_22.markdown(f'<h3>Electronic check: <br> {Electronic_check}</h3>', unsafe_allow_html= True)
box_23.markdown(f'<h3>Mailed check: <br> {Mailed_check}</h3>', unsafe_allow_html=True)
box_24.markdown(f'<h3>Bank transfer: <br> {Bank_transfer}</h3>', unsafe_allow_html=True)
box_25.markdown(f'<h3>Credit card: <br> {Credit_card}</h3>', unsafe_allow_html=True)

# Kpi 4
Churn_male = filtered_df["Churn"][df["gender"] == 'Male'].value_counts().sum()
Churn_female = filtered_df["Churn"][df["gender"] == 'Female'].value_counts().sum()
Churn_sum = filtered_df["Churn"].value_counts().sum()

# Columns 4
box_29.markdown(f'<h3>Churn Male: <br> {Churn_male}</h3>', unsafe_allow_html= True)
box_30.markdown(f'<h3>Churn Female: <br> {Churn_female}</h3>', unsafe_allow_html=True)
box_31.markdown(f'<h3>Total: <br> {Churn_sum}</h3>', unsafe_allow_html=True)


# Box and Grafik 1
# 2. Daten vorbereiten (Beispiel basierend auf deinem filtered_df)
gender_counts = filtered_df['gender'].value_counts().reset_index()
gender_counts.columns = ['Gender', 'Count']

# 
fig = px.bar(gender_counts, 
             x='Gender', 
             y='Count',
             color='Gender',
             text='Count',
             title='Gender Distribution',
             color_discrete_map={'Female': 'pink', 'Male': 'blue'})

fig.update_traces(textposition='auto')

# 4. In Spalte 12 anzeigen (WICHTIG: box_12. davor setzen)
box_1.plotly_chart(fig, use_container_width=True)



# Box and Grafik 2

# 1. Sicherstellen, dass die Variablen berechnet sind
counts = filtered_df["OnlineSecurity"].value_counts()
m_yes = counts[0]
m_no = counts[1]
m_no_internet= counts[2]
m_total = counts.sum()

# 2. Direkt Listen an Plotly übergeben
fig = px.bar(x=['No', 'Yes', 'No Internet'], 
             y=[m_yes, m_no, m_no_internet],
             color=['No', 'Yes', 'No Internet'],
             text=[m_yes, m_no, m_no_internet],
             labels={'x': 'OnlineSecurity Status', 'y': 'Count'},
             title=f'OnlineSecurity Distribution (Total: {m_total})')

fig.update_traces(textposition='inside')
# 4. In Spalte 12 anzeigen (WICHTIG: box_12. davor setzen)
box_2.plotly_chart(fig, use_container_width=True)

# Box and Grafik 3:

# 1. Sicherstellen, dass die Variablen berechnet sind
counts = filtered_df["PaymentMethod"].value_counts()
m_yes = counts[0]
m_no = counts[1]
m_no_internet= counts[2]
m_credit_card = counts[3]
m_total = counts.sum()

# 2. Direkt Listen an Plotly übergeben
fig = px.bar(x=['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], 
             y=[m_yes, m_no, m_no_internet, m_credit_card],
             color=['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
             text=[m_yes, m_no, m_no_internet, m_credit_card],
             labels={'x': 'PaymentMethod Status', 'y': 'Count'},
             title=f'PaymentMethod Distribution (Total: {m_total})')

fig.update_traces(textposition='inside')
box_3.plotly_chart(fig, use_container_width=True)

# Box 4 and Grafik 4

counts = filtered_df["Churn"].value_counts()
m_yes = counts[0]
m_no = counts[1]
m_total = counts.sum()

# 2. Direkt Listen an Plotly übergeben
fig = px.bar(x=['No', 'Yes'], 
             y=[m_yes, m_no],
             color=['No', 'Yes'],
             text=[m_yes, m_no],
             labels={'x': 'Churn Status', 'y': 'Count'},
             title=f'Churn Distribution (Total: {m_total})')

fig.update_traces(textposition='inside')
box_4.plotly_chart(fig, use_container_width=True)


#model = joblib.load( 'data/artifacts/knn_pipeline.pkl')
# pipeline = joblib.load(knn_pipeline)

#df = pd.read_csv('data/preprocessing data/preprocessing_data.csv')
