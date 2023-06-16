import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.write("""
# Influence of Music Factors on Mental Health Conditions Prediction App

This app predicts the **Mental Health Conditions** either "Improve", "No effect", "Worsen"!
""")

st.sidebar.header('User Input Features')


# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        st.sidebar.write("Block 0: Background")
        hours = st.sidebar.slider('Numbers of hours listening to music per day', 0,24,12)
        working = st.sidebar.selectbox('Do you listen to music while studying/working?', ('Yes', 'No'))
        instrumentalist = st.sidebar.selectbox('Do you play an instrument regularly?', ('Yes', 'No'))
        composer = st.sidebar.selectbox('Do you compose music?', ('Yes', 'No'))
        genre = st.sidebar.selectbox('Your favorite or top genre', ('Rock', 'Pop', 'Metal', 'Classical','Video game music','EDM','R&B','Hip hop','Folk','K pop','Country','Rap','Jazz','Lofi','Gospel','Latin'))
        exploratory = st.sidebar.selectbox('Do you actively explore new artists/genres?', ('Yes', 'No'))
        language = st.sidebar.selectbox('Do you regularly listen to music with lyrics in a language you are not fluent in?', ('Yes', 'No'))
        
        st.sidebar.write("Block 1: Mental Health")
        anxiety = st.sidebar.slider('Self-reported anxiety, on a scale of 0-10', 0,10,5)
        depression = st.sidebar.slider('Self-reported depression, on a scale of 0-10', 0,10,5)
        insomnia = st.sidebar.slider('Self-reported insomnia, on a scale of 0-10', 0,10,5)
        ocd = st.sidebar.slider('Self-reported OCD, on a scale of 0-10', 0,10,5)
        
        st.sidebar.write("Block 2: Music Genre")
        freq_classical = st.sidebar.selectbox('How frequently you listen to classical music?', ('Never', 'Rarely', 'Sometimes', 'Very frequently'))
        freq_country = st.sidebar.selectbox('How frequently you listen to country music?', ('Never', 'Rarely', 'Sometimes', 'Very frequently'))
        freq_edm = st.sidebar.selectbox('How frequently you listen to EDM music?', ('Never', 'Rarely', 'Sometimes', 'Very frequently'))
        freq_folk = st.sidebar.selectbox('How frequently you listen to folks music?', ('Never', 'Rarely', 'Sometimes', 'Very frequently'))
        freq_gospel = st.sidebar.selectbox('How frequently you listen to Gospel?', ('Never', 'Rarely', 'Sometimes', 'Very frequently'))
        freq_hiphop = st.sidebar.selectbox('How frequently you listen to hip hop music?', ('Never', 'Rarely', 'Sometimes', 'Very frequently'))
        freq_latin = st.sidebar.selectbox('How frequently you listen to Latin music?', ('Never', 'Rarely', 'Sometimes', 'Very frequently'))
        freq_lofi = st.sidebar.selectbox('How frequently you listen to lofi music?', ('Never', 'Rarely', 'Sometimes', 'Very frequently'))
        freq_metal = st.sidebar.selectbox('How frequently you listen to metal music?', ('Never', 'Rarely', 'Sometimes', 'Very frequently'))
        freq_pop = st.sidebar.selectbox('How frequently you listen to pop music?', ('Never', 'Rarely', 'Sometimes', 'Very frequently'))
        freq_rnb = st.sidebar.selectbox('How frequently you listen to R&B music?', ('Never', 'Rarely', 'Sometimes', 'Very frequently'))
        freq_rap = st.sidebar.selectbox('How frequently you listen to rap music?', ('Never', 'Rarely', 'Sometimes', 'Very frequently'))
        freq_rock = st.sidebar.selectbox('How frequently you listen to rock music?', ('Never', 'Rarely', 'Sometimes', 'Very frequently'))
        freq_vgm = st.sidebar.selectbox('How frequently you listen to video game music? ', ('Never', 'Rarely', 'Sometimes', 'Very frequently'))
        

        data = {
                'Hours per day': hours,
                'Anxiety': anxiety,
                'Depression': depression,
                'Insomnia': insomnia,
                'OCD': ocd,'While working': working,
                'Instrumentalist': instrumentalist,
                'Composer': composer,
                'Fav genre': genre,
                'Exploratory': exploratory,
                'Foreign languages': language,
                'Frequency [Classical]': freq_classical,
                'Frequency [Country]': freq_country,
                'Frequency [EDM]': freq_edm,
                'Frequency [Folk]': freq_folk,
                'Frequency [Gospel]': freq_gospel,
                'Frequency [Hip hop]': freq_hiphop,
                'Frequency [Latin]': freq_latin,
                'Frequency [Lofi]': freq_lofi,
                'Frequency [Metal]': freq_metal,
                'Frequency [Pop]': freq_pop,
                'Frequency [R&B]': freq_rnb,
                'Frequency [Rap]': freq_rap,
                'Frequency [Rock]': freq_rock,
                'Frequency [Video game music]': freq_vgm,
                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
dataset_raw = pd.read_csv('mxmh_survey_results.csv')
data = dataset_raw.drop(['Timestamp','Age','Primary streaming service','Permissions'], axis=1)
data2 = data.drop(['BPM'], axis=1)

ds = pd.DataFrame(data2)
dss = ds.dropna()
dataset = dss.drop(columns=['Music effects'])
df = pd.concat([input_df,dataset],axis=0)

# Specify the columns you want to keep
columns_to_keep = ['Hours per day', 'While working', 'Instrumentalist', 'Composer', 'Fav genre', 'Exploratory', 'Foreign languages'
                   , 'Frequency [Classical]', 'Frequency [Country]', 'Frequency [EDM]', 'Frequency [Folk]', 'Frequency [Gospel]', 'Frequency [Hip hop]'
                   , 'Frequency [Latin]', 'Frequency [Lofi]', 'Frequency [Metal]', 'Frequency [Pop]', 'Frequency [R&B]', 'Frequency [Rap]'
                   , 'Frequency [Rock]', 'Frequency [Video game music]', 'Anxiety', 'Depression', 'Insomnia', 'OCD']

# Drop all columns except the ones specified
df = df[columns_to_keep]

# One-hot Encoding for input variables
encode = ['While working', 'Instrumentalist', 'Composer', 'Fav genre', 'Exploratory', 'Foreign languages'
               , 'Frequency [Classical]', 'Frequency [Country]', 'Frequency [EDM]', 'Frequency [Folk]', 'Frequency [Gospel]'
               , 'Frequency [Hip hop]', 'Frequency [Latin]', 'Frequency [Lofi]', 'Frequency [Metal]', 'Frequency [Pop]'
               , 'Frequency [R&B]', 'Frequency [Rap]', 'Frequency [Rock]', 'Frequency [Video game music]']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('mental_effect_pred_svm.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)


st.subheader('Prediction')
mental_effect_cat = np.array(['Improve', 'No effect', 'Worsen'])
st.write(mental_effect_cat[prediction])
