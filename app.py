import streamlit as st
import pandas as pd
# import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import plotly.graph_objects as go
st.write("""
# IPL Match Result Predictor
This app predicts the **Match Result** 
Data obtained from the [ESPN CricInfo](https://www.espncricinfo.com/)
""")

st.sidebar.header('Select Team')


def user_input_features():
    Team_1 = st.sidebar.selectbox('Team 1',('MumbaiIndians', 'DelhiCapitals', 'SunrisersHyderabad','RajasthanRoyals', 'KolkataKnightRiders', 'PunjabLions','ChennaiSuperKings', 'RoyalChallengersBangalore'))
    Team_2 = st.sidebar.selectbox('Team 2',('DelhiCapitals', 'MumbaiIndians', 'SunrisersHyderabad','RajasthanRoyals', 'KolkataKnightRiders', 'PunjabLions','ChennaiSuperKings', 'RoyalChallengersBangalore'))
    
    data = {'Team 1': Team_1,'Team 2': Team_2}
    features = pd.DataFrame(data, index=[0])

    return features
    
input_df = user_input_features()


sort_bat_dict = {'KolkataKnightRiders': -0.1429884056497032, 'DelhiCapitals': -0.09568159714717964, 'RajasthanRoyals': -0.08782838652354537, 'RoyalChallengersBangalore': -0.03603592723019618, 'SunrisersHyderabad': -0.021699009247779087, 'PunjabLions': 0.019528966136976307, 'ChennaiSuperKings': 0.026760242674982237, 'MumbaiIndians': 0.10870125659946274}

sort_bowl_dict = {'DelhiCapitals': -0.07786219049160013, 'MumbaiIndians': -0.07627751857081548, 'PunjabLions': -0.02574726131346169, 'SunrisersHyderabad': -0.007227276121737493, 'ChennaiSuperKings': -0.0008597099517881068, 'RoyalChallengersBangalore': 0.0490952463760612, 'KolkataKnightRiders': 0.06782003313709033, 'RajasthanRoyals': 0.06876273852859732}

team_dict = {'Mum Indians':'MumbaiIndians' , 'Super Kings':'ChennaiSuperKings' ,'Sunrisers' : 'SunrisersHyderabad' ,
             'Capitals': 'DelhiCapitals' , 'Kings XI': 'PunjabLions', 'KKR': 'KolkataKnightRiders', 'Royals':'RajasthanRoyals' ,'RCB' :'RoyalChallengersBangalore' }

df = pd.read_html('https://stats.espncricinfo.com/ci/engine/records/team/match_results.html?id=13533;type=tournament', header = 0,flavor='html5lib')
df = df[0][['Team 1','Team 2']]
final_df = pd.concat([input_df,df],axis=0)

final_df = final_df.replace({"Team 1": team_dict,"Team 2": team_dict})

final_df['team1bat']=final_df['Team 1'].apply(lambda x: sort_bat_dict[x] if x in sort_bat_dict else '')
final_df['team1bowl']=final_df['Team 1'].apply(lambda x: sort_bowl_dict[x] if x in sort_bowl_dict else '')
final_df['team2bat']=final_df['Team 2'].apply(lambda x: sort_bat_dict[x] if x in sort_bat_dict else '')
final_df['team2bowl']=final_df['Team 2'].apply(lambda x: sort_bowl_dict[x] if x in sort_bowl_dict else '')


final_df = pd.get_dummies(final_df, columns=['Team 1','Team 2'])


final_df = final_df[:1]


# Reads in saved classification model
load_clf = pickle.load(open('IPL_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(final_df)
prediction_proba = load_clf.predict_proba(final_df)
new_header = input_df.iloc[0] #grab the first row for the header
input_df = input_df[1:] #take the data less the header row
input_df.columns = new_header #set the header row as the df header



new_dict = {}
for c in input_df.columns:
    if prediction[0] == c:
        new_dict[c] =  1 - prediction_proba.max() 

    else:
        new_dict[c] =  prediction_proba.max()

pred_df = pd.DataFrame([new_dict])


st.subheader('Prediction')

colors = ['gold', 'mediumturquoise']

# fig, ax = plt.subplots()
fig = go.Figure(data=[go.Pie(labels=[pred_df.columns[0],pred_df.columns[1]],values=[pred_df.iloc[0][0],pred_df.iloc[0][1]])])
fig.update_traces(hoverinfo='label+percent', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
# st.set_option('deprecation.showPyplotGlobalUse', False)
st.plotly_chart(fig)

st.write("""
The classifier model is trained on matches played in IPL 2020-21 season
""")