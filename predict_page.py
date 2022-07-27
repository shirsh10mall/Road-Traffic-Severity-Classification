import sys
import subprocess
# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'feature_engine'])
import streamlit as st
# import pickle
import numpy
import pandas as pd 
import joblib
import feature_engine
# import shap

# from streamlit_shap import st_shap

# def load_model():
#     with open(r'/home/shirsh/Downloads/Customer-Churn-Prediction-main/model_XGB_top.dat','rb') as file:
#         data = pickle.load(file)
#     return data

loaded_model = joblib.load("model_RFC_top.dat")

ore_Day_of_week=loaded_model['ore_Day_of_week']
ore_Age_band_of_driver=loaded_model['ore_Age_band_of_driver']
ore_Educational_level=loaded_model['ore_Educational_level']
ore_Driving_experience=loaded_model['ore_Driving_experience']
ore_Light_conditions=loaded_model['ore_Light_conditions']
ore_Service_year_of_vehicle=loaded_model['ore_Service_year_of_vehicle']
ore_Age_band_of_casualty=loaded_model['ore_Age_band_of_casualty']

one_hot_encoder_top=loaded_model['one_hot_encoder_top']

final_model=loaded_model['final_model']     # convert_Time_into_minutes function is not included


def show_predict_page():

    st.title('Road Traffic - Accident Severity Prediction')
    st.write("""### We need some information for prediction """)        
    
    option_Type_of_collision = ['Collision with roadside-parked vehicles', 'Vehicle with vehicle collision', 'Collision with roadside objects', 'Collision with animals', 'Other', 'Rollover', 'Fall from vehicles', 'Collision with pedestrians', 'With Train']

    option_Age_band_of_casualty = ['31-50', '18-30', 'Under 18', 'Over 51', '5']

    option_Sex_of_casualty = ['Male', 'Female']
    
    option_Casualty_class = ['Driver or rider', 'Pedestrian', 'Passenger']
    
    option_Vehicle_movement = ['Going straight', 'U-Turn', 'Moving Backward', 'Turnover', 'Waiting to go', 'Getting off', 'Reversing', 'Parked', 'Stopping', 'Overtaking', 'Other', 'Entering a junction']
    
    option_Types_of_Junction = ['No junction', 'Y Shape', 'Crossing', 'O Shape', 'Other', 'T Shape', 'X Shape']
    
    option_Service_year_of_vehicle = ['Above 10yr', '5-10yrs','1-2yr', '2-5yrs', 'Below 1yr']
    
    option_Cause_of_accident = ['Moving Backward', 'Overtaking', 'Changing lane to the left', 'Changing lane to the right', 'Overloading', 'Other', 'No priority to vehicle', 'No priority to pedestrian', 'No distancing', 'Getting off the vehicle improperly', 'Improper parking', 'Overspeed', 'Driving carelessly', 'Driving at high speed', 'Driving to the left', 'Overturning', 'Turnover', 'Driving under the influence of drugs', 'Drunk driving']
    
    option_Driving_experience = ['1-2yr', 'Above 10yr', '5-10yr', '2-5yr', 'No Licence', 'Below 1yr']
    
    option_Educational_level =  ['Above high school', 'Junior high school', 'Elementary school', 'High school', 'Illiterate', 'Writing & reading']
    
    option_Type_of_vehicle = ['Automobile', 'Public (> 45 seats)', 'Lorry (41?100Q)','Public (13?45 seats)', 'Lorry (11?40Q)', 'Long lorry', 'Public (12 seats)', 'Taxi', 'Pick up upto 10Q', 'Stationwagen', 'Ridden horse', 'Other', 'Bajaj', 'Turbo', 'Motorcycle', 'Special vehicle', 'Bicycle']
    
    option_Age_band_of_driver = ['18-30', '31-50', 'Under 18', 'Over 51']

    option_Area_accident_occured = ['Residential areas', 'Office areas', '  Recreational areas', ' Industrial areas', 'Other', ' Church areas', '  Market areas', 'Rural village areas', ' Outside rural areas', ' Hospital areas', 'School areas', 'Rural village areasOffice areas', 'Recreational areas']

    option_Lanes_or_Medians = ['Undivided Two way', 'other', 'Double carriageway (median)', 'One way', 'Two-way (divided with solid lines road marking)', 'Two-way (divided with broken lines road marking)']

    option_Day_of_week = ['Monday', 'Sunday', 'Friday', 'Wednesday', 'Saturday', 'Thursday', 'Tuesday']

    option_Light_conditions = ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting', 'Darkness - lights unlit']
    

    Type_of_collision = st.selectbox('Type of Collision', options=option_Type_of_collision  )
    Age_band_of_casualty = st.selectbox('Age Band of Casualty', options=option_Age_band_of_casualty )
    Sex_of_casualty = st.selectbox('Sex of casualty', options=option_Sex_of_casualty )
    Casualty_class = st.selectbox('Casualty class', options= option_Casualty_class)
    Vehicle_movement = st.selectbox('Vehicle movement', options=option_Vehicle_movement)
    Area_accident_occured = st.selectbox('Area accident occured', options=option_Area_accident_occured )
    Type_of_collision = st.selectbox('Type of collision', options=option_Type_of_collision)
    Types_of_Junction = st.selectbox('Types of Junction', options=option_Types_of_Junction)
    Service_year_of_vehicle = st.selectbox('Service Year of Vehicle', options=option_Service_year_of_vehicle)
    Cause_of_accident = st.selectbox('Cause of Accident', options=option_Cause_of_accident )
    Driving_experience = st.selectbox('Driving Experience', options=option_Driving_experience)
    Educational_level = st.selectbox('Educational Level', options=option_Educational_level)
    Type_of_vehicle = st.selectbox('Type of Vehicle', options=option_Type_of_vehicle)
    Age_band_of_driver = st.selectbox('Age Band of Driver', options=option_Age_band_of_driver)
    Area_accident_occured = st.selectbox('Area Accident Occured', options=option_Area_accident_occured)
    Lanes_or_Medians = st.selectbox('Lanes or Medians', options=option_Lanes_or_Medians )
    Day_of_week = st.selectbox('Day of Week', options=option_Day_of_week)
    Light_conditions = st.selectbox('Light Conditions', options=option_Light_conditions)    
    
    Number_of_vehicles_involved = st.slider( 'Number of vehicles involved', 1, 7, 1 )
    Number_of_casualties = st.slider( 'Number_of_casualties', 1, 8, 1 )
    Time_Hour = st.slider( 'Time (Hours)', 0, 24,1 )
    Time_Minute = st.slider( 'Time (Minute)', 0, 60, 1 )
    Time = Time_Hour*60 + Time_Minute

    ok = st.button("Predict")
    
    columns_name = ['Type_of_collision', 'Age_band_of_casualty', 'Sex_of_casualty', 'Casualty_class', 'Vehicle_movement', 'Types_of_Junction',
       'Service_year_of_vehicle', 'Cause_of_accident', 'Driving_experience', 'Educational_level', 'Type_of_vehicle', 'Age_band_of_driver',
       'Area_accident_occured', 'Lanes_or_Medians',  'Day_of_week', 'Light_conditions', 'Time', 'Number_of_vehicles_involved', 'Number_of_casualties']

    values = [Type_of_collision, Age_band_of_casualty, Sex_of_casualty, Casualty_class, Vehicle_movement, Types_of_Junction,
            Service_year_of_vehicle, Cause_of_accident, Driving_experience, Educational_level, Type_of_vehicle, Age_band_of_driver,
            Area_accident_occured, Lanes_or_Medians,  Day_of_week, Light_conditions, Time, Number_of_vehicles_involved, Number_of_casualties]
    
    if ok:
        data = pd.DataFrame( data=[values] ,columns=columns_name )

        data['Day_of_week'] = ore_Day_of_week.transform( data[['Day_of_week']] )
        data['Age_band_of_driver'] = ore_Age_band_of_driver.transform( data[['Age_band_of_driver']] )
        data['Educational_level'] = ore_Educational_level.transform( data[['Educational_level']] )
        data['Driving_experience'] = ore_Driving_experience.transform( data[['Driving_experience']] )
        data['Light_conditions'] = ore_Light_conditions.transform( data[['Light_conditions']] )
        data['Service_year_of_vehicle'] = ore_Service_year_of_vehicle.transform( data[['Service_year_of_vehicle']] )
        data['Age_band_of_casualty'] = ore_Age_band_of_casualty.transform( data[['Age_band_of_casualty']] )

        data = one_hot_encoder_top.transform( data )
    
        predict_value = final_model.predict(data)
        
        if predict_value == 0:
            st.subheader("Severity Prediction : Slight Injury ")
        elif predict_value == 1:
            st.subheader("Severity Prediction : Serious Injury ")
        elif predict_value == 2:
            st.subheader("Severity Prediction : Fatal Injury ")
        else:
            print('error')


        # # plot the force plot in streamlit in API
        # def st_shap(plot, height=None):
        #     shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        #     components.html(shap_html, height=height)

        # st.write("""### Explainable AI (using Shap)""")  
        # shap_values = shap.TreeExplainer(final_model).shap_values(data)
        
        # st_shap(shap.summary_plot(shap_values, data, plot_type="bar") )

        # # shap.initjs()
        # st_shap(shap.force_plot(shap.TreeExplainer(final_model).expected_value[0], shap_values[0][:], data))
        
        # # shap.initjs()
        # st_shap(shap.force_plot(shap.TreeExplainer(final_model).expected_value[0], shap_values[1][10], data.iloc[10]))
