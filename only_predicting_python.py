import pandas as pd
import pickle


def predict_without_columns(X, day,month,year,temp,rain,wind,sun,day_of_the_week,location,parken_event, parken_type, general_event, rf, gb, hgb,xgr, lgbr):
    new_row = {name:0 for name in X}
    new_row['Day'] = day
    new_row['Month'] = month
    new_row['Year'] = year
    new_row['Temperature'] = temp
    new_row['Wind'] = wind
    new_row['Rain'] = rain
    new_row['Sun'] = sun
    day_of_the_week = 'Day_of_the_week_' + day_of_the_week
    new_row[day_of_the_week] = 1
    location = 'Location_' + location
    new_row[location] = 1
    parken_event = 'Parken_Event_' + parken_event
    new_row[parken_event] = 1
    parken_type = 'Parken_Type_' + parken_type
    new_row[parken_type] = 1
    general_event = 'General_Event_' + general_event
    new_row[general_event] = 1
    new_pd = pd.DataFrame([new_row])

    prediction_rf = rf.predict(new_pd)
    prediction_xgr = xgr.predict(new_pd)
    prediction_gb = gb.predict(new_pd)
    prediction_hgb = hgb.predict(new_pd)
    prediction_lgbr = lgbr.predict(new_pd)
    mean_pred = (prediction_gb + prediction_hgb + prediction_lgbr + prediction_rf + prediction_xgr) / 5
    return mean_pred[0]

def predict(day,month,year, temp,rain,wind,sun, day_of_the_week, location, parken_event,parken_type, parken_place, general_event):
    
    model_rf_file = "model_rf.pkl"
    model_gb_file = "model_gb.pkl"
    model_hgb_file = "model_hgb.pkl"
    model_xgr_file = "model_xgr.pkl"
    model_lgbr_file = "model_lgbr.pkl"
    # load model from pickle file
    with open(model_rf_file, 'rb') as file:  
        rf = pickle.load(file)
    with open(model_gb_file, 'rb') as file:  
        gb = pickle.load(file)
    with open(model_hgb_file, 'rb') as file:  
        hgb = pickle.load(file)
    with open(model_xgr_file, 'rb') as file:  
        xgr = pickle.load(file)
    with open(model_lgbr_file, 'rb') as file:  
        lgbr = pickle.load(file)
    X = ['Day', 'Month', 'Year', 'Temperature', 'Rain', 'Wind', 'Sun','Day_of_the_week_Friday', 'Day_of_the_week_Monday','Day_of_the_week_Saturday', 'Day_of_the_week_Sunday','Day_of_the_week_Thursday', 'Day_of_the_week_Tuesday','Day_of_the_week_Wednesday', 'Location_DGH', 'Location_HDM','Location_HPNS', 'Location_LGV', 'Location_NHG', 'Location_VDV','Parken_Event_Concert', 'Parken_Event_None', 'Parken_Event_football','Parken_Type_Local', 'Parken_Type_Nacional', 'Parken_Type_None','Parken_Type_Otro', 'General_Event_Cultural', 'General_Event_Fashion','General_Event_Festivitie', 'General_Event_None','General_Event_Social', 'General_Event_Sport','General_Event_Worldwide']    
    actual_prediction = predict_without_columns(X, day,month,year,temp,rain,wind,sun,day_of_the_week,location,parken_event, parken_type, general_event, rf, gb, hgb,xgr, lgbr)
    return actual_prediction


day = 12
month = 2
year = 2024
temp = 1.8
rain = 5.6
wind = 2.4
sun = 0
day_of_the_week = 'Monday'
# HDM, HPNS, LGV, NHG, VDV
location = 'VDV'
parken_event = 'None'
parken_type = 'None'
parken_place = 'None'
general_event = 'None'

final_prediction = predict(day,month,year, temp,rain,wind,sun, day_of_the_week, location, parken_event,parken_type, parken_place, general_event)
print(final_prediction)














import gradio as gr


# launch a gradio interface
gr.Interface(predict, "image", "image").launch()