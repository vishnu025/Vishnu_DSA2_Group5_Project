from flask import Flask, render_template, request, jsonify, session
import numpy as np
import pandas as pd
import csv
import os
import random
import pickle
from catboost import CatBoostRegressor
import sklearn



scaler =pickle.load(open('scaler_pkl', 'rb'))
model=pickle.load(open('model_pkl_c', 'rb'))

app = Flask(__name__)
secret_key = os.urandom(16)
app.secret_key = secret_key


@app.route('/')
def home():
    session.pop('row', None)
    session.pop('posting_row', None)

    return render_template('home.html')


@app.route('/readings', methods=['GET','POST'])
def readings():
    if request.method == "GET":

        with open('static\quick.csv', 'r') as w:
            reader = csv.DictReader(w)
            rows = list(reader)
            row_dict = random.choice(rows)
            row_DF = pd.DataFrame.from_dict([row_dict])
            del row_DF["ID"]
            del row_DF[ "device"]
            del row_DF[ "site_latitude"]
            del row_DF[ "site_longitude"]
            del row_DF[ ""]

        output = model.predict(row_DF)

        return render_template('/home.html',
        
            r4=row_DF["humidity"][0],
            r5=row_DF["temp_mean"][0],
            r6=row_DF["SulphurDioxide_SO2_column_number_density"][0],
            r7=row_DF["SulphurDioxide_SO2_column_number_density_amf"][0],
            r8=row_DF["SulphurDioxide_SO2_slant_column_number_density"][0],
            r9=row_DF["SulphurDioxide_cloud_fraction"][0],
            r10=row_DF["SulphurDioxide_sensor_azimuth_angle"][0],
            r11=row_DF["SulphurDioxide_sensor_zenith_angle"][0],
            r12=row_DF["SulphurDioxide_solar_azimuth_angle"][0],
            r13=row_DF["SulphurDioxide_solar_zenith_angle"][0],
            r14=row_DF["SulphurDioxide_SO2_column_number_density_15km"][0],
            r15=row_DF["CarbonMonoxide_CO_column_number_density"][0],
            r16=row_DF["CarbonMonoxide_H2O_column_number_density"][0],
            r17=row_DF["CarbonMonoxide_cloud_height"][0],
            r18=row_DF["CarbonMonoxide_sensor_altitude"][0],
            r19=row_DF["NitrogenDioxide_NO2_column_number_density"][0],
            r20=row_DF["NitrogenDioxide_tropospheric_NO2_column_number_density"][0],
            r21=row_DF["NitrogenDioxide_stratospheric_NO2_column_number_density"][0],
            r22=row_DF["NitrogenDioxide_NO2_slant_column_number_density"][0],
            r23=row_DF["NitrogenDioxide_tropopause_pressure"][0],
            r24=row_DF["NitrogenDioxide_absorbing_aerosol_index"][0],
            r25=row_DF["NitrogenDioxide_cloud_fraction"][0],
            r26=row_DF["NitrogenDioxide_sensor_altitude"][0],
            r27=row_DF["Formaldehyde_tropospheric_HCHO_column_number_density"][0],
            r28=row_DF["Formaldehyde_tropospheric_HCHO_column_number_density_amf"][0],
            r29=row_DF["Formaldehyde_HCHO_slant_column_number_density"][0],
            r30=row_DF["Formaldehyde_cloud_fraction"][0],
            r31=row_DF["UvAerosolIndex_absorbing_aerosol_index"][0],
            r32=row_DF["UvAerosolIndex_sensor_altitude"][0],
            r33=row_DF["Ozone_O3_column_number_density"][0],
            r34=row_DF["Ozone_O3_column_number_density_amf"][0],
            r35=row_DF["Ozone_O3_slant_column_number_density"][0],
            r36=row_DF["Ozone_O3_effective_temperature"][0],
            r37=row_DF["Ozone_cloud_fraction"][0],
            prediction=output[0], value2=output[0])




@app.route('/userreadings', methods=['GET','POST'])
def userreadings():

    if request.method == "POST":
        s4 =float(request.form['s4'])
        s5= float(request.form['s5'])
        s6= float(request.form['s6'])
        s7= float(request.form['s7'])
        s8= float(request.form['s8'])
        s9= float(request.form['s9'])
        s10= float(request.form['s10'])
        s11= float(request.form['s11'])
        s12= float(request.form['s12'])
        s13= float(request.form['s13'])
        s14= float(request.form['s14'])
        s15= float(request.form['s15'])
        s16= float(request.form['s16'])
        s17= float(request.form['s17'])
        s18= float(request.form['s18'])
        s19= float(request.form['s19'])
        s20= float(request.form['s20'])
        s21= float(request.form['s21'])
        s22= float(request.form['s22'])
        s23= float(request.form['s23'])
        s24= float(request.form['s24'])
        s25= float(request.form['s25'])
        s26= float(request.form['s26'])
        s27= float(request.form['s27'])
        s28= float(request.form['s28'])
        s29= float(request.form['s29'])
        s30= float(request.form['s30'])
        s31= float(request.form['s31'])
        s32= float(request.form['s32'])
        s33= float(request.form['s33'])
        s34= float(request.form['s34'])
        s35= float(request.form['s35'])
        s36= float(request.form['s36'])
        s37= float(request.form['s37'])


        user_input = [s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,s25,
        s26,s27,s28,s29,s30,s31, s32, s33, s34, s35, s36, s37 ]

        user_output = model.predict([[s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,s25,
        s26,s27,s28,s29,s30,s31, s32, s33, s34, s35, s36, s37 ]])

        return render_template('/home.html',user_prediction=user_output[0], user_input = user_input)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
