import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('train.csv')
data=data.drop(['ID','device','site_latitude','site_longitude','date'],axis=1)
data=data.drop(['CarbonMonoxide_sensor_azimuth_angle',
       'CarbonMonoxide_sensor_zenith_angle',
       'CarbonMonoxide_solar_azimuth_angle',
       'CarbonMonoxide_solar_zenith_angle',
       'NitrogenDioxide_sensor_azimuth_angle',
       'NitrogenDioxide_sensor_zenith_angle',
       'NitrogenDioxide_solar_azimuth_angle',
       'NitrogenDioxide_solar_zenith_angle',
       'Formaldehyde_solar_zenith_angle',
       'Formaldehyde_solar_azimuth_angle',
       'Formaldehyde_sensor_zenith_angle',
       'Formaldehyde_sensor_azimuth_angle',
       'UvAerosolIndex_sensor_azimuth_angle',
       'UvAerosolIndex_sensor_zenith_angle',
       'UvAerosolIndex_solar_azimuth_angle',
       'UvAerosolIndex_solar_zenith_angle',
       'Ozone_sensor_azimuth_angle',
       'Ozone_sensor_zenith_angle',
       'Ozone_solar_azimuth_angle',
       'Ozone_solar_zenith_angle',
       'Cloud_sensor_azimuth_angle',
       'Cloud_sensor_zenith_angle',
       'Cloud_solar_azimuth_angle',
       'Cloud_solar_zenith_angle'],axis=1)
data=data.drop(['Cloud_cloud_fraction', 'Cloud_cloud_top_pressure',
       'Cloud_cloud_top_height', 'Cloud_cloud_base_pressure',
       'Cloud_cloud_base_height', 'Cloud_cloud_optical_depth',
       'Cloud_surface_albedo'],axis=1)

from sklearn.impute import KNNImputer
imputer=KNNImputer(n_neighbors=12,weights='distance',)
data=imputer.fit_transform(data)
data=pd.DataFrame(data)

data.columns=['humidity', 'temp_mean', 'SulphurDioxide_SO2_column_number_density',
       'SulphurDioxide_SO2_column_number_density_amf',
       'SulphurDioxide_SO2_slant_column_number_density',
       'SulphurDioxide_cloud_fraction', 'SulphurDioxide_sensor_azimuth_angle',
       'SulphurDioxide_sensor_zenith_angle',
       'SulphurDioxide_solar_azimuth_angle',
       'SulphurDioxide_solar_zenith_angle',
       'SulphurDioxide_SO2_column_number_density_15km',
       'CarbonMonoxide_CO_column_number_density',
       'CarbonMonoxide_H2O_column_number_density',
       'CarbonMonoxide_cloud_height', 'CarbonMonoxide_sensor_altitude',
       'NitrogenDioxide_NO2_column_number_density',
       'NitrogenDioxide_tropospheric_NO2_column_number_density',
       'NitrogenDioxide_stratospheric_NO2_column_number_density',
       'NitrogenDioxide_NO2_slant_column_number_density',
       'NitrogenDioxide_tropopause_pressure',
       'NitrogenDioxide_absorbing_aerosol_index',
       'NitrogenDioxide_cloud_fraction', 'NitrogenDioxide_sensor_altitude',
       'Formaldehyde_tropospheric_HCHO_column_number_density',
       'Formaldehyde_tropospheric_HCHO_column_number_density_amf',
       'Formaldehyde_HCHO_slant_column_number_density',
       'Formaldehyde_cloud_fraction', 'UvAerosolIndex_absorbing_aerosol_index',
       'UvAerosolIndex_sensor_altitude', 'Ozone_O3_column_number_density',
       'Ozone_O3_column_number_density_amf',
       'Ozone_O3_slant_column_number_density',
       'Ozone_O3_effective_temperature', 'Ozone_cloud_fraction',
       'pm2_5']


Q1=np.percentile(data['pm2_5'],25)
Q3=np.percentile(data['pm2_5'],75)   
data=data[data['pm2_5']<=120]

x=data.drop('pm2_5',axis=1)
y=data['pm2_5']

from sklearn.preprocessing import StandardScaler
std=StandardScaler()
std.fit(x)


import pickle
with open('scaler_pkl', 'wb') as files:
    pickle.dump(std, files)

x = std.transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from catboost import CatBoostRegressor

model = CatBoostRegressor(
    iterations=900,
    learning_rate=0.1,
    loss_function='RMSE',
    depth=8,
    l2_leaf_reg=1,
    max_ctr_complexity=3,
    subsample=0.9,
    colsample_bylevel=0.9
)

model.fit(x_train,y_train,verbose=100)
y_predictions = model.predict(x_test)


from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(y_test, y_predictions)
print("Mean Squared Error:", mse)
print('R squared value',r2_score(y_test,y_predictions))

import pickle
with open('model_pkl_c', 'wb') as files:
    pickle.dump(model, files)
