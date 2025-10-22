import numpy as np
import time
import sys
import math
from numpy.linalg import matrix_power
from nltk.corpus import stopwords
import nltk
import lib
from dataframe import DataFrame
from vectorizer import Vectorizer
import matplotlib.pyplot as plt
from r3_digital_twin import R3DigitalTwin, Field
from csm_aqua import CSM
import datetime
import gymnasium as gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.noise import NormalActionNoise
import torch
import os
from datetime import timedelta
#print(torch.cuda.is_available())  # Should return True if GPU is available
from datetime import datetime


def main():
    ti = time.time()
    
    variable = 'temperature_2m'
    output_dir="era5_land_africa_eto"
    start_date='2011-01-01' 
    end_date='2024-01-01'
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    #date_str = image.date().format('yyyyMMdd').getInfo()
    #out_name = f"{variable}_{date_str}.tif"
    """out_path = os.path.join(output_dir, out_name)
    if os.path.exists(out_path):
        print(f"Data already exists, skipping: {output_dir}/{out_name}")"""
    
    """data = DataFrame(r"data\r3_dt\field_state_monitoring.csv")
    data.column_to_date('date')
    data.reindex_dataframe('date')
    print(data.get_column('IrrDay')[:datetime.datetime(2016, 2, 6)+datetime.timedelta(1)].sum())"""
    
    
    """csm = CSM(
        weather_data_path='data/r3_dt/r3_aws_full.csv',
        irrigation_method='et'
        )
    csm.simulate()
    csm.monitor()"""
    
    # 26-01-2016
    """field = Field(
        name='R3P2',
        area=1,
        sowing_date='11/15',
        )
    date = datetime.datetime(2016, 2, 1)
    
    
    print(field.get_ks_of_date(date))"""
    #print(field.water_needs_till_date(date))
    #print(field.seasonal_irrigation())
    #start_date_simulation = field.csm_et.start_date_simulation
    #sowing_date = field.csm_et.sowing_date
    #field.field_state_monitoring_data.show()
    #field.field_state_monitoring_data.select_datetime_range(sowing_date, )
    
    
    #temp_dataframe.export('data/r3_dt/field_state_monitoring.csv')
    
    # 1 Nov
    #sowing_dates_series = np.zeros((32,))
    
    # 1 Oct
    #sowing_dates_series = np.full((32,), 17)
    #sowing_dates_series  = [(3*i) + 10 for i in range(0, 32)]
    
    # RL
    sowing_dates_series = [39, 35, 34, 33, 37, 26, 32, 32, 30, 28, 34, 31, 30, 29, 28, 34, 27, 34, 28, 32, 33, 33, 31, 34, 28, 33, 37, 33, 27, 29, 38, 30]
    # lagrange
    #sowing_dates_series  = [19, 26, 17, 18, 15, 38, 43, 52, 17, 27, 41, 33, 37, 29, 23, 31, 28, 47, 39, 21, 30, 14, 10, 10, 47, 40,  7, 19, 10, 31,  8, 42]
    #sowing_dates_series  = [19, 28, 19, 21, 12, 49, 44, 48, 22, 32, 37, 30, 33, 24, 22, 36, 17, 46, 41, 16, 28, 14, 11, 17, 47, 39,  9, 22,  9, 39, 17, 38,]
    #sowing_dates_series  = [16, 27, 16, 17, 15, 48, 42, 46, 23, 29, 43, 37, 30, 28, 33, 27, 25, 45, 39, 19, 28, 13,  7, 16, 43, 43,  8, 17, 18, 40, 16, 39]
    #sowing_dates_series  = [18, 27, 17, 14,  9, 45, 43, 51, 32, 30, 35, 39, 29, 27, 22, 32, 19, 43, 39, 16, 29, 17, 10, 17, 46, 35, 11, 13, 13, 40, 11, 36]
    #sowing_dates_series  = [19, 32, 10, 20, 14, 46, 41, 48, 22, 32, 37, 36, 32, 23, 21, 29, 18, 47, 34, 20, 29, 16, 13, 16, 45, 42, 11, 21, 17, 39, 14, 41]
    #sowing_dates_series  = [6, 54, 56, 41, 38, 47, 54, 39, 14, 35, 56, 53, 53, 26, 17, 59, 42, 58, 41, 48, 34, 10, 57, 59, 10, 56, 37,  1, 54, 18, 14, 60]
    
    
 
 
    # GA
    #sowing_dates_series = ['12/21', '12/12', '12/22', '11/25', '11/08', '12/12', '11/25', '12/04', '12/18', '12/30', '12/29', '11/22', '12/02', '12/31', '11/19', '12/10', '12/15', '11/01', '12/12', '12/23', '12/13', '11/28', '11/29', '12/30', '11/25', '11/09', '12/20', '11/23', '11/13', '12/06', '12/16', '12/11']
    #sowing_dates_series = ['11/06', '11/03', '12/07', '12/21', '12/22', '11/15', '11/04', '12/02', '11/16', '12/18', '12/31', '11/28', '12/25', '11/08', '12/14', '12/10', '11/14', '11/14', '12/22', '12/19', '11/29', '12/22', '12/17', '12/04', '12/24', '12/18', '11/09', '12/13', '12/06', '11/08', '12/24', '11/16']
    #sowing_dates_series = ['11/11', '11/17', '12/07', '12/25', '12/20', '11/18', '12/23', '11/21', '11/02', '11/03', '12/01', '12/21', '11/30', '11/17', '12/11', '12/25', '11/15', '11/12', '12/08', '11/06', '12/05', '12/14', '12/12', '11/18', '11/21', '11/05', '11/05', '11/07', '11/05', '11/26', '11/13', '11/12']
    
    r3 = R3DigitalTwin(
        sowing_date_series=sowing_dates_series
        )
    print(r3.fields.show_dataframe('fields'))
    print("Total Area (m2): ", r3.fields.get_column('fields', 'area').sum())
    r3.simulate()
    print(r3.fitness_sowing_dates_distribution())
    print(r3.show())
    
    #r3.export()
    
    
    #r3.fields.export('fields', 'data/r3_dt/field_state_monitoring.pqrauet', 'parquet')
    #r3.fields.export_dataframe('fields', 'data/r3_dt/field_state_monitoring.csv')
    #r3.show()
    #r3.fitness_sowing_dates_distribution()
    #random_action = np.zeros((32,))
    #r3.step(random_action)
    """r3.fields.transform_column('fields', 'area', 'area', lambda x: x*1e-4)
    r3.show()
    print('okkkkkkkkkkkkkkkkk')
    
    # Define the action noise (helps with exploration in continuous action spaces)
    n_actions = r3.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Create the DDPG agent
    model = DDPG("MlpPolicy", r3, action_noise=action_noise, verbose=1, device='cuda')
    
    # Initialize list to store rewards for plotting
    rewards = []

    # Custom callback function to track rewards during training
    def custom_callback(_locals, _globals):
        #print(_locals)
        episode_reward = _locals['rewards']
        rewards.append(episode_reward)  # Store the reward for each episode
        return True
    
    # Train the agent
    model.learn(total_timesteps=1000, callback=custom_callback)
    
    # Save the trained model
    model.save("data/ddpg_irrigation")
    
    # Test the trained model
    obs = r3.reset()
    rewards_testing = []
    for _ in range(3):
        action, _states = model.predict(obs)
        obs, reward, done, info = r3.step(action)
        rewards_testing.append(reward)
        if done:
            break"""
        
    print(time.time() - ti)


if __name__ == '__main__':
    main()
