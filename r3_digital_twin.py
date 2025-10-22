import imp
import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
from dataframe import DataFrame
import math
from gis import GIS
from dataframe import DataFrame
import random
import re
from csm_aqua import CSM
import geopandas as gpd
import contextily as cx
import datetime
import pandas as pd
import gymnasium as gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
import pandas as pd
import matplotlib
# Set the backend for matplotlib to avoid GUI issues in some environments
matplotlib.use('Agg')

class Field():
    
    def __init__(self, name, area, sowing_date='10/15', *args, **kwargs):
        self.name = name
        self.area = area
        self.sowing_date = sowing_date
        self.csm_rf = CSM(
            irrigation_method='rainfed',
            sowing_date=sowing_date
        )
        self.csm_et = CSM(
            irrigation_method='et',
            sowing_date=sowing_date
        )
        
        self.csm_rf.simulate()
        self.csm_et.simulate()
        
        self.csm_et.season_state_monitoring_data.drop_column('ks')
        
        self.csm_et.season_state_monitoring_data.add_column('ks', self.csm_rf.season_state_monitoring_data.get_column('ks'))
        self.field_state_monitoring_data = self.csm_et.season_state_monitoring_data.copy()

    def get_ks_threshold_date(self):
        temp_data = self.field_state_monitoring_data.filter_dataframe('ks', lambda x: x<0.7, False)
        temp_dataframe = DataFrame(temp_data, 'df')
        ks_threshold_date = temp_dataframe.get_index()[0]
        return ks_threshold_date
    
    def get_ks_of_date(self, date):
        return self.field_state_monitoring_data.get_column('ks')[date]
    
    def show(self,):
        self.field_state_monitoring_data.show()
        
    def water_needs_till_date(self, date):
        # Sum the column until the given index
        water_needs = self.field_state_monitoring_data.get_column('IrrDay')[:date+datetime.timedelta(1)].sum()
        return water_needs
    
    def seasonal_irrigation(self):
        # Sum the column until the given index
        water_needs = self.field_state_monitoring_data.get_column('IrrDay').sum()
        return water_needs
    
class R3DigitalTwin(gym.Env): 
    
    def __init__(
        self, 
        fields_path="data/r3_dt/fields.parquet", 
        canals_path="data/r3_dt/canals.parquet",
        weather_data_path='data/r3_dt/r3_aws_full.csv',
        start_date_simulation='2015/10/01',
        sowing_date_series=None,
        fitness_threshold=-1000,
        theta=0.5
                 ):
        
        self.fields_path = fields_path
        self.canals_path = canals_path
        self.weather_data_path = weather_data_path
        self.fields = GIS()
        self.canals = GIS()
        self.fields.add_data_layer('fields', self.fields_path, 'parquet')
        self.canals.add_data_layer('canals', self.canals_path, 'parquet')
        self.start_date_simulation = datetime.datetime.strptime(start_date_simulation, "%Y/%m/%d") 
        self.csm = CSM(weather_data_path=weather_data_path, start_date_simulation=start_date_simulation)
        self.sowing_dates_series = sowing_date_series
        self.sow(self.sowing_dates_series)
        self.fitness_threshold = fitness_threshold
        
        # balance between crop yields and water needs
        self.theta = theta
        self.simulation = {}
        
        # RL section
        # Action space: 32 fields with sowing dates between day 0 and day 365
        self.action_space = spaces.Box(low=0, high=61, shape=(32,), dtype=np.float32)
        
        # Observation space: 32 potential yields + 32 water supply deficits (total of 64)
        self.observation_space = spaces.Box(low=0, high=25, shape=(64,), dtype=np.float32)

        # Initialize environment state
        self.state = self.reset()
        
        
    def sow(self, sowing_dates_series=None, start_date_simulation='2015-10-15'):
        if sowing_dates_series is None: 
            # generate random sowing dates starting a given and ending at a given date
            self.fields.add_random_series_column('fields', 'sowing_date', 0, 77)
            self.fields.transform_column('fields', 'sowing_date', 'sowing_date', lambda o: R3DigitalTwin.dap_to_date(o, start_date_simulation))
            #self.layers.add_random_series_column('plots', 'sowing_dates')
        else:
            self.fields.add_column('fields', sowing_dates_series, 'sowing_date')
            self.fields.transform_column('fields', 'sowing_date', 'sowing_date', lambda o: R3DigitalTwin.dap_to_date(o, start_date_simulation))
            #self.layers.add_column('plots', sowing_dates_series, 'sowing_dates')
    def temp(self, sowing_dates_series=None):
        data = GIS()
    
        # read SHP files
        data.add_data_layer('fields', r"Z:\My Folders\crsa\papers\sowing_date_paper_rl\media\study_area\fields.parquet", data_type='parquet')
        data.add_data_layer('canals', r"Z:\My Folders\crsa\papers\sowing_date_paper_rl\media\study_area\canals.parquet", data_type='parquet')
        
        data.rename_columns('canals', {'CANAL': 'canal', 'Debit_Max': 'capacity'})
        data.keep_columns('canals', ['canal', 'capacity'])
        
        # plot fields on map grouped by canals
        #data.show('fields', 'Canal')
        
        data.rename_columns('fields', {'Canal': 'canal'})
        data.keep_columns('fields', ['geometry', 'canal', 'nparc'])
        data.group_by('fields', 'canal')
        
        data.add_area_column('fields', 'm2')
        data.index_to_column('fields')
        
        data.join_layers('fields', 'canals', 'canal')
        
        
        data.show_dataframe('fields')
        data.export('fields', 'data/pinns/fields.parquet', file_format='parquet')
    
    def get_state(self):
        """start_state = np.where(self.grid_state == self.AGENT)
        start_not_found = not (start_state[0] and goal_state[0])
        if start_not_found:
            print("Start state not present in the Gridworld. Check the Grid layout")

        #start_state = (start_state[0][0])
        start_state = 0"""
        
        
        return self.layers.get_column('sowing_dates')
    def first_irrigation_round_date(self, fields_percent=0.05):
        
        #self.fields.data_layers['fields'] = self.fields.data_layers['fields'].sort_values(by='ks_threshold_date')
        self.fields.data_layers['fields'] = self.fields.data_layers['fields'].sort_values(by='ks_threshold_date', inplace=False).reset_index(drop=True)
        

        self.fields.add_column('fields', self.fields.get_column('fields', 'area').cumsum(), 'cumulative_area')
        # Calculate the total area
        total_area = self.fields.get_column('fields', 'area').sum()

        # Calculate 5% of the total area
        threshold = 0.05 * total_area

        # Get the row where cumulative sum exceeds or equals 5% of the total area
        row = self.fields.data_layers['fields'][self.fields.data_layers['fields']['cumulative_area'] >= threshold].iloc[0]
        
        """ 
        # Sample sorted list of dates
        sorted_dates = sorted(self.fields.get_column('fields', 'ks_threshold_date').to_numpy())

        # Determine the length of the list
        n = len(sorted_dates)

        # Calculate the exact 5% index (which may not be an integer)
        index = n * fields_percent

        # Standard rounding
        index_round = round(index)
        print(sorted_dates)
        print(sorted_dates[index_round])

        self.first_irrigation_round_date = index_round"""
        
        return row['ks_threshold_date']
    
    def step(self, action):
        """
        Run one step into the env
        Args:
            state (Any): Current index state of the maze
            action (int): Discrete action for up, down,
            left, right
            slip (bool, optional): Stochasticity in the 
            env. Defaults to True.
        Raises:
            ValueError: If invalid action is provided as 
            input
        Returns:
            Tuple : Next state, reward, done, _
       
            return next observation, reward, done, info
        """
        # Simulate the sowing distribution (action) for 32 fields
        sowing_dates = np.round(action).astype(int)  # Convert to integers and round to nearest value
        print("Action Taken:", sowing_dates)

        # Simulate the season until the first irrigation round and calculate the water supply constraints
        self.sow(sowing_dates)
        self.simulate()
        potential_yields = self.fields.get_column('fields', 'scaled_yield')
        water_deficit = self.fields.get_column('fields', 'scaled_delta_ws_wn')
        
        self.state = np.concatenate([potential_yields, water_deficit])
        
        # Calculate the reward based on the potential yields and water deficit
        reward = self.fitness_sowing_dates_distribution()

        # Define when the season is over (after one round of simulation)
        done = True  # The episode is done at the end of the season
        
        return self.state, reward, done, {}
    
    def get_state_reward(self, action):
        actual_fitness = self.fitness_sowing_dates_distribution()
        random_plot = np.random.randint(0, 33)
        actual_sowing_dates = self.layers.get_column('pipelines', 'sowing_dates').to_numpy()
        #print(actual_sowing_dates)
        if action == self.ADD_DAY:
            if actual_sowing_dates[random_plot] <= 100: 
                actual_sowing_dates[random_plot] += 1
        elif action == self.SUB_DAY:
            if actual_sowing_dates[random_plot] != 0:
                actual_sowing_dates[random_plot] += -1
        elif action == self.SAME_DAY:
            pass
        #print("Taken action:", action)
        self.sow(actual_sowing_dates)
        
        #self.layers.set_row('pipelines', 'sowing_dates', random_plot, actual_sowing_dates)
        self.estimated_cluster_yield()
        reward = self.fitness_sowing_dates_distribution() - actual_fitness
        return self.layers.get_column('pipelines', 'sowing_dates').to_numpy(), reward
    
    def render(self):
        return self.show()
    
    def reset(self):
        # Reset environment state: initialize potential yields and water deficits randomly
        potential_yields = np.zeros((32,))  # Randomized potential yield for each field
        water_deficit = np.zeros((32,))     # Randomized water supply deficit
        self.state = np.concatenate([potential_yields, water_deficit])
        #return self.set_sowing_dates(np.ones((116, 1)))
        return self.state
    
    def get_sowing_dates(self):
        return self.fields.get_column('fields', 'sowing_dates')
    
    def total_yield(self):
        total_yield = self.fields.get_column('fields', 'yield_to').sum()
        return total_yield
    
    def total_actual_yield(self):
        total_yield = self.fields.get_column('fields', 'yield_to_actual').sum()
        return total_yield
    
    def total_water_needs(self):
        total_water_needs = self.fields.get_column('fields', 'water_needs_m3').sum()
        return total_water_needs
        
    def seasonal_irrigation(self):
        total_water_needs = self.fields.get_column('fields', 'seasonal_irrigation_m3').sum()
        return total_water_needs
    
    def total_water_supply(self):
        total_water_needs = self.fields.get_column('fields', 'water_supply_m3').sum()
        
    def deficit_or_network_violation_nbr(self):
        negative_count = (self.fields.get_column('fields', 'delta_ws_wn') < 0).sum()
        return negative_count
    
    def plot_sowing_dates(self):
        self.fig, self.ax = plt.subplots(figsize=(17,17))
        
        # Sample GeoDataFrame with MultiLineString geometry
        gdf = gpd.read_parquet(self.fields_path)
        #gdf = self.fields.data_layers['fields']
        
        #converted_gdf = gdf.to_crs(epsg=3857)
        converted_gdf = gdf
        
        self.ax = converted_gdf.plot(ax=self.ax, alpha=0.5, column='canal')
        
        cx.add_basemap(ax=self.ax, source=cx.providers.Esri.WorldImagery)
        import pandas  as pd
        import pandas as pd
        series = pd.Series(np.random.randint(0, 50, converted_gdf.shape[0]))
        converted_gdf['sowing_date'] = series
        
        
        # Add numbers to the plot
        for _, row in converted_gdf.iterrows():
            centroid = row['geometry'].centroid
            #ax.text(centroid.x, centroid.y, str(row['Numbers']), ha='center', va='center', fontsize=10)
            self.ax.annotate(row['sowing_date'], xy=(centroid.x, centroid.y), xytext=(3, 3),
                        textcoords='offset points', ha='center', va='center', fontsize=9, color='yellow')

        # Set plot title and axis labels
        self.ax.set_title('A spatiotemporal distribution of sowing dates')
        
        
        
    
    def show(self):
        import pandas as pd
        pd.options.display.float_format = '{:.4e}'.format 
        self.fields.show_dataframe('fields')
        self.fields.export_dataframe('fields', 'data/r3_dt/fields_debug.csv')
        # Round all columns to 2 decimal places except 'geometry'
        self.fields.transform_column('fields', 'yield_to', 'yield_to', lambda o: round(o, 2))
        self.fields.transform_column('fields', 'yield_to_actual', 'yield_to_actual', lambda o: round(o, 2))
        self.fields.transform_column('fields', 'water_needs_m3', 'water_needs_m3', lambda o: round(o, 2))
        self.fields.transform_column('fields', 'seasonal_irrigation', 'seasonal_irrigation', lambda o: int(o))
        #print("Sowing dates", self.fields.get_column('fields', 'sowing_date'))
        
        #self.fields.data_layers['fields'] = self.fields.data_layers['fields'].round(2)
        
        #self.fields.export_dataframe('fields', 'data/r3_dt/fields.csv')
        print("Total Area (m2): ", self.fields.get_column('fields', 'area').sum())
        print("Total Yield (ton): ", self.total_yield())
        print("Total Actual Yield (ton): ", self.total_actual_yield())
        print("R1 Water Needs (m3): ", self.total_water_needs())
        print("Seasonal Irrigation (m3): ", self.seasonal_irrigation())
        print("Deficit nbr: ", self.deficit_or_network_violation_nbr())
        #print("Min yield: ", self.fields.get_column('fields', 'yield_to').min())
        #print("Max yield: ", self.fields.get_column('fields', 'yield_to').max())
        # round the dataframe to 2 decimal places for better readability
        
        #self.plot_sowing_dates()
        #plt.show()

        
    def export(self):
        self.fields.export('fields', 'data/r3_dt/fields_viz.parquet', file_format='parquet')
     
        
    def simulate(self, irrigation_round_duration=15, verbose=False):
        seasonal_rainfall_data = DataFrame(self.weather_data_path)
        seasonal_rainfall_data.column_to_date('datetime')
        seasonal_rainfall_data.reindex_dataframe('datetime')
        seasonal_rainfall_data.keep_rows_by_year(2015)
        #seasonal_rainfall_data.select_datetime_range(self.start_date_simulation, self.start_date_simulation + datetime.timedelta(days=273))
        # sum of rainfall column
        #seasonal_rainfall_data.show()
        #print(seasonal_rainfall_data.get_column('p').sum())
        
        #self.fields.add_transformed_columns('fields', 'rainfall', 'area*yield_po')
        
        ks_threshold_dates_series = []
        yield_potentials_series = []
        yield_actual_series = []
        water_needs_series = []
        seasonal_irrigation_series = []
        #print('The simulation of the season is started...')
        for index, field in self.fields.get_data_layer('fields').iterrows():
            
            # verbose
            #print(field['canal'])
            
            # for testing purposes
            """if field['canal'] == 'R3P2-S1T2':
                break"""
                
            # simulate the season for each field
            field_simulation = Field(
                name=field['canal'],
                area=field['area'],
                sowing_date=field['sowing_date'],
            )
            self.simulation[field['canal']] = field_simulation
            ks_threshold_date = field_simulation.get_ks_threshold_date()
            
            # add the date where Ks goes below 0.7 for each field to a list
            ks_threshold_dates_series.append(ks_threshold_date)
            
            
            # estimate the potential yield for each field and convert to ta/m2
            #field_simulation.field_state_monitoring_data.export('data/r3_dt/field_state_monitoring.csv')
            yield_potentials_series.append(max(field_simulation.field_state_monitoring_data.get_column('yield'))*1e-4)
            yield_actual_series.append(max(field_simulation.field_state_monitoring_data.get_column('yield_actual'))*1e-4)
        
        # add the date where Ks goes below 0.7 for each field
        self.fields.add_column('fields', ks_threshold_dates_series, 'ks_threshold_date')
        self.fields.add_column('fields', yield_potentials_series, 'yield_po')
        self.fields.add_column('fields', yield_actual_series, 'yield_actual')
        
        #self.fields.show_dataframe('fields')
        self.fields.add_transformed_columns('fields', 'yield_to', 'area*yield_po')
        self.fields.add_transformed_columns('fields', 'yield_to_actual', 'area*yield_actual')
        
        date_first_round = self.first_irrigation_round_date()
        
        
        #verbose
        #print('The date of the first irrigation round:', date_first_round)
        ks_round1_series = []
        try:
            for index, field in self.fields.get_data_layer('fields').iterrows():
                #print(self.simulation['R3P2'])
                water_needs = self.simulation[field['canal']].water_needs_till_date(date_first_round)
                water_needs_series.append(water_needs)
                seasonal_irrigation_series.append(self.simulation[field['canal']].seasonal_irrigation())
                ks_round1_series.append(self.simulation[field['canal']].get_ks_of_date(date_first_round))
        except Exception as e:
            print(e)
        
        self.fields.add_column('fields', ks_round1_series, 'ks_round1')
        self.fields.add_column('fields', water_needs_series, 'water_needs')
        self.fields.add_column('fields', seasonal_irrigation_series, 'seasonal_irrigation')
        self.fields.add_transformed_columns('fields', 'water_needs_m3', 'area*water_needs*1e-3')
        irrigation_round_duration *= 24 * 3600
        self.fields.add_transformed_columns('fields', 'water_supply_m3', str(irrigation_round_duration) + '*capacity*1e-3')
        self.fields.add_transformed_columns('fields', 'delta_ws_wn', 'water_supply_m3-water_needs_m3')
        self.fields.add_transformed_columns('fields', 'scaled_yield', 'yield_to*1e-3')
        self.fields.add_transformed_columns('fields', 'scaled_delta_ws_wn', 'delta_ws_wn*1e-6')
        #pd.options.display.float_format = '{:.2e}'.format 
        
        
        self.fields.add_transformed_columns('fields', 'seasonal_irrigation_m3', 'area*seasonal_irrigation*1e-3')
        self.fields.add_transformed_columns('fields', 'scaled_seasonal_irrigation_m3', 'seasonal_irrigation_m3*1e-6')
    
    def fitness_sowing_dates_distribution(self):
        
        #self.fields.add_transformed_columns('fields', 'scaled_yield', 'yield_to*1e-3')
        #self.fields.add_transformed_columns('fields', 'scaled_delta_ws_wn', 'delta_ws_wn*1e-5')
        yield_term = self.fields.get_column('fields', 'scaled_yield').sum()
        deficit_ws_wn_term = self.fields.get_column('fields', 'scaled_delta_ws_wn').sum()
        scaled_seasonal_irrigation_m3_term = self.fields.get_column('fields', 'scaled_seasonal_irrigation_m3').sum()
        #print('scaled seasonal irrigation: ', scaled_seasonal_irrigation_m3_term)
        
        #print('yield term:', yield_term, 'deficit term:', deficit_ws_wn_term)
        
        yield_actual = self.fields.get_column('fields', 'yield_to_actual').sum() * 1000
        seasonal_irrigation = self.fields.get_column('fields', 'seasonal_irrigation_m3').sum() 
        
        water_use_efficiency = yield_actual / seasonal_irrigation 
        
        #fitness_term = self.theta * yield_term - (1 - self.theta) * deficit_ws_wn_term
        #fitness_term = yield_term - ( (abs(deficit_ws_wn_term) / 3) + scaled_seasonal_irrigation_m3_term)
        fitness_term = water_use_efficiency - (0.5 * abs(deficit_ws_wn_term))
        
        #print('fitness term:', fitness_term)
        #print('yield term:', yield_term)
        print('Fitness term:', fitness_term)
        #print('deficit term:', deficit_ws_wn_term)
        print('WUE:', water_use_efficiency)
        #print('scaled seasonal irrigation term:', scaled_seasonal_irrigation_m3_term)
        
        #print('fitness term:', fitness_term)
        
        
        return fitness_term
            
    def verify_irrigation_network_constraints(self):
        remaining_list = []
        delta_list = self.get_delta_list()
        for list_voisin in delta_list:
            list_score = 0
            for canal in list_voisin:
                splited_canal = re.findall('[A-Z]*\d+', canal)
                next_branch = ""
                total_remaining = 0
                for p in splited_canal:
                    temp_canals_list = list_voisin
                    temp_canals_list = list(map(lambda x: re.sub('-', '', x), temp_canals_list))
                    remaining = 0
                    next_branch += p
                    if self.layers.get_data_layer('pipelines')[self.layers.get_column('pipelines', 'canal_id_i') == next_branch]['capacity'].shape[0] > 0:
                        #print('verify ', next_branch)
                        if next_branch in temp_canals_list:
                            temp_canals_list.remove(next_branch)
                        common_canal_capacity = self.layers.get_data_layer('pipelines')[self.layers.get_column('pipelines', 'canal_id_i') == next_branch].capacity.to_numpy()[0]
                        activated_canals_sum = 0
                        for q in temp_canals_list:
                            if next_branch in q:
                                activated_canals_sum += self.layers.get_data_layer('pipelines')[self.layers.get_column('pipelines', 'canal_id_i') == q].capacity.to_numpy()[0]
                        
                        remaining = common_canal_capacity - activated_canals_sum
                        total_remaining += remaining
                list_score += total_remaining
                #print("List score:", list_score) 
            
            remaining_list.append(list_score)
            self.layers.add_column('pipelines', remaining_list, 'remaining')
            #print("Total remaining:", total_remaining)
            
    @staticmethod
    def dap_to_date(dap_date, start_date_simulation='2015-10-15'):
        start_date_simulation_temp = datetime.datetime.strptime(start_date_simulation, "%Y-%m-%d")
        date = start_date_simulation_temp + datetime.timedelta(days=dap_date)
        # return month and day formated 'month/day'
        return date.strftime('%m/%d')
    