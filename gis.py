import numpy as np
from numpy.linalg import matrix_power
from dataframe import DataFrame
from rl import *
import pandas as pd
from lib import Lib
import contextily as cx
from matplotlib import pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import xarray as xr

class GIS:
    """
    GIS class
    """
    def __init__(self):
        self.data_layers = {}
        self.fig, self.ax = plt.subplots(figsize=(17,17))

    def add_data_layer(self, layer_name, data_path, data_type='sf', lon_column_name=None, lat_column_name=None, crs='4326'):
        if data_type == 'gdf':
            self.data_layers[layer_name] = data_path
        if data_type == 'lon_lat_csv':
            data = DataFrame(data_path)
            geo_data = GIS.lon_lat_dataframe_to_geopandas(data.get_dataframe(), lon_column_name, lat_column_name, crs=crs)
            self.data_layers[layer_name] = geo_data
        elif data_type == 'shp':
            self.data_layers[layer_name] = gpd.read_file(data_path)
        elif data_type == 'parquet':
            self.data_layers[layer_name] = gpd.read_parquet(data_path)
        elif data_type == 'geojson':
            self.data_layers[layer_name] = gpd.read_file(data_path)
        
    def get_data_layer(self, layer_name):
        return self.data_layers.get(layer_name)
    
    def get_shape(self, layer_name):
        return self.data_layers[layer_name].shape
    
    def set_row(self, layer_name, column_name, row_index, new_value):
        if isinstance(row_index, int):
            self.data_layers[layer_name][column_name].iloc[row_index] = new_value
        self.data_layers[layer_name][column_name].loc[row_index] = new_value
    
    def add_random_series_column(self, layer_name, column_name='random',min=0, max=100, distribution_type='random', mean=0, sd=1):
        if distribution_type == 'random':
            series = pd.Series(np.random.randint(min, max, self.get_shape(layer_name)[0]))
        elif distribution_type == 'standard_normal':
            series = pd.Series(np.random.standard_normal(self.get_shape(layer_name)[0]))
        elif distribution_type == 'normal':
            series = pd.Series(np.random.normal(mean, sd, self.get_shape(layer_name)[0]))
        else:
            series = pd.Series(np.random.randn(self.get_shape(layer_name)[0]))
        self.add_column(layer_name, series, column_name)
    
    def join_layers(self, left_layer, right_layer, on, how='inner'):
        self.data_layers[left_layer] = self.data_layers.get(left_layer).merge(self.data_layers.get(right_layer), on=on, how=how)
    def column_to_list(self, layer_name, column_name, verbose=True):
        column_as_list = self.get_column(layer_name, column_name).tolist()
        if verbose is True:
            print(column_as_list)
        return column_as_list
        
    def add_one_value_column(self, layer_name, column_name, value, length=None):
        """
        Add a column with a single repeated value to the specified GIS layer.
        
        Parameters:
        -----------
        layer_name : str
            Name of the layer to add the column to
        column_name : str
            Name of the new column
        value : any
            Value to fill the column with (can be numeric, string, datetime, etc.)
        length : int, optional
            Length of the column. If None, uses the layer's row count.
            
        Returns:
        --------
        GeoDataFrame
            The modified geodataframe
        """
        if layer_name not in self.data_layers:
            raise ValueError(f"Layer '{layer_name}' not found in data_layers")
        
        # Get the layer's geodataframe
        gdf = self.data_layers[layer_name]
        
        # Determine the length of the column
        if length is None:
            length = len(gdf)
        
        # For non-numeric values like datetime, we can't use np.zeros + fill
        # Instead, create a pandas Series with the repeated value
        gdf[column_name] = pd.Series([value] * length, index=gdf.index)
        
        return gdf
    
    def show(self, layer_name, column4color=None, color=None, alpha=0.5, legend=False, 
             figsize_tuple=(16,9), cmap=None, interactive_mode=False, save_fig=False, savefig_path='out.png', **kwargs):
        """_summary_

        Args:
            layer_name (_type_): _description_
            column4color (_type_, optional): _description_. Defaults to None.
            color (_type_, optional): _description_. Defaults to None.
            alpha (float, optional): _description_. Defaults to 0.5.
            legend (bool, optional): _description_. Defaults to False.
            figsize_tuple (tuple, optional): _description_. Defaults to (15,10).
            cmap (str, optional): example: 'Reds' for heatmaps. Defaults to None.
        """
        
        fig, ax = plt.subplots(figsize=figsize_tuple)
        
        layer = self.data_layers.get(layer_name).to_crs(epsg=3857)
        ax = layer.plot(ax=ax, alpha=alpha, edgecolor='k', color=color, legend=legend, 
                   column=column4color, figsize=figsize_tuple, cmap=cmap, **kwargs)
        cx.add_basemap(ax=ax, source=cx.providers.Esri.WorldImagery, crs=layer.crs.to_string(), attribution=False)
        
        
        if save_fig is True:
            # Adjust the position of the tiles to appear in one line at the bottom
            ax.set_anchor('SW')  # Set the anchor point to the southwest (bottom left) corner
            
            # Adjust the plot limits to cover the entire image
            ax.set_xlim(layer.total_bounds[0], layer.total_bounds[2])
            ax.set_ylim(layer.total_bounds[1], layer.total_bounds[3])
            
            import matplotlib as mpl
            mpl.rcParams['agg.path.chunksize'] = 10000
            fig.savefig(savefig_path, dpi=720, bbox_inches='tight')
                    
        if interactive_mode is True: 
            return self.data_layers.get(layer_name).explore()
        else:
            ax.set_aspect('equal')
            plt.show()
        
    def get_crs(self, layer_name):
        """
        Cordonate Reference System
        EPSG: european petroleum survey group
        """
        return self.get_data_layer(layer_name).crs
    
    def reorder_columns(self, layer_name, new_order_as_list):
        self.data_layers[layer_name].reindex_axis(new_order_as_list, axis=1)
        
    
    def export_dataframe(self, layer_name, destination_path='dataframe.csv', type='csv', index=True):
        """
        Export a GIS layer as a regular DataFrame (without geometry) to a file.
        
        Parameters:
        -----------
        layer_name : str
            Name of the layer to export
        destination_path : str
            Path to save the file
        type : str, default 'csv'
            Format to export ('csv', 'json', or 'pkl')
        index : bool, default True
            Whether to include the index in the output
        """
        if layer_name not in self.data_layers:
            print(f"❌ Error: Layer '{layer_name}' not found")
            return
            
        try:
            # Make a copy so we don't modify the original layer
            df = self.data_layers[layer_name].copy()
            
            # Drop the geometry column to get only the DataFrame
            if 'geometry' in df.columns:
                df = df.drop('geometry', axis=1)
            
            if type == 'json':
                df.to_json(destination_path)
            elif type == 'csv':
                df.to_csv(destination_path, index=index)
            elif type == 'pkl':
                df.to_pickle(destination_path)
            else:
                print(f"❌ Error: Unsupported file format '{type}'")
                return
                
            print(f"✅ Successfully exported layer '{layer_name}' as DataFrame to {destination_path} in {type} format")
            
        except Exception as e:
            print(f"❌ Error exporting DataFrame from layer '{layer_name}': {e}")
    
    def export(self, layer_name, file_name, file_format='geojson'):
        """
        Export a GIS layer to a file in the specified format.
        
        Parameters:
        -----------
        layer_name : str
            Name of the layer to export
        file_name : str
            Path to save the file
        file_format : str, default 'geojson'
            Format to export ('geojson', 'shapefile', or 'parquet')
        """
        if layer_name not in self.data_layers:
            print(f"❌ Error: Layer '{layer_name}' not found")
            return
            
        try:
            if file_format == 'geojson':
                self.data_layers[layer_name].to_file(file_name, driver='GeoJSON')
            elif file_format == 'shapefile':
                self.data_layers[layer_name].to_file(file_name, driver='ESRI Shapefile')
            elif file_format == 'parquet':
                self.data_layers[layer_name].to_parquet(file_name)
            else:
                print(f"❌ Error: Unsupported file format '{file_format}'")
                return
                
            print(f"✅ Successfully exported layer '{layer_name}' to {file_name} in {file_format} format")
            
        except Exception as e:
            print(f"❌ Error exporting layer '{layer_name}': {e}")
            
    def to_crs(self, layer_name, epsg="4326"):
        self.data_layers[layer_name] = self.data_layers[layer_name].to_crs(epsg)
        
    def set_crs(self, layer_name, epsg="4326"):
        self.data_layers[layer_name] = self.data_layers[layer_name].set_crs(epsg)
        
    def show_points(self, x_y_csv_path, crs="4326"):
        pass
    
    def show_point(self, x_y_tuple, crs="4326"):
        pass
    
    def add_point(self, x_y_tuple, layer_name, crs="4326"):
        point = Point(0.0, 0.0)
        #self.__dataframe = self.get_dataframe().append(row_as_dict, ignore_index=True)
        row_as_dict = {'geometry': point}
        self.data_layers[layer_name].append(row_as_dict, ignore_index=True)
    
    def new_data_layer(self, layer_name, crs="EPSG:3857"):
        self.data_layers[layer_name] = gpd.GeoDataFrame(crs=crs)
        self.data_layers[layer_name].crs = crs
        
    def add_column(self, layer_name, column, column_name):
        y = column
        if (not isinstance(column, pd.core.series.Series or not isinstance(column, pd.core.frame.DataFrame))):
            y = np.array(column)
            y = np.reshape(y, (y.shape[0],))
            y = pd.Series(y)
        self.data_layers[layer_name][column_name] = y
        
    def show_dataframe(self, layer_name, number_of_row=None):
        if number_of_row is None:
            print(self.get_data_layer(layer_name))
        elif number_of_row < 0:
            return self.get_data_layer(layer_name).tail(abs(number_of_row)) 
        else:
            return self.get_data_layer(layer_name).head(number_of_row) 
        
    def add_row(self, layer_name, row_as_dict):
        self.data_layers[layer_name] = self.get_data_layer(layer_name).append(row_as_dict, ignore_index=True)
    
    def get_row(self, layer_name, row_index, column=None):
        if column is not None:
            return self.data_layers[layer_name].loc[self.data_layers[layer_name][column] == row_index].reset_index(drop=True)
        return self.data_layers[layer_name].iloc[row_index]
    
    def get_layer_shape(self, layer_name):
        """
        return (Number of lines, number of columns)
        """
        return self.data_layers[layer_name].shape
    
    def get_columns_names(self, layer_name):
        header = list(self.data_layers[layer_name].columns)
        return header 
    
    def drop_column(self, layer_name, column_name):
        """Drop a given column from the dataframe given its name

        Args:
            column (str): name of the column to drop

        Returns:
            [dataframe]: the dataframe with the column dropped
        """
        self.data_layers[layer_name] = self.data_layers[layer_name].drop(column_name, axis=1)
        return self.data_layers[layer_name]
    
    def keep_columns(self, layer_name, columns_names_as_list):
        for p in self.get_columns_names(layer_name):
            if p not in columns_names_as_list:
                self.data_layers[layer_name] = self.data_layers[layer_name].drop(p, axis=1)
    def group_by(self, layer_name, group_by_column_name, agg_func='sum'):
        # Dissolve the fields by the 'canal' column
        self.data_layers[layer_name] = self.data_layers[layer_name].dissolve(by=group_by_column_name, aggfunc='sum')
                
    def get_area_column(self, layer_name):
        return self.get_data_layer(layer_name).area
    
    def get_perimeter_column(self, layer_name):
        return self.get_data_layer(layer_name).length
    
    def get_row_area(self, layer_name, row_index):
        return self.data_layers[layer_name].area.iloc[row_index]
    
    def get_distance(self, layer_name, index_column, row_index_a, row_index_b):
        if 1 == 1:
            other = self.get_row(layer_name, row_index_b, index_column)
            return self.get_row(layer_name, row_index_a, index_column).distance(other)
    
    def filter_dataframe(self, layer_name, column, func_de_decision, in_place=True, *args):
        if in_place is True:
            if len(args) == 2:
                self.set_dataframe(
                    self.data_layers[layer_name].loc[self.get_column(column).apply(func_de_decision, args=(args[0], args[1]))])
            else:
                self.set_dataframe(
                    self.data_layers[layer_name].loc[self.get_column(column).apply(func_de_decision)])
        else:
            if len(args) == 2:
                return self.data_layers[layer_name].loc[self.get_column(column).apply(func_de_decision, args=(args[0], args[1]))]
            else:
                return self.data_layers[layer_name].loc[self.get_column(column).apply(func_de_decision)]

    def transform_column(self, layer_name, column_to_trsform, column_src, fun_de_trasformation, in_place= True,*args):
        if in_place is True:
            if (len(args) != 0):
                self.set_column(layer_name, column_to_trsform, self.get_column(layer_name, column_src).apply(fun_de_trasformation, args=(args[0],)))
            else:
                self.set_column(layer_name, column_to_trsform, self.get_column(layer_name, column_src).apply(fun_de_trasformation)) 
        else:
            if (len(args) != 0):
                return self.get_column(layer_name, column_src).apply(fun_de_trasformation, args=(args[0],))
            else:
                return self.get_column(layer_name, column_src).apply(fun_de_trasformation)
            
    def set_column(self, layer_name, column_name, new_column):
        self.data_layers[layer_name][column_name] = new_column
    
    def get_column(self, layer_name, column_name):
        return self.data_layers[layer_name][column_name]
    
    def reindex_dataframe(self, layer_name, index_as_liste=None, index_as_column_name=None):
        if index_as_liste is not None:
            new_index = new_index = index_as_liste
            self.data_layers[layer_name].index = new_index
        if index_as_column_name is not None:
            self.data_layers[layer_name].set_index(index_as_column_name, inplace=True)
        if index_as_column_name is None and index_as_liste is None:
            new_index = pd.Series(np.arange(self.get_shape()[0]))
            self.data_layers[layer_name].index = new_index
            
    def get_era5_land_grib_as_dataframe(self, file_path, layer_name):
        grip_path = file_path
        ds = xr.load_dataset(grip_path, engine="cfgrib")
        self.data_layers[layer_name] = DataFrame()
        self.data_layers[layer_name].set_dataframe(ds.to_dataframe())
        return ds.to_dataframe()
    
    def rename_columns(self, layer_name, column_dict_or_all_list, all_columns=False):
        if all_columns is True:
            types = {}
            self.data_layers[layer_name].columns = column_dict_or_all_list
            for p in column_dict_or_all_list:
                types[p] = str
            self.data_layers[layer_name] = self.get_dataframe().astype(types)
        else:
            self.data_layers[layer_name].rename(columns=column_dict_or_all_list, inplace=True)
            
    def add_area_column(self, layer_name, unit='ha'):
        self.add_column(layer_name, self.get_area_column(layer_name), 'area')
        if unit == 'ha':
            self.add_column(layer_name, self.get_area_column(layer_name) / 10000, 'area')
            
    def index_to_column(self, layer_name, column_name=None, drop_actual_index=False, **kwargs):
        self.data_layers[layer_name].reset_index(drop=drop_actual_index, inplace=True, **kwargs) 
        if column_name is not None:
            self.rename_columns({'index': column_name})
            
        
    def calculate_perimeter_as_column(self, layer_name):
        self.add_column(layer_name, self.get_perimeter_column(layer_name), 'perimeter')
        
    def count_occurrence_of_each_row(self, layer_name, column_name):
        return self.data_layers[layer_name].pivot_table(index=[column_name], aggfunc='size')
    
    def count_occurrence_of_each_row(self, layer_name, column_name):
        return self.data_layers[layer_name].pivot_table(index=[column_name], aggfunc='size')
    
    def add_transformed_columns(self, layer_name, dest_column_name="new_column", transformation_rule="okk*2"):
        columns_names = self.get_columns_names(layer_name)
        columns_dict = {}
        for column_name in columns_names:
            if column_name in transformation_rule:
                columns_dict.update({column_name: self.get_column(layer_name, column_name)})
        y_transformed = eval(transformation_rule, columns_dict)
        self.data_layers[layer_name][dest_column_name] = y_transformed
    
    @staticmethod
    def lon_lat_dataframe_to_geopandas(src_dataframe, lon_column_name, lat_column_name, crs=4326):
        from pyproj import CRS
        # Assuming your GeoDataFrame is named 'gdf' and you want to reproject to EPSG:4326
        target_crs = CRS.from_epsg(crs)

        geometry = [Point(lon, lat) for lon, lat in zip(src_dataframe[lon_column_name], src_dataframe[lat_column_name])]
        geo_data = gpd.GeoDataFrame(src_dataframe, geometry=geometry, crs=target_crs)
        
        # Reproject the GeoDataFrame
        gdf_reprojected = geo_data.to_crs(target_crs)
        
        return gdf_reprojected

    @staticmethod
    def new_geodaraframe_from_points():
        map.new_data_layer('valves', crs="ESRI:102191")
        for p in range(map.get_layer_shape('pipelines')[0]):
            #print(map.get_row('pipelines', p))
            vi = Point(map.get_row('pipelines', p)['X_Start'], map.get_row('pipelines', p)['Y_Start'])
            vf = Point(map.get_row('pipelines', p)['X_End'], map.get_row('pipelines', p)['Y_End'])
            id_pipeline = map.get_row('pipelines', p)['Nom_CANAL']
            row_as_dict = {'id_pipeline': id_pipeline,
                        'geometry': vi}
            map.add_row('valves', row_as_dict)
            row_as_dict = {'id_pipeline': id_pipeline,
                        'geometry': vf}
            map.add_row('valves', row_as_dict)
        
    
        
    