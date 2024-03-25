import geopandas as gpd
import pandas as pd
from shapely import wkt
from tqdm import tqdm
import os

def gen_adjacency_matrix_df(pre_cfg_dict):
    if os.path.exists(pre_cfg_dict['adjacency_csv_path']):
        adjacency_df = pd.read_csv(pre_cfg_dict['adjacency_csv_path'])
        print(f'Loaded adjacency matrix from cache')

    else:
        df = pd.read_csv(pre_cfg_dict['polygon_path'])
        df = df[list(df.columns)[::-1]]
        df['geometry'] = df['geometry'].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(df, crs='epsg:4326')
        FIPS_col = gdf.FIPS.values.tolist()
        adjacency_df = pd.DataFrame([], columns=FIPS_col)
        for i in tqdm(range(len(gdf)), desc='Generating adjacency matrix'):
            temp = []
            for j in range(len(gdf)):
                touch = int(gdf.geometry[i].touches(gdf.geometry[j]))
                temp.append(touch)
            temp_df = pd.DataFrame(dict(zip(FIPS_col, temp)), index=[0])
            adjacency_df = pd.concat([adjacency_df, temp_df])
        adjacency_df = adjacency_df.set_index([pd.Index(FIPS_col)])
        adjacency_df.to_csv(pre_cfg_dict['adjacency_csv_path'])
    return adjacency_df

