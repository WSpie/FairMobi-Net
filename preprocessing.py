import warnings
warnings.filterwarnings('ignore')
import os, sys, re, pickle
import pandas as pd
import geopandas as gpd
import numpy as np
import dask
import dask.dataframe as dd
import dask.array as da
from shapely import wkt
import seaborn as sns
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from pygris import tracts
from sklearn.model_selection import train_test_split

from dask.distributed import Client
client = Client(n_workers=40)

## Table 1.a: census tract polygon table
def get_poly_df(state: str, counties: str, year=2020):
    poly_df = tracts(state=state, county=counties, cb=False, year=year)[['GEOID', 'geometry']]
    print(poly_df.crs)
    poly_df = poly_df.rename(columns={'GEOID': 'census_tract'})
    poly_df['centroid'] = poly_df['geometry'].centroid
    poly_df = poly_df.reset_index(drop=True)
    poly_df.to_csv(f'data/processed/{counties.split(" ")[0]}_poly.csv', index=False)


def get_poly_dask_df(state: str, counties: str, year=2019, npartition=4) -> dd.core.DataFrame:
    poly_df = tracts(state=state, county=counties, cb=False, year=year)[['GEOID', 'geometry']]
    poly_df = poly_df.rename(columns={'GEOID': 'FIPS'})
    poly_df['geometry_wkt'] = poly_df['geometry'].to_wkt()
    poly_df = poly_df.drop('geometry', axis=1).reset_index(drop=True)
    poly_df = dd.from_pandas(poly_df, npartitions=npartition)
    return poly_df

## Table 1.b: census tract features table
def get_feature_dask_df_from_city(city_path, poly_df):
    city_feature_df = dd.read_csv(city_path)
    city_feature_df['FIPS'] = city_feature_df['FIPS'].astype(str)
    feature_df = poly_df.merge(city_feature_df, on='FIPS', how='left')
    return feature_df.drop(['LOCATION', 'streets_per_node_counts', 'streets_per_node_proportions', 'graph’s average node degree'], axis=1)

## Table 1.c: census tract population table
def get_census_tract_popul_df(path, feat_attr_df):
    popul_df = dd.read_csv(path).loc[1:][['GEO_ID', 'P2_001N']]
    popul_df = popul_df.rename(columns={'GEO_ID': 'FIPS', 'P2_001N': 'pop'})
    popul_df['pop'] = popul_df['pop'].astype(int)
    popul_df['FIPS'] = popul_df['FIPS'].apply(lambda x: x[-11:]).astype(str)
    feat_attr_pop_df = feat_attr_df.merge(popul_df, on='FIPS', how='left')
    return feat_attr_pop_df

## Table 1.d: census tract protected attribute table
def get_attr_dask_df_from_city(city_path, feature_df):
    city_attr_df = dd.read_csv(city_path)[['PCI', 'FIPS']]
    city_attr_df['FIPS'] = city_attr_df['FIPS'].astype(str)
    feature_attr_df = feature_df.merge(city_attr_df, on='FIPS', how='left')
    return feature_attr_df

## Table I: Individual summarized table
def get_summarized_ddf(state, city, county):
    poly_ddf = get_poly_dask_df(state, county)
    feature_ddf = get_feature_dask_df_from_city(f'data/processed/{city}_osm.csv', poly_ddf)
    feature_pop_ddf = get_census_tract_popul_df(f'data/processed/{county.split(" ")[0]}_popul.csv', feature_ddf)
    feature_pop_attr_ddf = get_attr_dask_df_from_city(f'data/processed/{city}_pci.csv', feature_pop_ddf).fillna(0)
    return feature_pop_attr_ddf

## Table 2.a: census tract flow table
def get_county_ct_flow_dask_df(county_fips: str, days=14, npartitions=10) -> dd.core.DataFrame:  
    dir = 'data/raw/flow_files'
    folder = os.path.join(dir, county_fips)
    flow_dfs = [dd.read_csv(os.path.join(folder, f_name)) for f_name in os.listdir(folder) if f_name.endswith('.csv')]
    flow_df = dd.concat(flow_dfs[:days], axis=0)
    flow_df = flow_df.rename(columns={'Origin_ID': 'FIPS_i', 'Destination_ID': 'FIPS_j', 'Count': 'count'})
    flow_df = flow_df[['FIPS_i', 'FIPS_j', 'count']]
    # Only get ct to ct counts
    flow_df = flow_df.loc[(flow_df['FIPS_i'].astype(str).apply(len)==11) & (flow_df['FIPS_j'].astype(str).apply(len)==11)]
    flow_df = flow_df.groupby(['FIPS_i', 'FIPS_j']).agg({'count': 'sum'}).reset_index().dropna()
    flow_df[['FIPS_i', 'FIPS_j']] = flow_df[['FIPS_i', 'FIPS_j']].astype(str)

    # Add 0 flows as non-linearty
    unique_fips = flow_df[['FIPS_i', 'FIPS_j']].melt().drop_duplicates().drop('variable', axis=1).rename(columns={'value': 'FIPS'}).reset_index(drop=True)
    # Perform a Cartesian join on the unique FIPS codes to generate all possible combinations of FIPS_i and FIPS_j
    all_combinations = unique_fips.assign(key=1).merge(unique_fips.assign(key=1), on='key', suffixes=('_i', '_j')).drop('key', axis=1)
    merged_df = all_combinations.merge(flow_df, on=['FIPS_i', 'FIPS_j'], how='left').fillna(0)
    new_rows = merged_df[merged_df['count'] == 0]
    new_rows_sample = new_rows.sample(frac=0.25)
    flow_df_extended = flow_df.append(new_rows_sample).reset_index(drop=True)
    flow_df_extended = flow_df_extended.groupby(['FIPS_i', 'FIPS_j']).agg({'count': 'sum'}).reset_index().dropna()
    flow_df_extended = flow_df_extended.repartition(npartitions=npartitions)
    flow_df_extended = flow_df_extended.map_partitions(lambda df: df.sample(frac=1))

    return flow_df_extended.reset_index(drop=True)

## Table 2.b: + features + protected attributes + geometry_wkt
def merge_flow_df(ind_df, flow_df):
    merged_flow_i_df = flow_df.merge(ind_df, left_on='FIPS_i', right_on='FIPS', suffixes=('_', '_i')).drop('FIPS', axis=1)
    merged_flow_ij_df = merged_flow_i_df.merge(ind_df, left_on='FIPS_j', right_on='FIPS', suffixes=('_i', '_j')).drop('FIPS', axis=1)
    return merged_flow_ij_df

## Table 2.c: + distance
def compute_distance(geometry_wkt_i, geometry_wkt_j):
    geom_i = wkt.loads(geometry_wkt_i)
    geom_j = wkt.loads(geometry_wkt_j)

    centroid_i = geom_i.centroid
    centroid_j = geom_j.centroid

    return geodesic((centroid_i.y, centroid_i.x), (centroid_j.y, centroid_j.x)).kilometers

def apply_distance(row):
    return compute_distance(row['geometry_wkt_i'], row['geometry_wkt_j'])

def compute_distance_dask_df(df: dd.DataFrame, npartitions: int = 10) -> dd.DataFrame:
    def apply_distance(row):
        return compute_distance(row['geometry_wkt_i'], row['geometry_wkt_j'])

    # Repartition the dataframe using the specified number of partitions
    df = df.repartition(npartitions=npartitions)

    # Compute the distance using map_partitions
    df['distance'] = df.map_partitions(lambda df: df.apply(apply_distance, axis=1), meta=('distance', 'f8'))

    return df

## Table 2.d: + income diff
def add_income_diff(feat_flow_ddf):
    feat_flow_ddf['income_diff'] = feat_flow_ddf.apply(lambda row: abs(row['PCI_i'] - row['PCI_j']), axis=1)
    income_range = feat_flow_ddf['income_diff'].max().compute() - feat_flow_ddf['income_diff'].min().compute()
    income_range /= 5
    def categorize_income_diff(income_diff):
        return 0 if income_diff < income_range else (1 if income_diff < 2 * income_range else 2)
    feat_flow_ddf['income_diff'] = feat_flow_ddf['income_diff'].map_partitions(lambda s: s.apply(categorize_income_diff), meta=('income_diff', 'i8'))
    return feat_flow_ddf

## Table II: Summarized flow table
def get_summarized_flow_df(fips_code, feat_pop_attr_ddf):
    flow_ddf = get_county_ct_flow_dask_df(fips_code)
    feat_flow_ddf = merge_flow_df(feat_pop_attr_ddf, flow_ddf)
    feat_flow_ddf['distance'] = feat_flow_ddf.map_partitions(lambda df: df.apply(apply_distance, axis=1), meta=('distance', 'f8'))
    feat_flow_ddf = compute_distance_dask_df(feat_flow_ddf)
    feat_flow_ddf = add_income_diff(feat_flow_ddf)
    count = feat_flow_ddf['count']
    feat_flow_ddf = feat_flow_ddf.drop(['geometry_wkt_i', 'geometry_wkt_j', 'count'], axis=1)
    feat_flow_ddf['count'] = count
    feat_flow_df = feat_flow_ddf.compute()
    return feat_flow_df
    
# Aggregate all the data
def aggregation():
    processed_folder = 'data/processed'
    feat_paths = [os.path.join(processed_folder, f) for f in os.listdir(processed_folder) if f.endswith('_feat.parquet')]
    df_dict = {re.search(r"/([^/]+)_feat", feat_path).group(1): pd.read_parquet(feat_path) for feat_path in feat_paths}
    return df_dict

## Data Processing
def data_process(df_dict):
    processed_df_dict = {}
    main_train_df = pd.DataFrame()
    main_val_df = pd.DataFrame()
    test_dict = {}

    for key, df in df_dict.items():
        
        df = df[df['count'] <= 50]
        
        train_df, temp_df = train_test_split(df, test_size=0.4, shuffle=True, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, shuffle=True, random_state=42)
        
        if main_train_df.empty:
            main_train_df = train_df
        else:
            main_train_df = pd.concat([main_train_df, train_df], axis=0)
        if main_val_df.empty:
            main_val_df = val_df
        else:
            main_val_df = pd.concat([main_val_df, val_df], axis=0)
        
        test_dict[key] = test_df

    processed_df_dict['train_df'] = main_train_df
    processed_df_dict['val_df'] = main_val_df
    processed_df_dict['test_dict'] = test_dict

    with open('data/processed/df_dict.pkl', 'wb') as f:
        pickle.dump(processed_df_dict, f)


def remove_irr_feats(df):
    remove_cols_prefix = ['streets_per_node_avg', 'street_length_avg', 'circuity_avg', 'self_loop_proportion',
                          'node_density_km', 'intersection_density_km', 'edge_density_km', 'street_density_km',
                          'edge_length_avg', 'edge_length_total']
    cols_to_remove = [col for col in df.columns if any(col.startswith(prefix) for prefix in remove_cols_prefix)]
    df_clean = df.drop(columns=cols_to_remove)
    return df_clean

def load_df(place):
    def one_hot(df):
        # use one-hot to encode 'income_diff'
        df['income_diff'] = pd.cut(df['income_diff'], bins=[-np.inf, 0.9, 1.9, np.inf], labels=[0, 1, 2])
        df = pd.get_dummies(df, columns=['income_diff'], prefix='inc_diff')
        df['inc_diff_0'] = df['inc_diff_0'].astype(int)
        df['inc_diff_1'] = df['inc_diff_1'].astype(int)
        df['inc_diff_2'] = df['inc_diff_2'].astype(int)
        return df
    
    df = pd.read_parquet(f'data/processed/{place}_feat.parquet')
    df = remove_irr_feats(df)
    df = df[df['count'] <= 50]
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=0, stratify=df['income_diff'])
    
    indices = range(len(train_df))
    train_mask, val_mask = train_test_split(indices, test_size=0.2, random_state=0, stratify=train_df['income_diff'])
    train_df = one_hot(train_df)
    test_df = one_hot(test_df)
    feature_cols = train_df.drop(columns=['FIPS_i', 'FIPS_j', 'count']).columns
    feat_scales = train_df[feature_cols].values.ptp(axis=0)
    return train_df, test_df, train_mask, val_mask, feature_cols, feat_scales