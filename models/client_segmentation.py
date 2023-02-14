import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import squarify
import configparser
import json

parser = configparser.ConfigParser()
parser.read("config.txt")


def load_pickle(path: str) -> pd.DataFrame:
    return pd.read_pickle(path)


def create_df_rfm(transactions: pd.DataFrame, agg_dict:dict=None) -> pd.DataFrame:
    if agg_dict is None:
        agg_dict = {
            'product_id': 'count',
            'date_order': 'max',
            'sales_net': 'sum'
        }
    df_rfm = transactions.groupby('client_id').agg(agg_dict).reset_index()
    df_rfm.columns = ['client_id', 'frequency', 'max_date', 'monetary']
    max_date = transactions.date_order.max()
    df_rfm['recency'] = (max_date - df_rfm['max_date']).dt.days
    return df_rfm.drop(['max_date'], axis=1)


def log_scale_df_rfm_features(df_rfm: pd.DataFrame) -> pd.DataFrame:
    df_rfm['frequency'] = np.log(df_rfm['frequency'] + 1)
    df_rfm['monetary'] = np.log(df_rfm['monetary'] + 1).fillna(0)
    df_rfm['recency'] = np.log(df_rfm['recency'] + 1).max() - np.log(df_rfm['recency'] + 1)
    return df_rfm


def create_rfm_scores(df_rfm: pd.DataFrame) -> pd.DataFrame:
    r_labels, f_labels, m_labels = range(5, 0, -1), range(1,6), range(1,6)
    df_rfm['r_score'] = pd.qcut(df_rfm['recency'], q=5, labels=r_labels).astype(int)
    df_rfm['f_score'] = pd.qcut(df_rfm['frequency'], q=5, labels=f_labels).astype(int)
    df_rfm['m_score'] = pd.qcut(df_rfm['monetary'], q=5, labels=m_labels).astype(int)
    df_rfm['rfm_sum'] = df_rfm['r_score'] + df_rfm['f_score'] + df_rfm['m_score']
    return df_rfm


def assign_label(df_rfm: pd.DataFrame, r_rule:tuple[int], fm_rule:tuple[int], label:str, colname='rfm_label') -> pd.DataFrame:
    df_rfm.loc[(df_rfm['r_score'].between(r_rule[0], r_rule[1]))
            & (df_rfm['f_score'].between(fm_rule[0], fm_rule[1])), colname] = label
    return df_rfm


def labelize(df_rfm: pd.DataFrame, profile_thresholds:dict) -> pd.DataFrame:
    df_rfm['rfm_label'] = ''
    for k, v in profile_thresholds.items():
        df_rfm = assign_label(df_rfm, v['r'], v['f'], k)
    return df_rfm


if __name__ == "__main__":
    
    with open(parser['path']['profile_thresholds']) as json_file:
        profile_thresholds = json.load(json_file)
    transactions = load_pickle(parser['path']['transactions_dataset'])
    df_rfm = transactions.pipe(create_df_rfm) \
                         .pipe(log_scale_df_rfm_features) \
                         .pipe(create_rfm_scores) \
                         .pipe(labelize, profile_thresholds)
    print(df_rfm)
    if parser['client_segmentation']['save_raw_distributions']:
        pass

    if parser['client_segmentation']['save_labeled_distributions']:
        pass

    if parser['client_segmentation']['save_labeled_df_rfm']:
        pass

    if parser['client_segmentation']['save_labels_graphs']:
        pass
