import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import squarify
import configparser
import json
import os
import datetime
import warnings
warnings.simplefilter("ignore")

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

def create_analysis_for_graphs(df_rfm: pd.DataFrame, agg_dict:dict=None) -> pd.DataFrame:
    if agg_dict is None:
        agg_dict = {
            'client_id': 'count',
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'sum'
        }
    df_analysis = df_rfm.groupby('rfm_label').agg(agg_dict).sort_values(by='recency').reset_index()
    df_analysis.rename({'rfm_label': 'label', 'client_id': 'count'}, axis=1, inplace=True)
    df_analysis['count_share'] = df_analysis['count'] / df_analysis['count'].sum()
    df_analysis['monetary_share'] = df_analysis['monetary'] / df_analysis['monetary'].sum()
    df_analysis['monetary'] = df_analysis['monetary'] / df_analysis['count']    
    return df_analysis

if __name__ == "__main__":
    print('Starting job!')
    print(' > creating the RFM dataset...')
    with open(parser['path']['profile_thresholds']) as json_file:
        profile_thresholds = json.load(json_file)
    transactions = load_pickle(parser['path']['transactions_dataset'])
    df_rfm = transactions.pipe(create_df_rfm) \
                         .pipe(log_scale_df_rfm_features) \
                         .pipe(create_rfm_scores) \
                         .pipe(labelize, profile_thresholds)
    print(' > RFM dataset created...')
    if any([
        parser['client_segmentation']['save_raw_distributions'],
        parser['client_segmentation']['save_labeled_distributions'],
        parser['client_segmentation']['save_labeled_df_rfm'],
        parser['client_segmentation']['save_labels_graphs']
        ]):
        print(' > making new folder in data/results/ ...')
        datetime_token = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        folder_path = "../data/results/" + datetime_token
        os.mkdir(folder_path)
        rfm_features = ['recency', 'frequency', 'monetary']
        print(' >', datetime_token, 'folder created in data/results/ ...')
        print(' > saving data and images...')
    if parser['client_segmentation']['save_raw_distributions']:
        for feature in rfm_features:
            fig, ax = plt.subplots(figsize=(12,3), facecolor='white')
            sns.distplot(df_rfm[feature])
            ax.set_title('Distribution of %s' % feature)
            plt.savefig(folder_path + '/raw_distrib_' + feature + '.png', format='png')
    
    if parser['client_segmentation']['save_labeled_distributions']:
        segments = ['loyal customers', 'hibernating', 'potential loyalist']
        for feature in rfm_features:
            fig, ax = plt.subplots(figsize=(12,3), facecolor='white')
            for segment in segments:
                sns.distplot(df_rfm[df_rfm['rfm_label']==segment][feature], label=segment)
            ax.set_title('Distribution of %s' % feature)
            plt.legend()
            plt.savefig(folder_path + '/labeled_distrib_' + feature + '.png', format='png')

    if parser['client_segmentation']['save_labeled_df_rfm']:
        df_rfm.to_csv(folder_path + '/client_base_with_label.csv')

    if parser['client_segmentation']['save_labels_graphs']:
        colors = ['#37BEB0', '#DBF5F0', '#41729F', '#C3E0E5', '#0C6170', '#5885AF', '#E1C340', '#274472', '#F8EA8C', '#A4E5E0', '#1848A0']
        df_analysis = create_analysis_for_graphs(df_rfm)
        for dimension in ['count', 'monetary']:
            labels = df_analysis['label'] + df_analysis[dimension + '_share'].apply(lambda x: ' ({0:.1f}%)'.format(x*100))
            fig, ax = plt.subplots(figsize=(16,6))
            squarify.plot(sizes=df_analysis[dimension], label=labels, alpha=.8, color=colors)
            ax.set_title('RFM Segments of Customers (%s)' % dimension)
            plt.axis('off')
            plt.savefig(folder_path + '/labels_graph_' + dimension + '.png', format='png')
    print('Job done!')