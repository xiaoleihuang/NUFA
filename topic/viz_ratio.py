import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import math
from collections import OrderedDict


data_list = [
    ('twitter', 'Twitter'),
    ('amazon', 'Amazon'),
    ('yelp_hotel', 'Yelp Hotel'),
    ('yelp_rest', 'Yelp Restaurant'),
]

factor_list = [
    ('gender', ['female', 'male']),
    ('age', ['old', 'young']),
    ('country', ['US', 'no-US']),
    ('region', ['W+MW', 'S+NE']),
]


def run(data_name):
    results = dict()

    for factor, types in factor_list:
        filep = './results/' + data_name + '/' + factor + '.pkl'
        data = pickle.load(open(filep, 'rb')).to_dict()

        factor_capital = factor.capitalize()        

        results[factor_capital] = dict()

        print(data)
        for key in data:
            if factor != 'region':
                if data[key][types[1]] == 0:
                    data[key][types[1]] = data[key][types[0]]/2

                results[factor_capital]['Topic ' + str(key)] = round(math.log2(
                    data[key][types[0]]/data[key][types[1]]), 3
                )
            else:
                results[factor_capital]['Topic ' + str(key)] = round(math.log2(
                    (data[key]['W']+data[key]['MW'])/(data[key]['S'] + data[key]['NE'])), 3
                )
    print(results)

    return results


# visualization
def viz_perform(df, title='default', outpath='./overlap.pdf'):
    """
    Heatmap visualization
    :param df: an instance of pandas DataFrame
    :return: 
    """
    a4_dims = (14.7, 12.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    sns.set(font_scale=1.2)
    viz_plot = sns.heatmap(df, annot=True, cbar=False, center=0, ax=ax, annot_kws={"size": 36}, cmap='RdBu_r', vmin=df.values.min(), fmt='.3f')
    plt.xticks(rotation=0, fontsize=25)
    plt.yticks(rotation=0, fontsize=25)
    plt.xlabel('Demographic Factors', fontsize=25)
    plt.ylabel('Topic Ratios', fontsize=25)
    plt.title(title, fontsize=36)
    viz_plot.get_figure().savefig(outpath, format='pdf')
    plt.close()


for data_name, title in data_list:
    results = run(data_name)
    results = pd.DataFrame(OrderedDict(results))
    viz_perform(results, title=title, outpath='./ratios/'+data_name+'_ratio.pdf')

