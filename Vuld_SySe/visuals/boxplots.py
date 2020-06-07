import json
import numpy as np
# %matplotlib inline
import pandas as pd
from glob import glob
from pdb import set_trace
from pathlib import Path
# import seaborn as sns; sns.set()
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# --- Configure seaborn ---
# matplotlib.rcParams['pdf.fonttype'] = 3
# matplotlib.rcParams['ps.fonttype'] = 3
# sns.set_style({'font.family': 'Palatino'})
# sns.set_style("ticks")
# sns.set_context("poster", font_scale=1.30, rc={"lines.linewidth": 2.0})

boxprops = dict(linestyle='-', linewidth=3)
medianprops = dict(linestyle='-', linewidth=3)
flierprops=dict(linestyle='--', linewidth=3)

import sys

def plot(data, output_file):
    # ax = sns.boxplot(data=data, width=0.55)
    # # iterate over boxes
    # for i, box in enumerate(ax.artists):
    #     box.set_edgecolor('black')
    #     box.set_facecolor('white')
    #     # iterate over whiskers and median lines
    #     for j in range(6 * i, 6 * (i + 1)):
    #         ax.lines[j].set_color('black')
    data.boxplot(boxprops=boxprops,
                medianprops=medianprops, showmeans=True, fontsize=28)
    plt.savefig(output_file, bbox_inches = "tight")
    plt.show()

def precision_recall_f1():
    project = ['verum', 'devign']  # sys.argv[1]
    part = ['precision', 'recall', 'f1']  # sys.argv[2]
    for p in project:
        for pa in part:
            file = '/home/saikatc/Documents/Bias-Result/' + p + '/' + pa + '.csv'
            output_file = '/home/saikatc/Desktop/boxplots/' + p + '-' + pa + '.pdf'
            data = pd.read_csv(file)
            plt.figure(figsize=(6, 5))
            data = data.rename(columns={
                "VulDeePecker": "Vul",
                "SySeVR": "Sys",
                "Russel et. al.": "Rus",
                "GGNN": "Dev",
                "\\tool": "Rev"
            })
            # ax = sns.boxplot(data=data, width=0.55)
            # # iterate over boxes
            # for i, box in enumerate(ax.artists):
            #     box.set_edgecolor('black')
            #     box.set_facecolor('white')
            #     # iterate over whiskers and median lines
            #     for j in range(6 * i, 6 * (i + 1)):
            #         ax.lines[j].set_color('black')
            # plt.savefig(output_file)
            # plt.show()
            plot(data, output_file)

def ggnn_ablation():
    for p in ['verum', 'devign']:
        file_name = '/home/saikatc/Documents/Bias-Result/' + p +'/ggnn-abl-f1.csv'
        output_path = '/home/saikatc/Desktop/boxplots/' + p + '-ggnn-abl-f1.pdf'
        data = pd.read_csv(file_name)
        data = data.rename(columns={
            "No-ggnn": "W/o GGNN",
            "\\tool": "With GGNN"
        })
        plt.figure(figsize=(7, 6))
        # ax = sns.boxplot(data=data, width=0.5)
        # # iterate over boxes
        # for i, box in enumerate(ax.artists):
        #     box.set_edgecolor('black')
        #     box.set_facecolor('white')
        #     # iterate over whiskers and median lines
        #     for j in range(6 * i, 6 * (i + 1)):
        #         ax.lines[j].set_color('black')
        # plt.savefig(output_path)
        # plt.show()
        plot(data, output_path)

def smote_ablation():
    for p in ['verum', 'devign']:
        file_name = '/home/saikatc/Documents/Bias-Result/' + p +'/smote-abl-f1.csv'
        output_path = '/home/saikatc/Desktop/boxplots/' + p + '-smote-abl-f1.pdf'
        data = pd.read_csv(file_name)
        data = data.rename(columns={
            "Without-SMOTE": "W/o SMOTE",
            "With-SMOTE": "With SMOTE"
        })
        plt.figure(figsize=(7, 6))
        # ax = sns.boxplot(data=data, width=0.5)
        # # iterate over boxes
        # for i, box in enumerate(ax.artists):
        #     box.set_edgecolor('black')
        #     box.set_facecolor('white')
        #     # iterate over whiskers and median lines
        #     for j in range(6 * i, 6 * (i + 1)):
        #         ax.lines[j].set_color('black')
        # plt.savefig(output_path)
        # plt.show()
        plot(data, output_path)

def model_ablation():
    for p in ['verum', 'devign']:
        file_name = '/home/saikatc/Documents/Bias-Result/' + p +'/model-abl-f1.csv'
        output_path = '/home/saikatc/Desktop/boxplots/' + p + '-model-abl-f1.pdf'
        data = pd.read_csv(file_name)
        print(data.columns)
        data = data.rename(columns={
            "svm": "SVM",
            "rf": "RF",
            "mlp": "MLP",
            "\\tool": "ReVeal"
        })
        plt.figure(figsize=(7, 6))
        # ax = sns.boxplot(data=data, width=0.5)
        # # iterate over boxes
        # for i, box in enumerate(ax.artists):
        #     box.set_edgecolor('black')
        #     box.set_facecolor('white')
        #     # iterate over whiskers and median lines
        #     for j in range(6 * i, 6 * (i + 1)):
        #         ax.lines[j].set_color('black')
        # plt.savefig(output_path)
        # plt.show()
        plot(data, output_path)

if __name__ == '__main__':
    precision_recall_f1()
    ggnn_ablation()
    smote_ablation()
    model_ablation()
    # print(data)