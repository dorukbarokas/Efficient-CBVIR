#!/usr/bin/env python
import matplotlib.pyplot as plt
import pandas as pd


cm = 1/2.54  # convert inches to centimeters
_figsize = (13*cm, 13*cm)


def plot_ap(_data):
    fig, ax = plt.subplots(figsize=_figsize)

    ax.plot(_data['size'], _data['ap'])

    ax.set_title(f"Average Precision@{_data['k'].iloc[0]} vs. Size")
    ax.set_xlabel('Size [px]')
    ax.set_ylabel(f"AP@{_data['k'].iloc[0]}")

    plt.xlim([min(_data['size']), max(_data['size'])])

    plt.grid()
    plt.show()


def plot_recall(_data):
    fig, ax = plt.subplots(figsize=_figsize)

    ax.plot(_data['size'], _data['recall'])

    ax.set_title(f"Recall@{_data['k'].iloc[0]} vs. Size")
    ax.set_xlabel('Size [px]')
    ax.set_ylabel(f"Recall@{_data['k'].iloc[0]}")

    plt.xlim([min(_data['size']), max(_data['size'])])

    plt.grid()
    plt.show()


data = pd.read_csv('../../local-files/measurements/performance.csv')

data_selection = data[(data['name'] == 'tudelft-ewi-2') & (data['search_image'] == 'ewi-tudelft-logo.jpg')]
print(data_selection)
plot_ap(data_selection)
plot_recall(data_selection)
