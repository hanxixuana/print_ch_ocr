#!/usr/bin/env python

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LogLocator

from io import StringIO


def draw_record(path_to_record, path_to_parameters, n_param_per_line = 3):
    with open(path_to_record, 'r') as file:
        records = file.read()
    lines = StringIO(
        '\n'.join(
            records.split('\n')[::2][1:]
        )
    )
    frame = pd.read_csv(lines, sep=' ', header=None)

    train_t1_error_series = frame.iloc[:, 5]
    train_t3_error_series = frame.iloc[:, 9]
    validate_t1_error_series = frame.iloc[:, 15]
    validate_t3_error_series = frame.iloc[:, 19]

    fig = plt.figure(figsize=[12, 9])
    ax = fig.add_subplot(111)
    ax.plot(train_t1_error_series, '--r', label='train top-1')
    ax.plot(
        train_t1_error_series.values.argmin(),
        train_t1_error_series[train_t1_error_series.values.argmin()],
        'or',
        label=(
                'min tr top-1: %f epoch: %d' %
                (
                    train_t1_error_series.values.min(),
                    train_t1_error_series.values.argmin()
                )
        )
    )
    ax.plot(validate_t1_error_series, '--b', label='validate top-1')
    ax.plot(
        validate_t1_error_series.values.argmin(),
        validate_t1_error_series[validate_t1_error_series.values.argmin()],
        'ob',
        label=(
                'min val top-1: %f epoch: %d' %
                (
                    validate_t1_error_series.values.min(),
                    validate_t1_error_series.values.argmin()
                )
        )
    )
    ax.plot(train_t3_error_series, '-g', label='train top-3')
    ax.plot(
        train_t3_error_series.values.argmin(),
        train_t3_error_series[train_t3_error_series.values.argmin()],
        'og',
        label=(
                'min tr top-3: %f epoch: %d' %
                (
                    train_t3_error_series.values.min(),
                    train_t3_error_series.values.argmin()
                )
        )
    )
    ax.plot(validate_t3_error_series, '-k', label='validate top-3')
    ax.plot(
        validate_t3_error_series.values.argmin(),
        validate_t3_error_series[validate_t1_error_series.values.argmin()],
        'ok',
        label=(
                'min val top-3: %f epoch: %d' %
                (
                    validate_t3_error_series.values.min(),
                    validate_t3_error_series.values.argmin()
                )
        )
    )
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.3f"))
    ax.grid(which='minor')
    ax.grid(which='major')
    ax.legend()

    with open(path_to_parameters, 'r') as file:
        params = json.load(file)

    lr = {'lr': params.pop('lr')}
    title_list = list()
    for idx, key in enumerate(params.keys()):
        if idx % n_param_per_line == 0:
            title_list.append(
                [key + ':' + str(params[key])]
            )
        else:
            title_list[-1].append(
                key + ':' + str(params[key])
            )
    title = (
        '\n'.join(
            [' '.join(item) for item in title_list]
            +
            [str(lr)[1:-1]]
        )
    )
    ax.set_title(title)
    fig.savefig(
        '\\'.join(
            path_to_record.split('\\')[:-1]
            +
            ['record.png']
        )
    )
    plt.close()


def draw_record_for_all_logs(path_to_logs):
    path_to_logs = os.path.abspath(path_to_logs)
    logs = os.listdir(
        os.path.abspath(path_to_logs)
    )
    for log in logs:
        files = os.listdir(
            os.path.join(path_to_logs, log)
        )
        if 'record.txt' in files:
            draw_record(
                os.path.join(path_to_logs, log, 'record.txt'),
                os.path.join(path_to_logs, log, 'params.json')
            )
            print(
                'Draw a plot for %s and saved to %s.' %
                (
                    os.path.join(path_to_logs, log, 'record.txt'),
                    os.path.join(path_to_logs, log, 'record.png')
                )
            )


if __name__ == '__main__':
    draw_record_for_all_logs('../logs')
