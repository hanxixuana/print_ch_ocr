#!/usr/bin/env python

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter


def draw_record(path_to_record, path_to_parameters, n_param_per_line = 3):
    with open(path_to_record, 'r') as file:
        records = file.read()

    rows = [
        item[item.find('train'):]
        for item in [
            row for row in records.split('\n')
            if 'train' in row
        ]
    ]
    lines = dict()
    for row in rows:
        row = row.replace(' train', ': train')
        row = row.replace(' validate', ': validate')
        content_list = row.split(': ')
        key = None
        for idx, content in enumerate(content_list):
            if idx % 2 == 0:
                if content not in lines:
                    lines[content] = list()
                key = content
            else:
                lines[key].append(float(content))

    def draw(ax, line, label, line_type, color_list):

        def make_line(color):
            ax.plot(
                line,
                '%s%s' % (line_type, color),
                label=label
            )

        def find_min(color):
            ax.plot(
                np.argmin(line),
                np.min(line),
                'o%s' % color,
                label=(
                        'min %s at %d: %f' %
                        (
                            label,
                            int(np.argmin(line)),
                            np.min(line)
                        )
                )
            )
            return np.argmin(line)

        t3_min_arg = 50
        t1_min_arg = 50
        if 'loss' in label:
            make_line(color_list[0])
            find_min(color_list[0])
        elif 'error' in label:
            if 'top-3' in label:
                make_line(color_list[1])
                t3_min_arg = find_min(color_list[1])
            else:
                make_line(color_list[2])
                t1_min_arg = find_min(color_list[2])
        return max([t3_min_arg, t1_min_arg])

    min_arg_list = list()
    fig = plt.figure(figsize=[12, 9])
    ax = fig.add_subplot(111)
    for label in lines:
        line = lines[label]
        if 'train' in label:
            min_arg_list.append(
                draw(ax, line, label, '--', ['k', 'b', 'r'])
            )
        elif 'validate' in label:
            min_arg_list.append(
                draw(ax, line, label, '-', ['k', 'b', 'r'])
            )
    if len(min_arg_list) == 0:
        return lines

    ax.set_xlim(0, max(min_arg_list) * 1.2)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.3f"))
    ax.grid(which='minor')
    ax.grid(which='major')
    ax.legend()

    with open(path_to_parameters, 'r') as file:
        params = json.load(file)

    lr = {'lr': params.pop('lr')}
    try:
        params.pop('char_list')
    except KeyError:
        pass
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

    return lines


def draw_record_for_every_log(path_to_logs, logs, lines_list):
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    fig = plt.figure(figsize=[12, 9])
    ax = fig.add_subplot(111)
    for idx, (log, lines) in enumerate(zip(logs, lines_list)):
        validate_error_name_list = [
            line_name for line_name in lines
            if (
                    'validate' in line_name
                    and
                    'error' in line_name
            )
        ]
        for line_name in validate_error_name_list:
            line_format = '-%s' % color_list[idx%len(color_list)]
            circle_format = 'o%s' % color_list[idx%len(color_list)]
            ax.plot(
                lines[line_name],
                line_format,
                label=log
            )
            ax.plot(
                np.argmin(lines[line_name]),
                np.min(lines[line_name]),
                circle_format,
                label=(
                        '%s: %d, %f' %
                        (
                            log,
                            int(np.argmin(lines[line_name])),
                            np.min(lines[line_name])
                        )
                )
            )
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.3f"))
    ax.grid(which='minor')
    ax.grid(which='major')
    ax.legend()
    ax.set_title('Validate Error')
    if not os.path.exists(os.path.join(path_to_logs, 'summary')):
        os.mkdir(os.path.join(path_to_logs, 'summary'))
    fig.savefig(
        os.path.join(
            path_to_logs, 'summary', 'val_error_summary.png'
        )
    )
    print(
        'Saved a summary of validate error to %s.' %
        os.path.join(
            path_to_logs, 'summary', 'val_error_summary.png'
        )
    )


def draw_record_for_all_logs(path_to_logs):
    path_to_logs = os.path.abspath(path_to_logs)
    logs = os.listdir(
        os.path.abspath(path_to_logs)
    )
    lines_list = list()
    for log in logs:
        try:
            files = os.listdir(
                os.path.join(path_to_logs, log)
            )
        except NotADirectoryError:
            continue
        if 'record.txt' in files:
            lines_list.append(
                draw_record(
                    os.path.join(path_to_logs, log, 'record.txt'),
                    os.path.join(path_to_logs, log, 'params.json')
                )
            )
            print(
                'Draw a plot for %s and saved to %s.' %
                (
                    os.path.join(path_to_logs, log, 'record.txt'),
                    os.path.join(path_to_logs, log, 'record.png')
                )
            )
    draw_record_for_every_log(path_to_logs, logs, lines_list)


if __name__ == '__main__':
    draw_record_for_all_logs('../logs')
