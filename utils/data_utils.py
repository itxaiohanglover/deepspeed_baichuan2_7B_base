#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os


line_num = 1000


def read_big_file_lines(big_file, encoding='utf-8'):
    file = open(big_file, 'r', encoding=encoding)
    while 1:
        lines = file.readlines(line_num)
        if not lines:
            break
        yield from lines

    return None


def save_line_to_file(save_file, line, mode='a'):
    file_token = save_file.split('/')
    file_name = file_token[len(file_token) - 1]
    file_dir = save_file.split(file_name)[0]
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)

    with open(save_file, mode) as f:
        f.write(line)


def get_file_names(file_dir):
    path_list = []
    for root, dirs, files in os.walk(file_dir):
        print("root", root)
        print("dirs", dirs)
        print("files", files)
        for file in files:
            if not root.endswith('/'):
                root += '/'
            path_list.append(root + file)
    return path_list

