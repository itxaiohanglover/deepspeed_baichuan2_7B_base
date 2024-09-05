#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import torch
import bottle
import logging
import functools
from bottle import route, response, request, static_file, hook

sys.path.append("../")

from plugins.common import error_print
from plugins.common import CounterLock, allowCROS
from plugins.common import settings
from module_interface import llm_interface


logging.captureWarnings(True)


mutex = CounterLock()


@route('/static/<path:path>')
def staticjs(path='-'):
    if path.endswith(".html"):
        noCache()
    if path.endswith(".js"):
        return static_file(path, root="views/static/", mimetype="application/javascript")
    return static_file(path, root="views/static/")


@route('/:name')
def static(name='-'):
    if name.endswith(".html"):
        noCache()
    return static_file(name, root="views")


@route('/api/llm')
def llm_js():
    noCache()
    return static_file('llm_'+settings.llm_type+".js", root="llms")


@route('/api/plugins')
def read_auto_plugins():
    noCache()
    plugins = []
    for root, dirs, files in os.walk("autos"):
        for file in files:
            if(file.endswith(".js")):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding='utf-8') as f:
                    plugins.append(f.read())
    return "\n".join(plugins)


def noCache():
    # 请求头里的Cache-Control是no-cache，是浏览器通知服务器：本地没有缓存数据
    response.set_header("Pragma", "no-cache")
    response.add_header("Cache-Control", "must-revalidate")
    response.add_header("Cache-Control", "no-cache")
    response.add_header("Cache-Control", "no-store")


def pathinfo_adjust_wrapper(func):
    @functools.wraps(func)
    def _(s, environ):
        environ["PATH_INFO"] = environ["PATH_INFO"].encode(
            "utf8").decode("latin1")
        return func(s, environ)
    return _


bottle.Bottle._handle = pathinfo_adjust_wrapper(bottle.Bottle._handle)


@hook('before_request')
def validate():
    REQUEST_METHOD = request.environ.get('REQUEST_METHOD')
    HTTP_ACCESS_CONTROL_REQUEST_METHOD = request.environ.get(
        'HTTP_ACCESS_CONTROL_REQUEST_METHOD')
    if REQUEST_METHOD == 'OPTIONS' and HTTP_ACCESS_CONTROL_REQUEST_METHOD:
        request.environ['REQUEST_METHOD'] = HTTP_ACCESS_CONTROL_REQUEST_METHOD


@route('/')
def index():
    noCache()
    return static_file("index.html", root="views")


@route('/api/chat_now', method=('GET', "OPTIONS"))
def api_chat_now():
    allowCROS()
    noCache()
    return {'queue_length': mutex.get_waiting_threads()}


@route('/api/chat_stream', method=("POST", "OPTIONS"))
def api_chat_stream():
    allowCROS()
    data = request.json
    print(data)
    if not data:
        return '0'
    prompt = data.get('prompt')
    footer = '///'
    print('prompt is : ' + prompt)
    try:
        _, response = llm_interface.get_llm_answer(prompt)
        if response:
            yield response + footer
    except Exception as e:
        error = str(e)
        error_print("错误", error)
        response = ''
    torch.cuda.empty_cache()
    if response == '':
        yield "发生错误，服务结束 ///"
        exit(0)
    print(response)
    yield "/././"


bottle.debug(True)

bottle.run(server='paste', host="0.0.0.0", port=settings.port, quiet=True)
