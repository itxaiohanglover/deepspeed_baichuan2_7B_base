#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import requests
sys.path.append("../")
from common import constants


def ask(query='我的宠物狗死了，我非常难过'):
    url = constants.HTTP_SERVER + '/ask'
    post_data = {'appid': 'bot',
                 'userId': 'user_test_001',
                 'apiType': 'ask',
                 'sessionId': 'session_2023_0721',
                 'logId': 'log_id_001',
                 'query': query,
                 'query_type': "llm_test"}
    try:
        response = requests.post(url, json=post_data)
        print('状态码：', response.status_code)
        print('响应正文：', response.text)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    ask()
