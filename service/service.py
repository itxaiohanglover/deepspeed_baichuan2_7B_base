#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import ujson
import pydantic
import uvicorn
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

sys.path.append("../")

from common import constants
from module_interface import llm_interface

# 递归深度设置
sys.setrecursionlimit(10000)
app = Starlette()


async def get_request_data(request):
    try:
        if request.method == 'POST':
            # 异步获取请求体
            data = ujson.loads(await request.body())
        elif request.method == 'GET':
            data = dict(request.query_params)
        return data
    except pydantic.error_wrappers.ValidationError:
        raise HTTPException(400, detail="bad request")
    except ValueError as e:
        raise HTTPException(400, detail=e.__repr__())


@app.route('/health')
async def healthz(request):
    return JSONResponse({'health': '200'})


@app.route('/internal/ask', methods=['GET', 'POST'])
async def ask(request):
    # 获取请求内容
    request_data = await get_request_data(request)
    print('http_server_ask: request is ' + str(request_data))
    # 参数校验
    if request_data[constants.API_TYPE_STR] != constants.ASK_STR:
        request_data[constants.ERR_MSG_STR] = constants.API_TYPE_ERROR_STR
        return request_data
    # 获取 query 内容
    query = request_data[constants.QUERY_STR]
    print('http_server_ask: query is ' + query)
    # 大模型生成结果
    flag, llm_result = llm_interface.get_llm_answer(query=query)
    if flag:
        response = {'llm_result': llm_result}
    else:
        response = {'llm_result': None}
    # 设置 Response 的 header
    headers = {"Content-Type": "application/json; charset=UTF-8"}
    # 返回 Response
    return JSONResponse(response, headers=headers)


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=10820)
