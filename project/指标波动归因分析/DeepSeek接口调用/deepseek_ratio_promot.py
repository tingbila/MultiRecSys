# -*- coding: utf-8 -*-
import argparse
from openai import OpenAI
from pyhive import hive
import pandas as pd
import json
import requests



def prompt_con():
    prompt_system = """
    你是一名数据分析专家，擅长使用 Adtributor 算法进行归因分析和指标波动解读。
    我将给你不同维度的归因指标数据，请你根据数据直接给出简明扼要的结论，
    重点总结各维度及元素对整体波动的影响大小，避免解释算法细节，只输出结论和排序。
    """

    prompt_user = f"""
    请根据以下数据，基于 Adtributor 归因算法的思想，结合数据给出指标波动结论。：

    dim,element,m1_before,m1_after,m1_pre_sum,m1_aft_sum,m1_p,m1_q,m1_surprise,m1_ep,m2_before,m2_after,m2_pre_sum,m2_aft_sum,m2_p,m2_q,m2_surprise,m2_ep,surprise,surprise_rank,ep,ep_sum,lag_ep_sum,surprise_sum,overall_dim_surprise_rank
    channel,unknown,5627,5495,67854,63245,0.082928051404,0.086884338683,1.000852e-05,0.028639618138,20742,19709,213504,197279,0.097150404676,0.099904196595,4.17844e-06,0.063667180277,1.418696e-05,5,0.357071720878,0.78991378308,0.432842062202,0.000148694383,1
    channel,googleadwords_int,32280,31621,67854,63245,0.475727296843,0.49997628271,6.5439248e-05,0.142981123888,101832,99016,213504,197279,0.476955935252,0.501908464662,6.9068175e-05,0.173559322034,0.000134507423,2,0.432842062202,0.432842062202,0.432842062202,0.000148694383,1
    channel_type,其他,7590,7342,67854,63245,0.111857812362,0.116088228318,8.524783e-06,0.053807767412,27292,25883,213504,197279,0.127828986811,0.131199975669,4.763245e-06,0.086841294299,1.3288028e-05,4,0.362085284222,0.906771858713,0.544686574491,0.00011674755799999999,2
    channel_type,自投,38229,37088,67854,63245,0.563400831196,0.586417898648,5.0029243e-05,0.247559123454,121610,117088,213504,197279,0.569591202038,0.593514768424,5.3430287e-05,0.278705701079,0.00010345953,2,0.544686574491,0.544686574491,0.544686574491,0.00011674755799999999,2

    请完成以下任务：
    1. 按 surprise_sum 排序，指出哪个维度对整体影响最大；
    2. 对每个维度内的元素，按影响大小排序。

    示例格式：
    【指标波动结论】
     DAU次日留存率ratio今日数据：m1_aft_sum（X） / m2_aft_sum（X） = x, 昨日数据:m1_pre_sum（X） / m2_pre_sum （X）= x, 增加了(降低了) X，原因如下:
     1. 维度影响排序：
                 A维度（surprise_sum = X）
                 B维度（surprise_sum = X）
     2. 维度内元素排序：
        - 渠道维度：  
                 a1 分子变化:(m1_before x -> m1_after x) 分母变化：(m2_before x -> m2_after x)
                 a2 分子变化:(m1_before x -> m1_after x) 分母变化：(m2_before x -> m2_after x)
                 a3 分子变化:(m1_before x -> m1_after x) 分母变化：(m2_before x -> m2_after x)
        - 新老客维度：
                 b1 分子变化:(m1_before x -> m1_after x) 分母变化：(m2_before x -> m2_after x) 
                 b2 分子变化:(m1_before x -> m1_after x) 分母变化：(m2_before x -> m2_after x) 
    注意：只输出结论，不要解释算法或过程，输出结果不要带有*这种特殊字符，同时不要进行科学计数法表示。
    """

    return [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt_user}
    ]



def send_messages(messages):
    headers = {"Content-Type": "application/json; charset=utf-8"}
    client = OpenAI(
        api_key="xxxx",
        base_url="xxxx",
    )

    completion = client.chat.completions.create(
        model="deepseek-r1",  # 确认模型名称是否正确
        messages=messages,
        reasoning_effort="high",  # 确认推理努力程度
        stream=False  # 确认是否支持 stream 参数
    )

    print("messages内容：")
    print(messages)
    print("----------------------------------------------=------------------------------------------------------------")
    print("----------------------------------------------=------------------------------------------------------------")
    print("思考过程：")
    print(completion.choices[0].message.reasoning_content)
    print("----------------------------------------------=------------------------------------------------------------")
    print("----------------------------------------------=------------------------------------------------------------")
    print("最终答案：")
    openai_messages  = completion.choices[0].message.content
    cleaned_json_str = openai_messages.replace("```json", "").replace("```", "").strip()
    print("----------------------------------------------1------------------------------------------------------------")
    print(cleaned_json_str)





if __name__ == '__main__':
    messages = prompt_con()
    send_messages(messages)
