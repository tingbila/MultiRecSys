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

    prompt_user = """
    请根据以下数据，基于 Adtributor 归因算法的思想，结合数据给出指标波动结论。：

    dim,element,before,after,pre_sum,aft_sum,p,q,surprise,surprise_rank,ep,ep_sum,lag_ep_sum,surprise_sum,overall_dim_surprise_rank
    channel,organic,1624032,1656180,3056814,3121483,0.531282570677,0.530574730024,5.1231e-08,12,0.49711608344,0.888617421021,0.391501337581,1.054355e-06,1
    channel,googleadwords_int,859705,885023,3056814,3121483,0.281242169134,0.28352645201,1.003124e-06,1,0.391501337581,0.391501337581,0.391501337581,1.054355e-06,1
    region,Area_IN,1119484,1143835,3056814,3121483,0.366225750078,0.36643960579,6.777e-09,24,0.376548268877,0.376548268877,0.376548268877,6.777e-09,2

    请完成以下任务：
    1. 按 surprise_sum 排序，指出哪个维度对整体影响最大；
    2. 对每个维度内的元素，按影响大小排序。

    示例格式：
    【指标波动结论】
    今天整体指标增加了(降低了) X（用aft_sum-pre_sum），原因如下:
    1. 维度影响排序：A维度（surprise_sum = X）> B维度（surprise_sum = X）
    2. 维度内元素排序：
       - 渠道维度：  a1（surprise 0.007697）(before x -> after x) >  a2（surprise 0.001973）(before x -> after x)> a3（surprise 0.000725）(before x -> after x)
       - 新老客维度：b1（surprise 0.000557）(before x -> after x) >  b2（surprise 0.000426）(before x -> after x)
    注意：只输出结论，不要解释算法或过程，输出结果不要带有*这种特殊字符。
    """

    return [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt_user}
    ]



def send_messages(messages):
    headers = {"Content-Type": "application/json; charset=utf-8"}
    client = OpenAI(
        api_key="xxxxx",
        base_url="xxxxx",
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


"""
思考过程：
好的，用户给了一个关于Adtributor算法分析的任务，需要处理给出的数据并输出结论。我得先仔细看看用户提供的数据。

首先，数据中有两个维度：app_version和language，每个维度包含不同的元素。用户需要按surprise_sum排序，找出影响最大的维度，然后每个维度内按元素的surprise排序。

看一下示例的输出格式，他们希望先计算整体指标的变化，然后列出维度排序和维度内元素的排序。数据中的aft_sum和pre_sum的总和变化是3047331 - 3119334 = -72003，所以指标是降低了72003。

接下来看维度的surprise_sum，app_version维度的surprise_sum是0.000590605353，language维度的是3.47952e-07，所以维度排序肯定是app_version > language。

然后是每个维度内部的元素排序。在app_version中，元素是8.86.4和9.8.1，对应的surprise值分别是0.000207和0.000383。所以9.8.1的surprise更大，排在前面。但要注意检查是否每个维度的surprise_rank是否已经排好序。数据中app_version的两个元素的surprise_rank是2和1，所以正确的内部排序应该是9.8.1（surprise_rank1）然后8.86.4（surprise_rank2）。

在language维度中，hi的surprise是1.09743e-07（rank21），en的是2.38209e-07（rank14），但根据surprise值的大小，en的更大，所以排序是en > hi。注意虽然他们的surprise_rank看起来靠后，但实际surprise的值可能不同。需要看surprise字段的具体数值，而不是rank。因此，在language维度里，en的surprise值更大，所以排在hi前面。

然后组织成用户要求的格式，确保不要使用特殊符号，直接陈述结论。检查数据无误后，就可以按示例的格式输出了。

----------------------------------------------=------------------------------------------------------------
----------------------------------------------=------------------------------------------------------------
最终答案：
----------------------------------------------1------------------------------------------------------------
【指标波动结论】  
今天整体指标降低了72003（3047331->3119334），原因如下:  
1. 维度影响排序：app_version（surprise_sum = 0.000591）> language（surprise_sum = 0.000000347）  
2. 维度内元素排序：  
   - app_version： 9.8.1（surprise 0.000383）(before 46993 -> after 19710) > 8.86.4（surprise 0.000207）(before 463389 -> after 382996)  
   - language： en（surprise 0.000000238）(before 929755 -> after 911782) > hi（surprise 0.000000109）(before 728372 -> after 709467)
"""