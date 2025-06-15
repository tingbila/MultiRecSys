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

    dim    element  surprise   surprise_rank  ep        ep_sum     surprise_sum  overall_dim_surprise_rank
    渠道    A        0.001973   2              0.373333  0.64       0.010395      1
    渠道    C        0.000725   4              0.36      1          0.010395      1
    渠道    B        0.007697   1              0.266667  0.266667   0.010395      1
    新老客  老客     0.000426   2               0.666667  1          0.000983      2
    新老客  新客     0.000557   1               0.333333  0.333333   0.000983      2

    请完成以下任务：
    1. 按 surprise_sum 排序，指出哪个维度对整体影响最大；
    2. 对每个维度内的元素，按影响大小排序。

    示例格式：
    【指标波动结论】
    1. 维度影响排序：A维度（surprise_sum = X）> B维度（surprise_sum = X）
    2. 维度内元素排序：
       - 渠道维度：  a1（surprise 0.007697）> a2（surprise 0.001973）> a3（surprise 0.000725）
       - 新老客维度：b1（surprise 0.000557）>  b2（surprise 0.000426）
    注意：只输出结论，不要解释算法或过程。
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
[hadoop()@sg-prod-datacentersg-emrrouter-1 zmy]$ /home/hadoop/hao.liu/python3/bin/python3.12   bb.py
messages内容：
[{'role': 'system', 'content': '\n    你是一名数据分析专家，擅长使用 Adtributor 算法进行归因分析和指标波动解读。\n    我将给你不同维度的归因指标数据，请你根据数据直接给出简明扼要的结论，\n    重点总结各维度及元素对整体波动的影响大小，避免解释算法细节，只输出结论和排序。\n    '}, {'role': 'user', 'content': '\n    请根据以下数据，基于 Adtributor 归因算法的思想，结合数据给出指标波动结论。：\n\n    dim    element  surprise   surprise_rank  ep        ep_sum     surprise_sum  overall_dim_surprise_rank\n    渠道    A        0.001973   2              0.373333  0.64       0.010395      1\n    渠道    C        0.000725   4              0.36      1          0.010395      1\n    渠道    B        0.007697   1              0.266667  0.266667   0.010395      1\n    新老客  老客     0.000426   2               0.666667  1          0.000983      2\n    新老客  新客     0.000557   1               0.333333  0.333333   0.000983      2\n\n    请完成以下任务：\n    1. 按 surprise_sum 排序，指出哪个维度对整体影响最大；\n    2. 对每个维度内的元素，按影响大小排序。\n\n    示例格式：\n    【指标波动结论】\n    1. 维度影响排序：A维度（surprise_sum = X）> B维度（surprise_sum = X）\n    2. 维度内元素排序：\n       - 渠道维度：  a1（surprise 0.007697）> a2（surprise 0.001973）> a3（surprise 0.000725）\n       - 新老客维度：b1（surprise 0.000557）>  b2（surprise 0.000426）\n    注意：只输出结论，不要解释算法或过程。\n    '}]
----------------------------------------------=------------------------------------------------------------
----------------------------------------------=------------------------------------------------------------
思考过程：
嗯，首先我要看看用户给的这些数据。任务是使用Adtributor算法来分析指标波动，但不需要讲算法细节，只要结论和排序。数据里有渠道和新老客两个维度，每个维度下的元素对应的surprise值和相关排名。还有ep、ep_sum、surprise_sum这些字段。
首先第一个任务是按surprise_sum排序，找出哪个维度对整体影响最大。看数据里的surprise_sum，渠道维度是0.010395，新老客是0.000983。显然渠道的surprise_sum更大，所以渠道维度影响更大。
接下来第二个任务是对每个维度内的元素按影响大小排序。维度内的元素需要根据各自的surprise值来排。渠道维度下有A、C、B三个元素，对应的surprise值分别是0.001973，0.000725，0.007697。比较这三个，最大的B的0.007697，然后是A，最后是C。所以渠道维度内的排序应该是B > A > C。
新老客维度有老客和新客，surprise分别是0.000426和0.000557。显然新客的surprise更高，所以新客排第一，老客第二。
检查一下有没有哪里弄错。比如注意surprise_rank这个列，渠道中的B是排第一，对应的surprise也是最大的0.007697，没问题。同样新老客维度里新客的surprise_rank是1，对得上。这样得出的结论应该准确。整体的话，按照用户的示例格式来输出结果，确认数据没看错的话就完成任务了。

----------------------------------------------=------------------------------------------------------------
----------------------------------------------=------------------------------------------------------------
最终答案：
----------------------------------------------1------------------------------------------------------------
【指标波动结论】  
1. 维度影响排序：渠道维度（surprise_sum = 0.010395）> 新老客维度（surprise_sum = 0.000983）  
2. 维度内元素排序：  
   - 渠道维度：B（surprise 0.007697）> A（surprise 0.001973）> C（surprise 0.000725）  
   - 新老客维度：新客（surprise 0.000557）> 老客（surprise 0.000426）
"""

"""
[hadoop()@sg-prod-datacentersg-emrrouter-1 zmy]$ /home/hadoop/hao.liu/python3/bin/python3.12   bb.py
messages内容：
[{'role': 'system', 'content': '\n    你是一名数据分析专家，擅长使用 Adtributor 算法进行归因分析和指标波动解读。\n    我将给你不同维度的归因指标数据，请你根据数据直接给出简明扼要的结论，\n    重点总结各维度及元素对整体波动的影响大小，避免解释算法细节，只输出结论和排序。\n    '}, {'role': 'user', 'content': '\n    请根据以下数据，基于 Adtributor 归因算法的思想，结合数据给出指标波动结论。：\n\n    dim,element,before,after,pre_sum,aft_sum,p,q,surprise,surprise_rank,ep,ep_sum,lag_ep_sum,surprise_sum,overall_dim_surprise_rank\n    channel,organic,1624032,1656180,3056814,3121483,0.531282570677,0.530574730024,5.1231e-08,12,0.49711608344,0.888617421021,0.391501337581,1.054355e-06,1\n    channel,googleadwords_int,859705,885023,3056814,3121483,0.281242169134,0.28352645201,1.003124e-06,1,0.391501337581,0.391501337581,0.391501337581,1.054355e-06,1\n    region,Area_IN,1119484,1143835,3056814,3121483,0.366225750078,0.36643960579,6.777e-09,24,0.376548268877,0.376548268877,0.376548268877,6.777e-09,2\n\n    请完成以下任务：\n    1. 按 surprise_sum 排序，指出哪个维度对整体影响最大；\n    2. 对每个维度内的元素，按影响大小排序。\n\n    示例格式：\n    【指标波动结论】\n    1. 维度影响排序：A维度（surprise_sum = X）> B维度（surprise_sum = X）\n    2. 维度内元素排序：\n       - 渠道维度：  a1（surprise 0.007697）> a2（surprise 0.001973）> a3（surprise 0.000725）\n       - 新老客维度：b1（surprise 0.000557）>  b2（surprise 0.000426）\n    注意：只输出结论，不要解释算法或过程。\n    '}]
----------------------------------------------=------------------------------------------------------------
----------------------------------------------=------------------------------------------------------------
思考过程：
嗯，我现在需要根据用户提供的Adtributor算法的数据来分析指标波动。首先，用户要求按照surprise_sum对维度进行排序，并指出哪个维度对整体影响最大。然后要对每个维度内的元素按影响大小排序。

首先看数据里的维度，有三个行数据：channel有两个元素（organic和googleadwords_int），region有一个元素（Area_IN）。用户给出的数据中还有dim列，分别是channel和region。接下来要看每个维度的surprise_sum值，用户示例中的surprise_sum是每个维度的总和。例如，在数据中，channel维度的两条记录的surprise_sum都是1.054355e-06，而region的是6.777e-09。不过我需要仔细看一下每个维度对应的surprise_sum值是否正确，因为原始数据中每个维度可能有多个元素，surprise_sum应该是该维度所有元素的surprise相加后的总和？

看数据里的每一行，每个维度元素的surprise值：对于channel的第一个元素organic的surprise是5.1231e-08，第二个googleadwords_int的是1.003124e-06。而它们的surprise_sum在同一维度（channel）下的值都是1.054355e-06，这可能是该维度下两个元素的surprise相加后的结果，5.1231e-08 + 1.003124e-06 ≈ 1.054355e-06，这没错。然后region维度的元素Area_IN的surprise是6.777e-09，对应的surprise_sum也是这个值，因为只一个元素。因此，各维度的surprise_sum排序应该是channel的1.054355e-06大于region的6.777e-09。因此，维度影响最大的是channel。

接下来对每个维度内的元素排序。在channel维度里，两个元素的surprise值分别是googleadwords_int的1.003124e-06和organic的5.1231e-08。所以googleadwords_int的影响更大，排在前面。对于region维度只有一个元素Area_IN，自然排第一。

所以结论就是维度影响排序channel>region。维度内元素排序，channel维度下googleadwords_int第一，organic第二。region只有Area_IN。

----------------------------------------------=------------------------------------------------------------
----------------------------------------------=------------------------------------------------------------
最终答案：
----------------------------------------------1------------------------------------------------------------
【指标波动结论】  
1. 维度影响排序：  
channel维度（surprise_sum = 1.054355e-06） > region维度（surprise_sum = 6.777e-09）  

2. 维度内元素排序：  
- channel维度：  
  googleadwords_int（surprise 1.003124e-06） > organic（surprise 5.1231e-08）  
- region维度：  
  Area_IN（surprise 6.777e-09）
"""