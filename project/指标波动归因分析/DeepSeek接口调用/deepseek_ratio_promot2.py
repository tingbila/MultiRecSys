#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# function: DAU 和 DNU 指标惊讶度分析
# author: zhangmingyang

import sys
import logging
import pandas as pd
from pyhive import hive
import argparse
from openai import OpenAI
from pyhive import hive
import pandas as pd
import json
import requests
import time
import json
import requests

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def execute_complex_sql(partition_value):
    """
    执行复杂的 Hive SQL 查询，计算维度下的 EP、惊讶度、排序等指标。
    """
    conn = hive.Connection(host='xxx', port=7001)

    sql = f"""
    with base_info as (
         select
               dim,            
               element,
               cast(before as bigint)  as before,
               cast(after  as bigint)  as after
         from  starx_ads.ads_sm_ug_new_device_retention_ratio_adtributor_di
         where dt = '{partition_value}'
    ),
    m1_and_m2 as (
         SELECT
               dim,
               element,
               m1_before,
               m1_after,
               m1_pre_sum,
               m1_aft_sum,
               m1_p,
               m1_q,
               m1_surprise,
               m1_ep,
               m2_before,
               m2_after,
               m2_pre_sum,
               m2_aft_sum,
               m2_p,
               m2_q,
               m2_surprise,
               m2_ep,
               m1_m2_ep,                  
               m1_m2_surprise,              
               ROUND(m1_m2_ep / ROUND(sum(m1_m2_ep) over (partition by dim),12),12) as m1_m2_ep_normalization
         from (
               SELECT    
                     dim,
                     element,    
                     m1_before,
                     m1_after,
                     m1_pre_sum,
                     m1_aft_sum,
                     m1_p,
                     m1_q,
                     m1_surprise,
                     m1_ep,
                     m2_before,
                     m2_after,
                     m2_pre_sum,
                     m2_aft_sum,
                     m2_p,
                     m2_q,
                     m2_surprise,
                     m2_ep,
                     ROUND(((m1_after - m1_before) * m2_pre_sum - (m2_after - m2_before) * m1_pre_sum) / (m2_pre_sum * (m2_pre_sum + m2_after - m2_before)), 12) as m1_m2_ep, 
                     ROUND(COALESCE(m1_surprise, 0) + COALESCE(m2_surprise, 0), 12) AS m1_m2_surprise
               from (
                     SELECT    
                           regexp_extract(dim, '^(.*)_[^_]+$', 1) as dim,
                           element,  
                           MAX(IF(dim rlike '分子', before, null))                      AS m1_before,
                           MAX(IF(dim rlike '分子', after, null))                       AS m1_after,
                           MAX(IF(dim rlike '分子', pre_sum, null))                     AS m1_pre_sum,
                           MAX(IF(dim rlike '分子', aft_sum, null))                     AS m1_aft_sum,
                           MAX(IF(dim rlike '分子', p, null))                           AS m1_p,
                           MAX(IF(dim rlike '分子', q, null))                           AS m1_q,
                           MAX(IF(dim rlike '分子', surprise, null))                    AS m1_surprise,
                           MAX(IF(dim rlike '分子', ep, null))                          AS m1_ep,  
                           MAX(IF(dim rlike '分母', before, null))                      AS m2_before,
                           MAX(IF(dim rlike '分母', after, null))                       AS m2_after,
                           MAX(IF(dim rlike '分母', pre_sum, null))                     AS m2_pre_sum,
                           MAX(IF(dim rlike '分母', aft_sum, null))                     AS m2_aft_sum,
                           MAX(IF(dim rlike '分母', p, null))                           AS m2_p,
                           MAX(IF(dim rlike '分母', q, null))                           AS m2_q,
                           MAX(IF(dim rlike '分母', surprise, null))                    AS m2_surprise,
                           MAX(IF(dim rlike '分母', ep, null))                          AS m2_ep
                     from (
                           select
                                 t3.dim,
                                 t3.element,
                                 t3.before,
                                 t3.after,
                                 t3.pre_sum,
                                 t3.aft_sum,
                                 t3.p,
                                 t3.q,
                                 ROUND(0.5 * (p * LN(2 * p / (p + q)) / LN(10) + q * LN(2 * q / (p + q)) / LN(10)), 12)  as surprise,
                                 ROUND((after - before) / (aft_sum - pre_sum), 12) as ep
                           from (
                                 select
                                       t1.dim,
                                       t1.element,
                                       t1.before,
                                       t1.after,
                                       t2.pre_sum,
                                       t2.aft_sum,   
                                       ROUND(ABS(t1.before) / ABS(t2.pre_sum), 12) AS p,
                                       ROUND(ABS(t1.after)  / ABS(t2.aft_sum), 12) AS q
                                 from  base_info t1
                                 left  join (  
                                       select
                                             dim,
                                             sum(before) as pre_sum,
                                             sum(after)  as aft_sum
                                       from  base_info
                                       group by dim
                                 ) t2
                                 on t1.dim = t2.dim
                           ) t3
                     ) t4
                     group by regexp_extract(dim, '^(.*)_[^_]+$', 1),element         
               ) t5
         ) t6
    )
    
    
    
    
    select     
          dim,
          element, 
          m1_before,
          m1_after,
          m1_pre_sum,
          m1_aft_sum,
          m1_p,
          m1_q,
          m1_surprise,
          m1_ep,   
          m2_before,
          m2_after,
          m2_pre_sum,
          m2_aft_sum,
          m2_p,
          m2_q,
          m2_surprise,
          m2_ep,  
          surprise,
          surprise_rank,
          ep,
          ep_sum,
          lag_ep_sum,
          surprise_sum,
          overall_dim_surprise_rank
    from (
          select
                dim,
                element,
                m1_before,
                m1_after,
                m1_pre_sum,
                m1_aft_sum,
                m1_p,
                m1_q,
                m1_surprise,
                m1_ep,
                m2_before,
                m2_after,
                m2_pre_sum,
                m2_aft_sum,
                m2_p,
                m2_q,
                m2_surprise,
                m2_ep, 
                ep,
                surprise,
                surprise_rank,
                ep_sum,
                lag_ep_sum,
                surprise_sum,
                dense_rank() over (order by surprise_sum desc) as overall_dim_surprise_rank
          from (
                select
                      dim,
                      element,
                      m1_before,
                      m1_after,
                      m1_pre_sum,
                      m1_aft_sum,
                      m1_p,
                      m1_q,
                      m1_surprise,
                      m1_ep, 
                      m2_before,
                      m2_after,
                      m2_pre_sum,
                      m2_aft_sum,
                      m2_p,
                      m2_q,
                      m2_surprise,
                      m2_ep,
                      ep,
                      surprise,
                      surprise_rank,
                      ep_sum,
                      lag_ep_sum,
                      sum(surprise) over (partition by dim) as surprise_sum
                from (
                      select       
                            dim,
                            element, 
                            m1_before,
                            m1_after,
                            m1_pre_sum,
                            m1_aft_sum,
                            m1_p,
                            m1_q,
                            m1_surprise,
                            m1_ep, 
                            m2_before,
                            m2_after,
                            m2_pre_sum,
                            m2_aft_sum,
                            m2_p,
                            m2_q,
                            m2_surprise,
                            m2_ep,
                            ep,
                            surprise,
                            surprise_rank,
                            ep_sum,
                            lag(ep_sum,1,ep_sum) over (partition by dim order by surprise_rank asc) as lag_ep_sum
                      from (
                            select     
                                  dim,
                                  element,    
                                  m1_before,
                                  m1_after,
                                  m1_pre_sum,
                                  m1_aft_sum,
                                  m1_p,
                                  m1_q,
                                  m1_surprise,
                                  m1_ep,    
                                  m2_before,
                                  m2_after,
                                  m2_pre_sum,
                                  m2_aft_sum,
                                  m2_p,
                                  m2_q,
                                  m2_surprise,
                                  m2_ep,  
                                  ep,
                                  surprise,
                                  surprise_rank, 
                                  sum(abs(ep)) over (partition by dim order by surprise_rank asc rows between unbounded preceding and current row ) as ep_sum
                            from (
                                  select                            
                                       dim,
                                       element,    
                                       m1_before,
                                       m1_after,
                                       m1_pre_sum,
                                       m1_aft_sum,
                                       m1_p,
                                       m1_q,
                                       m1_surprise,
                                       m1_ep,   
                                       m2_before,
                                       m2_after,
                                       m2_pre_sum,
                                       m2_aft_sum,
                                       m2_p,
                                       m2_q,
                                       m2_surprise,
                                       m2_ep,  
                                       m1_m2_ep_normalization as ep,
                                       m1_m2_surprise         as surprise,  
                                       row_number() over (partition by dim order by m1_m2_surprise desc) as surprise_rank
                                  from m1_and_m2
                            ) t1
                                  
                            where abs(t1.ep) >= 0.2         
                      ) t2
                ) t3
                where t3.ep_sum <= 0.8 or (t3.ep_sum > 0.8 and t3.lag_ep_sum < 0.8)
          ) t4
    ) t5  
    where t5.overall_dim_surprise_rank <= 2
    """

    logging.info("执行 SQL 查询中...")
    df = pd.read_sql(sql, conn)
    return df


def build_prompt(csv_data, partition_dt):
    prompt_system = """
    你是一名数据分析专家，擅长使用 Adtributor 算法进行归因分析和指标波动解读。
    我将给你不同维度的归因指标数据，请你根据数据直接给出简明扼要的结论，
    重点总结各维度及元素对整体波动的影响大小，避免解释算法细节，只输出结论和排序。
    """

    prompt_user = f"""
    请根据以下数据，基于 Adtributor 归因算法的思想，结合数据给出指标波动结论。：

    {csv_data}

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


def call_deepseek_api(csv_data: str, partition_dt:str,api_key: str, base_url: str,model_name="deepseek-r1") -> str:
    """
    调用 DeepSeek API 完成分析任务。
    :param csv_data: 查询结果 CSV 字符串
    :param api_key: API 密钥
    :param base_url: API 基础 URL
    :param model_name: 使用的模型名称
    :return: 返回的模型最终内容字符串
    """
    messages = build_prompt(csv_data,partition_dt)

    headers = {"Content-Type": "application/json; charset=utf-8"}  # 可移除，如 OpenAI SDK 自处理
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        reasoning_effort="high",  # 注意检查是否必须参数
        stream=False
    )

    print("messages内容：")
    print(messages)
    print("------------------------------------------------------------")
    print("思考过程：")
    print(completion.choices[0].message.reasoning_content)
    print("------------------------------------------------------------")
    print("最终答案：")
    raw_output = completion.choices[0].message.content
    cleaned_output = raw_output.replace("```json", "").replace("```", "").strip()

    return cleaned_output


def send(chat_id, tat, userid, userid2, today, msg_info):
    url = "https://open.feishu.cn/open-apis/im/v1/messages"
    params = {"receive_id_type": "chat_id"}
    con = {
        "header": {
            "template": "blue",
            "title": {
                "content": f"📊 DeepSeek-DAU次日留存率归因模块（{today}）",  # 拼接 today 日期
                "tag": "plain_text"
            }
        },
        "i18n_elements": {
            "zh_cn": [
                {
                    "tag": "hr"
                },
                {
                    "tag": "markdown",
                    "content": msg_info
                }
            ]
        }
    }
    req = {
        "receive_id": chat_id,  # chat_id
        "content": json.dumps(con),
        "msg_type": "interactive",  # image,text,file,post
    }
    payload = json.dumps(req)
    headers = {
        'Authorization': 'Bearer ' + tat,  # your access token
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, params=params, headers=headers, data=payload)



# 主程序入口
if __name__ == '__main__':
    try:
        if len(sys.argv) < 2:
            logging.error("Usage: python script.py <分区日期>")
            sys.exit(1)

        partition_dt = sys.argv[1]
        logging.info("分析分区日期：%s", partition_dt)

        # 1. 查询数据并转换为 CSV 格式
        result_df = execute_complex_sql(partition_dt)
        csv_data = result_df.to_csv(index=False)
        logging.info("查询完成，展示数据:")
        print(csv_data)

        # 2. 调用 DeepSeek 模型分析
        result = call_deepseek_api(
            csv_data,
            partition_dt,
            api_key="xxx",
            base_url="xxxx",
        )
        print("------------------------------------------------------------")
        print("清洗后的最终结果：")
        print(result)

        # 3. 调用飞书接口发送消息
        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal/"
        post_data = {"app_id": "xxx","app_secret": "xxx"}
        r = requests.post(url, data=post_data)
        tat = r.json()["tenant_access_token"]
        print(tat)
        chat_id = 'xxx'
        msg_info = result

        send(chat_id, tat,'mingyang.zhang@ushow.media', 'mingyang.zhang@ushow.media', partition_dt, msg_info)

    except Exception as e:
        logging.error("执行过程中发生错误：", exc_info=e)