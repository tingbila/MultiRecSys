-- 1. 根据实验，这应该是一个回归问题，而不是一个分类问题，这里面的回归用的是指标的波动相减做的target
-- 2. 在原始的 FM 特征差值统计 SQL 上，添加了 HAVING ABS(...) > 100 的条件，这起到了两个目的：过滤噪声样本（即无明显变化的样本）
-- 可以有效去除这类“无显著变化”的样本，使得分析更聚焦于：
-- 真实波动人群
-- 影响 DAU 波动的核心维度组合
select
      -- FM算法的维度
	  platform,
	  app_name,
	  app_version,
	  country,
	  region,
	  language,
	  channel,
	  create_date,
	  active_last_date,
	  -- FM算法的指标
		(SUM(IF(dt = '${end_dt}', dau, 0)) - SUM(IF(dt = '${start_dt}', dau, 0))) as dau_diff
from (
	select
	      dt,
		  platform,
		  app_name,
		  app_version,
		  country,
		  region,
		  language,
		  channel,
		  create_date,
		  active_last_date,
		  dau
	from starx_ads.ads_sm_flow_device_indicators_di
	where dt between '${start_dt}' and '${end_dt}'
) t1
group by
		  platform,
		  app_name,
		  app_version,
		  country,
		  region,
		  language,
		  channel,
		  create_date,
		  active_last_date
having abs((SUM(IF(dt = '${end_dt}', dau, 0)) - SUM(IF(dt = '${start_dt}', dau, 0)))) > 100

-- 3. 调用FM算法进行训练，我们会发现LOSS损失一直在变小,当模型训练停止时候;
-- import numpy as np
--
-- # cross_weights 是对称矩阵，V_matrix 是 (num_features, emb_size)
-- num_features = cross_weights.shape[0]
--
-- # 保存所有特征对及其交互值（只保留上三角非对角）
-- interactions = []
-- for i in range(num_features):
--   for j in range(i + 1, num_features):
--       interactions.append(((i, j), cross_weights[i, j]))
--
-- # 按绝对值排序（从大到小）
-- top_k = sorted(interactions, key=lambda x: abs(x[1]), reverse=True)[:10]
--
-- # 输出 Top10 特征交互对
-- print("Top 10 特征交互对（按交互强度）:")
-- column_names = ["platform", "app_name", "app_version", "country", "region", "language", "channel", "create_date","active_last_date"]
-- for (i, j), weight in top_k:
--   name_i = column_names[i]
--   name_j = column_names[j]
--   print(f"{name_i} × {name_j} : 权重 = {weight:.6f}")
--
-- # Top 10 特征交互对（按交互强度）:
-- # create_date × active_last_date : 权重 = 0.066382
-- # app_version × active_last_date : 权重 = 0.032802
-- # platform × active_last_date : 权重 = -0.029710
-- # app_version × create_date : 权重 = 0.015355
-- # platform × create_date : 权重 = -0.015010l
-- # app_name × app_version : 权重 = 0.011878
-- # country × active_last_date : 权重 = -0.010133
-- # country × create_date : 权重 = -0.009827
-- # app_name × active_last_date : 权重 = 0.009772