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

