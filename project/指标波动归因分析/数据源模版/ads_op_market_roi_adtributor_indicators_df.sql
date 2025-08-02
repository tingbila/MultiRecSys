------------------------------------------------------------------------------
-- author：张明阳
-- create：2025年8月2日19:44:45
-- function：-- 功能：潮玩ROI相关指标-归因分析
-- document: 无
------------------------------------------------------------------------------

-- 资源参数
set hive.exec.parallel = true;
set hive.exec.parallel.thread.number=8;
set spark.default.parallelism=200;
set spark.sql.shuffle.partitions = 200;
set hive.auto.convert.join=false;
set hive.auto.convert.join.noconditionaltask=false;
set spark.sql.autoBroadcastJoinThreshold=-1;
set hive.strict.checks.no.partition.filter=false;
set hive.mapred.mode=nonstrict;
set hive.vectorized.execution.enabled=false;
set hive.vectorized.execution.reduce.enabled=false;


alter table openow_ads.ads_op_market_roi_adtributor_indicators_df drop if exists partition (dt='${dt}');


with t1 as (
     select
           dt,
           active_date as reg_date,
           split(dim_value, ':')[0] as dim,
           split(dim_value, ':')[1] as value,
           mobi_cost_amt,
           charge_0d_usd_amt
     from (
           select
                 dt,
                 active_date,
                 concat_ws(',',
                       concat('country', ':', coalesce(country, 'other')),
                       concat('platform', ':', coalesce(platform, 'other')),
                       concat('channel', ':', coalesce(channel, 'other')),
                       concat('af_campaign_id', ':', coalesce(af_campaign_id, 'other')),
                       concat('af_adset_id', ':', coalesce(af_adset_id, 'other')),
                       concat('af_ad_id', ':', coalesce(af_ad_id, 'other')),
                       concat('material_id', ':', coalesce(material_id, 'other')),
                       concat('model_id', ':', coalesce(model_id, 'other'))
                 ) as infos,
                 mobi_cost_amt,      -- 分母
                 charge_0d_usd_amt   -- 分子
           from  openow_ads.ads_op_market_roi_indicators_df
           where dt = '${dt}'
           and   active_date  in ('${start_dt}', '${end_dt}')  -- 基准日期
     ) T
     lateral view explode(split(infos, ',')) A as dim_value
)






--  	dim	        element	    before               after
-- 1	region_分子	Area_ME	    4450	             3858
-- 2	region_分母	Area_ME	    12568	             10854
insert overwrite table openow_ads.ads_op_market_roi_adtributor_indicators_df partition (dt='${dt}')
select
      a.dim,
      a.element,
      a.before,
      a.after,
      b.pre_sum,
      b.aft_sum
from (
      select
            dt,
            concat(dim,'_','分子') as dim,
            value as element,
            sum(case when reg_date = '${start_dt}' then charge_0d_usd_amt else 0 end) as before,
            sum(case when reg_date = '${end_dt}' then charge_0d_usd_amt else 0 end)   as after
      from  t1
      group by  dt,dim, value
) a
inner  join (
      select
            dt,
            concat(dim,'_','分子') as dim,
            sum(case when reg_date = '${start_dt}' then charge_0d_usd_amt else 0 end) as pre_sum,
            sum(case when reg_date = '${end_dt}' then charge_0d_usd_amt else 0 end)   as aft_sum
      from  t1
      group by  dt,dim
) b
on a.dt = b.dt and a.dim = b.dim

union all

select
      a.dim,
      a.element,
      a.before,
      a.after,
      b.pre_sum,
      b.aft_sum
from (
      select
            dt,
            concat(dim,'_','分母') as dim,
            value as element,
            sum(case when reg_date = '${start_dt}' then mobi_cost_amt else 0 end) as before,
            sum(case when reg_date = '${end_dt}' then mobi_cost_amt else 0 end)   as after
      from  t1
      group by  dt,dim, value
) a
inner  join (
      select
            dt,
            concat(dim,'_','分母') as dim,
            sum(case when reg_date = '${start_dt}' then mobi_cost_amt else 0 end) as pre_sum,
            sum(case when reg_date = '${end_dt}' then mobi_cost_amt else 0 end)   as aft_sum
      from  t1
      group by  dt,dim
) b
on a.dt = b.dt and a.dim = b.dim