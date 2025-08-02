------------------------------------------------------------------------------
-- author：张明阳
-- create：2025年8月3日06:04:32
-- function：-- 功能：潮玩DAU相关指标-归因分析
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


alter table openow_ads.ads_op_flow_device_adtributor_indicators_df drop if exists partition (dt='${dt}');


with t0 as (
     select
           dt,
           country,
           platform,
           app_version,
           af_campaign_id,
           af_adset_id,
           af_ad_id,
           count(deviceid) as new_device_count    -- 当日新增设备数
     from  openow_dws.dws_op_flow_device_indicators_di
     where dt in ('${start_dt}', '${end_dt}')
     and   if_td_create = 1  -- 当日新增设备
     group by  dt,country,platform,app_version,af_campaign_id,af_adset_id,af_ad_id
),
t1 as (
     select
           dim,
           value as element,
           sum(case when dt = '${start_dt}' then new_device_count else 0 end) as before,
           sum(case when dt = '${end_dt}' then new_device_count else 0 end)   as after
     from (
           select
                 dt,
                 split(dim_value, ':')[0] as dim,
                 split(dim_value, ':')[1] as value,
                 new_device_count
           from (
                 select
                       dt,
                       concat_ws(',',
                             concat('country', ':', coalesce(country, 'other')),
                             concat('platform', ':', coalesce(platform, 'other')),
                             concat('app_version', ':', coalesce(app_version, 'other')),
                             concat('af_campaign_id', ':', coalesce(af_campaign_id, 'other')),
                             concat('af_adset_id', ':', coalesce(af_adset_id, 'other')),
                             concat('af_ad_id', ':', coalesce(af_ad_id, 'other'))
                       ) as infos,
                       new_device_count
                 from  t0
           ) T
           lateral view explode(split(infos, ',')) A as dim_value
     ) T1
     group by dim, value
)


--  	dim	        element	    before               after
-- 1	region	  Area_ME	    4450	             3858
-- 2	region	  Area_US	    12568	             10854
insert overwrite table openow_ads.ads_op_flow_device_adtributor_indicators_df partition (dt='${dt}')
select
      a.dim,
      a.element,
      a.before,
      a.after,
      b.pre_sum,
      b.aft_sum
from  t1 a
inner join (
      select
            dim,
            sum(before) as pre_sum,
            sum(after)  as aft_sum
      from  t1
      group by  dim
) b
on a.dim = b.dim