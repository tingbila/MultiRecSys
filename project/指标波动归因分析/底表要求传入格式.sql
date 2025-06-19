-----------------------------------------------------------------------------
-- author：张明阳
-- create：2025年6月16日12:40:18
-- function：DAU和DNU指标监控
-- document:
------------------------------------------------------------------------------


alter table starx_ads.ads_sm_flow_device_indicators_adtributor_di drop if exists partition (dt = '${end_dt}');


insert overwrite table starx_ads.ads_sm_flow_device_indicators_adtributor_di partition (dt='${end_dt}')
select
      dim,
      value as element,
      sum(case when dt = '${start_dt}' then dau else 0 end) as dau_before,
      sum(case when dt = '${end_dt}' then dau else 0 end)   as dau_after,
      sum(case when dt = '${start_dt}' then install_device_cnt else 0 end) as install_before,
      sum(case when dt = '${end_dt}' then install_device_cnt else 0 end)   as install_after
from (
    select
          dt,
          split(dim_value, ':')[0] as dim,
          split(dim_value, ':')[1] as value,
          dau,
          install_device_cnt
    from (
          select
                dt,
                concat_ws(',',
                    concat('platform', ':', coalesce(platform, 'other')),
                    concat('app_name', ':', coalesce(app_name, 'other')),
                    concat('app_version', ':', coalesce(app_version, 'other')),
                    concat('country', ':', coalesce(country, 'other')),
                    concat('region', ':', coalesce(region, 'other')),
                    concat('language', ':', coalesce(language, 'other')),
                    concat('channel', ':', coalesce(channel, 'other')),
                    concat('create_date', ':', coalesce(create_date, 'other')),
                    concat('active_last_date', ':', coalesce(active_last_date, 'other'))
                ) as infos,
                dau,
                install_device_cnt
          from  starx_ads.ads_sm_flow_device_indicators_di
          where dt in ('${start_dt}', '${end_dt}')
    ) T
    lateral view explode(split(infos, ',')) A as dim_value
) T1
group by dim, value






-----------------------------------------------------------------------------
-- author：张明阳
-- create：2025年6月16日12:40:18
-- function：次留指标监控
-- document:
------------------------------------------------------------------------------

alter table starx_ads.ads_sm_ug_new_device_retention_ratio_adtributor_di drop if exists partition (dt = '${dt}');
with t1 as (
     select
           dt,
           reg_date,
           split(dim_value, ':')[0] as dim,
           split(dim_value, ':')[1] as value,
           new_device,
           1day_retention
     from (
           select
                 dt,
                 reg_date,
                 concat_ws(',',
                       concat('platform', ':', coalesce(platform, 'other')),
                       concat('app_name', ':', coalesce(app_name, 'other')),
                       concat('app_version', ':', coalesce(app_version, 'other')),
                       concat('region', ':', coalesce(region, 'other')),
                       concat('country', ':', coalesce(country, 'other')),
                       concat('channel', ':', coalesce(channel, 'other')),
                       concat('channel_type', ':', coalesce(channel_type, 'other')),
                       concat('language', ':', coalesce(language, 'other'))
                 ) as infos,
                 new_device,
                 1day_retention
           from  starx_da_ads.ads_sm_ug_new_device_retention_di
           where dt='${dt}'
           and   reg_date in ('${start_dt}', '${end_dt}')  -- 基准日期
     ) T
     lateral view explode(split(infos, ',')) A as dim_value
)



--  	dim	        element	    before               after
-- 1	region_分子	Area_ME	    4450	             3858
-- 2	region_分母	Area_ME	    12568	             10854
insert overwrite table starx_ads.ads_sm_ug_new_device_retention_ratio_adtributor_di partition (dt='${dt}')
select
      concat(dim,'_','分子') as dim,
      value as element,
      sum(case when reg_date = '${start_dt}' then 1day_retention else 0 end) as before,
      sum(case when reg_date = '${end_dt}' then 1day_retention else 0 end)   as after
from  t1
group by  dim, value
union all
select
      concat(dim,'_','分母') as dim,
      value as element,
      sum(case when reg_date = '${start_dt}' then new_device else 0 end) as before,
      sum(case when reg_date = '${end_dt}' then new_device else 0 end)   as after
from  t1
group by  dim, value