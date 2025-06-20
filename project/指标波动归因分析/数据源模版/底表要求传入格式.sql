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
-- function：次留留存率指标监控
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








-----------------------------------------------------------------------------
-- author：张明阳
-- create：2025年6月20日17:24:00
-- function：净利计算
-- document:
------------------------------------------------------------------------------
alter table starx_ads.ads_sm_revenue_global_income_cost_info_adtributor_di drop if exists partition (dt = '${end_dt}');


insert overwrite table starx_ads.ads_sm_revenue_global_income_cost_info_adtributor_di partition (dt='${end_dt}')
select
      dim,
      value as element,
      sum(case when dt = '${start_dt}' then money else 0 end) as before,
      sum(case when dt = '${end_dt}'   then money else 0 end) as after
from (
    select
          dt,
          split(dim_value, ':')[0] as dim,
          split(dim_value, ':')[1] as value,
          split(dim_value, ':')[2] as money
    from (
          SELECT
                dt,
                concat_ws(',',
                       concat('净利', ':', '充值收入', ':',charge_amount),
                       concat('净利', ':', '广告收入', ':',ads_amount),
                       concat('净利', ':', '订阅收入', ':',sub_amount),
                       concat('净利', ':', 'exchange转金币收入', ':',exchange_amount),
                       concat('净利', ':', '金币调整收入', ':',gold_balance_amount),
                       concat('净利', ':', '其他收入', ':',other_revenue),

                       concat('净利', ':', '渠道税费成本', ':',-channel_cost),
                       concat('净利', ':', '语音房产出钻石', ':',-voice_diamond),
                       concat('净利', ':', '直播产出钻石', ':',-live_diamond),
                       concat('净利', ':', '作品送礼产出钻石', ':',-recording_diamond),
                       concat('净利', ':', '其他产出钻石', ':',-other_diamond),

                       concat('净利', ':', '语音房主播工资', ':',-voice_broadcast_salary),
                       concat('净利', ':', '语音房工会长工资', ':',-voice_president_salary),
                       concat('净利', ':', '直播主播工资', ':',-live_broadcast_salary),
                       concat('净利', ':', '直播工会长工资', ':',-live_president_salary),

                       concat('净利', ':', '语音房活动成本', ':',-voice_activity_profit),
                       concat('净利', ':', '直播活动成本', ':',-live_activity_profit),
                       concat('净利', ':', '活动成本', ':',-activity_profit),
                       concat('净利', ':', '本地活动成本', ':',-local_activity_profit),
                       concat('净利', ':', '其他活动成本', ':',-other_activity_profit),

                       concat('净利', ':', '家族兑换成本', ':',-family_exchange_profit),
                       concat('净利', ':', '明星家族成本', ':',-star_family_profit),
                       concat('净利', ':', '直播间宝箱成本', ':',-live_box_profit),

                       concat('净利', ':', '其他成本', ':',-other_profit),
                       concat('净利', ':', '捕鱼分成成本', ':',-sys_fish_profit),

                       concat('净利', ':', 'PGC主播工资',':', -pgc_diamond_salary),
                       concat('净利', ':', '弹幕游戏分成成本', ':',-barrage_profit),
                       concat('净利', ':', '发财虎分成成本', ':',-fortunetiger_profit),
                       concat('净利', ':', '全球每日净利成本预估值', ':',-daily_estimate_cost)
                ) AS infos
          from  starx_temp.ads_sm_revenue_global_income_cost_info_di
          where dt in ('${start_dt}', '${end_dt}')
    ) T
    lateral view explode(split(infos, ',')) A as dim_value
) T1
group by dim, value

