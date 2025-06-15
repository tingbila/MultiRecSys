-- CREATE EXTERNAL TABLE `ads_sm_flow_device_indicators_di`(
--   -- 联合主键
--   `platform` string COMMENT '操作系统平台',
--   `app_name` string COMMENT 'app类型',
--   `app_version` string COMMENT 'app软件版本号',
--   `country` string COMMENT '国家',
--   `region` string COMMENT '大区',
--   `language` string COMMENT '语言',
--   `channel` string COMMENT '渠道',
--   `create_date` string COMMENT '设备新增日期 default:2022-06-14',
--   `active_last_date` string COMMENT '设备上次活跃日期 default:2022-06-14',
--
--   -- 设备指标数据
--   `dau` bigint COMMENT '日活设备数',   -- count(deviceid)
--   `install_device_cnt` bigint COMMENT '新增设备数')  -- sum(if(is_new_device = 1, 1, 0))
-- COMMENT '设备指标报表'
-- PARTITIONED BY (
--   `dt` string COMMENT '统计日期')  -- 基准日期
-- ROW FORMAT SERDE
--   'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe'
-- STORED AS INPUTFORMAT
--   'org.apache.hadoop.mapred.TextInputFormat'
-- OUTPUTFORMAT
--   'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'
-- LOCATION
--   'cosn://starmaker-analytics-sg-1256122840/data_warehouse/starx_ads/ads_sm_flow_device_indicators_di'
-- TBLPROPERTIES (
--   'transient_lastDdlTime'='1677215037'
-- )



-- use  starx_ads;
-- drop table ads_sm_flow_device_indicators_adtributor_di;
-- CREATE EXTERNAL TABLE `ads_sm_flow_device_indicators_adtributor_di`(
--   `dim` string COMMENT 'dim',
--   `element` string COMMENT 'element',
--   `dau_before` bigint COMMENT 'dau_before',
--   `dau_after` bigint COMMENT 'dau_after',
--   `install_before` bigint COMMENT 'install_before',
--   `install_after` bigint COMMENT 'install_after'
-- )
-- COMMENT 'ads_sm_flow_device_indicators_adtributor_di指标归因'
-- PARTITIONED BY (
--   `dt` string)
-- ROW FORMAT SERDE
--   'org.apache.hadoop.hive.ql.io.orc.OrcSerde'
-- STORED AS INPUTFORMAT
--   'org.apache.hadoop.hive.ql.io.orc.OrcInputFormat'
-- OUTPUTFORMAT
--   'org.apache.hadoop.hive.ql.io.orc.OrcOutputFormat'
-- LOCATION
--   'cosn://starmaker-analytics-sg-1256122840/data_warehouse/starx_ads/ads_sm_flow_device_indicators_adtributor_di'
-- TBLPROPERTIES (
--   'author'='mingyang.zhang',
--   'orc.compress'='snappy',
--   'primary key'='dim、element',
--   'ttl'='1 year'
-- );




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





