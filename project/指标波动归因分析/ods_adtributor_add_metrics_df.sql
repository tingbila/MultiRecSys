
with base_info as (
	 select
	 	   dim,
	 	   element,
	 	   cast(before as bigint)  as before,
	 	   cast(after  as bigint)  as after,
	 	   'on' as join_column
	 from starx_ods.ods_adtributor_add_metrics_df
	 where dt = '20250613'
)


with base_info as (
	 select
	 	   dim,
	 	   element,
	 	   cast(dau_before as bigint)  as before,
	 	   cast(dau_after  as bigint)  as after,
	 	   'on' as join_column
	 from  starx_ads.ads_sm_flow_device_indicators_adtributor_di
	 where dt = '${dt}'
)



select
      dim,
      element,
      before,
      after,
      pre_sum,
      aft_sum,
      p,
      q,
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
            before,
            after,
            pre_sum,
            aft_sum,
            p,
            q,
            surprise,
            surprise_rank,
            ep,
            ep_sum,
            lag_ep_sum,
            surprise_sum,
            -- 10. 按照计算好的各维度S汇总值从大到小对dim进行排序
            dense_rank() over (order by surprise_sum desc) as overall_dim_surprise_rank
      from (
            select
                  dim,
                  element,
                  before,
                  after,
                  pre_sum,
                  aft_sum,
                  p,
                  q,
                  surprise,
                  surprise_rank,
                  ep,
                  ep_sum,
                  lag_ep_sum,
                  -- 9. 在每个维度下汇总各元素的S值，得到各维度S值的汇总结果。
                  sum(surprise) over (partition by dim) as surprise_sum
            from (
                  select
                        dim,
                        element,
                        before,
                        after,
                        pre_sum,
                        aft_sum,
                        p,
                        q,
                        surprise,
                        surprise_rank,
                        ep,
                        ep_sum,
                        -- 取当前行的上一行（1）的 ep_sum 值；如果没有上一行（例如是第一行），就使用默认值 ep_sum（即当前行的值）
                        lag(ep_sum,1,ep_sum) over (partition by dim order by surprise_rank asc) as lag_ep_sum
                  from (
                        select
                              t5.dim,
                              t5.element,
                              t5.before,
                              t5.after,
                              t5.pre_sum,
                              t5.aft_sum,
                              t5.p,
                              t5.q,
                              t5.surprise,
                              t5.surprise_rank,
                              t5.ep,
                              -- 7. 基于6筛选完单个元素EP值之后，在对每个维度下通过筛选的元素EP值进行累加
                              -- 这里额外也添加了一个绝对值
                              -- 这里用绝对值累加，是想统计所有元素贡献度的大小和，忽略正负方向:这样设计是对的，因为你想选出贡献总量达到阈值的元素集。
                              sum(abs(ep)) over (partition by dim order by surprise_rank asc rows between unbounded preceding and current row ) as ep_sum
                        from (
                              select
                                    t4.dim,
                                    t4.element,
                                    t4.before,
                                    t4.after,
                                    t4.pre_sum,
                                    t4.aft_sum,
                                    t4.p,
                                    t4.q,
                                    t4.surprise,
                                    -- 4. 在每个维度内将元素按照惊讶度S从高到低对数据进行排序
                                    row_number() over (partition by dim order by surprise desc) as surprise_rank,
                                    -- 5. 计算贡献率EP:即每个元素波动对于总体波动的贡献，以A渠道为例，A渠道的EP=（A渠道活动后销售额-A渠道活动前销售额）/（总体活动后销售额-总体活动前销售额）。
                                    --  如果不取绝对值，结果的含义:
                                    --      EP 可能为正或负，且整体指标变动可能为正或负
                                    --      EP 值正，表示该元素的变化方向和整体变化方向一致，是“正向贡献”
                                    --      EP 值负，表示元素变化方向与整体变化方向相反，是“负向贡献”
                                    --  优点：
                                    --      真实反映贡献的方向性，能看出哪些元素拉动指标上升，哪些元素抑制指标上升（或者拉低指标）。
                                    -- 这里不加绝对值，保留正负，方便看贡献方向（正向/负向贡献）。
                                    -- 由于某些元素的变动幅度远大于整体变动，故 EP 值可能超过 1 或小于 -1，如果某个元素的变化量大于整体变化量（也就是说这个元素的波动是主要驱动甚至远大于整体的），就会出现 EP > 1 或 EP < -1
                                    -- 📌 举个例子说明：
                                    -- 假设：
                                    --     整体 aft_sum = 1100，pre_sum = 1000，→ 整体增长 100
                                    --     某个元素的 after = 800，before = 600，→ 该元素增长 200
                                    -- 此时：
                                    -- ep = (800 - 600) / (1100 - 1000) = 200 / 100 = 2.0
                                    -- 就得到了一个 EP = 2，说明这个元素对整体增长的贡献超过了100%，是 强正向拉动因素。
                                    ROUND((after - before) / (aft_sum - pre_sum), 12) as ep
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
                                          -- JS散度公式s = 0.5 * (p * math.log10(2 * p / (p + q)) + q * math.log10(2 * q / (p + q)))
                                          -- 3. 惊讶度（Surprise，用S表示）是一个用来衡量指标结构前后变化程度的指标，回答的是"哪个元素的波动最让人惊讶"的问题。
                                          -- JS散度要求概率非负且0~1之间，加绝对值避免负值导致log计算出错
                                          ROUND(0.5 * (p * LN(2 * p / (p + q)) / LN(10) + q * LN(2 * q / (p + q)) / LN(10)), 12)  as surprise
                                    from (
                                          select
                                                t1.dim,
                                                t1.element,
                                                t1.before,
                                                t1.after,
                                                t2.pre_sum,
                                                t2.aft_sum,
                                                -- 2. 计算活动前销售额占比p和活动后销售额占比q
                                                -- 加绝对值可以避免负值导致后续JS散度计算出错:p 和 q 表示概率或占比，理论上是非负且小于等于1的数值。它们是活动前后某元素销售额占总销售额的比例，不应出现负值。
                                                ROUND(ABS(t1.before) / ABS(t2.pre_sum), 12) AS p,
                                                ROUND(ABS(t1.after)  / ABS(t2.aft_sum), 12) AS q
                                          from  base_info t1
                                          left  join (
                                                -- 1. 先计算活动前和活动后销售额的总体数据-对于每个维度pre_sum和aft_sum应该是一样的
                                                select
                                                      sum(before) as pre_sum,
                                                      sum(after)  as aft_sum,
                                                      'on' as join_column
                                                from  base_info
                                                where dim = 'region'   -- 选择一个渠道即可，因为所有维度的sum(after)和sum(before)是一样的
                                          ) t2
                                          on t1.join_column = t2.join_column
                                    ) t3
                              ) t4
                        ) t5
                        -- 6. 根据设定的单个元素EP阈值，遍历所有元素的EP值是否高于0.2，如果高于，则通过筛选
                        where abs(t5.ep) >= 0.2   -- 这里加了一个绝对值
                  ) t6
            ) t7
            -- 8. 整体EP(单维度下)（波动贡献率）的筛选：意味着只要选中元素贡献率之和超过60%，就已经能够解释大部分波动原因了
            -- 在根据总EP阈值批量筛选时:包含第一个大于总EP阈值的元素:lag_ep_sum是为了处理这种情况的
            -- 0.5  0.5
            -- 0.9  0.5
            where t7.ep_sum <= 0.8 or (t7.ep_sum > 0.8 and t7.lag_ep_sum < 0.8)
      ) t8
) t9
-- 11. 假设我们最终的目标是筛选影响最大的top2的维度进行原因定位
where t9.overall_dim_surprise_rank <= 2




