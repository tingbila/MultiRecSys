with base_info as (
	 select
	 	   dim,
	 	   element,
	 	   before,
	 	   after,
	 	   -- 1. 先计算活动前和活动后销售额的总体数据--对于每个维度pre_sum和aft_sum大部分情况应该是一样的，但是有的时候可能不相同。
	 	   pre_sum,
	 	   aft_sum
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
      abs_x,               -- 每个element的绝对值变化量
      abs_x_rank,          -- 维度内对abs_x进行升序
      dim_n,               -- 统计维度内元素个数
      G_fenzi_element,     -- 计算每项系数和绝对值相乘 (2i-n-1)xi
      G_fenmu,             -- 计算基尼系数的分母
      G_fenzi,             -- 基尼系数的分子
      G,                   -- 维度级别的基尼系数
      G_rank,              -- 全局基尼系数从大到小对dim进行降序排序
      ep,                  -- 每个元素波动对于总体波动的贡献
      ep_rank,             -- 维度内对ep降序排序
      ep_sum,              -- 绝对值累加：从第一行到当前行
      lag_ep_sum           -- 取当前行的上一行（1）的 ep_sum 值
from (
      select
            dim,
            element,
            before,
            after,
            pre_sum,
            aft_sum,
            abs_x,
            abs_x_rank,
            dim_n,
            G_fenzi_element,
            G_fenmu,
            G_fenzi,
            G,
            G_rank,
            ep,
            ep_rank,
            ep_sum,
            -- 14. 取当前行的上一行（1）的 ep_sum 值；如果没有上一行（例如是第一行），就使用默认值 ep_sum（即当前行的值）
            lag(ep_sum,1,ep_sum) over (partition by dim order by ep_rank asc) as lag_ep_sum
      from (
            select
                  dim,
                  element,
                  before,
                  after,
                  pre_sum,
                  aft_sum,
                  abs_x,
                  abs_x_rank,
                  dim_n,
                  G_fenzi_element,
                  G_fenmu,
                  G_fenzi,
                  G,
                  G_rank,
                  ep,
                  ep_rank,
                  -- 13. 基于12筛选完单个元素EP值之后，在对每个维度下通过筛选的元素EP值进行累加
                  -- 这里额外也添加了一个绝对值
                  -- 这里用绝对值累加，是想统计所有元素贡献度的大小和，忽略正负方向:这样设计是对的，因为你想选出贡献总量达到阈值的元素集。
                  sum(abs(ep)) over (partition by dim order by ep_rank asc rows between unbounded preceding and current row ) as ep_sum
            from (
                  select
                        dim,
                        element,
                        before,
                        after,
                        pre_sum,
                        aft_sum,
                        abs_x,
                        abs_x_rank,
                        dim_n,
                        G_fenzi_element,
                        G_fenmu,
                        G_fenzi,
                        G,
                        G_rank,
                        ep,
                        -- 12. 维度内对ep降序排序
                        row_number() over (partition by dim order by abs(ep) desc) as ep_rank
                  from (
                        select
                              dim,
                              element,
                              before,
                              after,
                              pre_sum,
                              aft_sum,
                              abs_x,
                              abs_x_rank,
                              dim_n,
                              G_fenzi_element,
                              G_fenmu,
                              G_fenzi,
                              -- 9. 计算维度级别的基尼系数
                              G_fenzi / G_fenmu as G,
                              -- 10. 按照计算好的各维度的基尼系数从大到小对dim进行降序排序
                              row_number() over(order by G_fenzi / G_fenmu desc) as G_rank,
                              ep
                        from (
                              select
                                    t4.dim,
                                    t4.element,
                                    t4.before,
                                    t4.after,
                                    t4.pre_sum,
                                    t4.aft_sum,
                                    t4.abs_x,
                                    t4.abs_x_rank,
                                    t4.dim_n,
                                    t4.G_fenzi_element,
                                    t4.G_fenmu,
                                    --8. 计算基尼系数的分子
                                    sum(t4.G_fenzi_element) over(partition by t4.dim_n) as G_fenzi,
                                    t4.ep
                              from (
                                    select
                                          t3.dim,
                                          t3.element,
                                          t3.before,
                                          t3.after,
                                          t3.pre_sum,
                                          t3.aft_sum,
                                          t3.abs_x,
                                          t3.abs_x_rank,
                                          t3.dim_n,
                                          -- 5. 计算每项系数和绝对值相乘 (2i-n-1)xi
                                          (2 * t3.abs_x_rank - t3.dim_n - 1) * t3.abs_x as G_fenzi_element,
                                          -- 6. 计算基尼系数的分母
                                          t3.dim_n * (sum(t3.abs_x) over (partition by t3.dim))  as G_fenmu,
                                          -- 7. 计算贡献率EP:即每个元素波动对于总体波动的贡献，以A渠道为例，A渠道的EP=（A渠道活动后销售额-A渠道活动前销售额）/（总体活动后销售额-总体活动前销售额）。
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
                                                t2.dim,
                                                t2.element,
                                                t2.before,
                                                t2.after,
                                                t2.pre_sum,
                                                t2.aft_sum,
                                                t2.abs_x,
                                                -- 3.维度内对abs_x进行升序
                                                row_number() over (partition by t2.dim order by t2.abs_x asc) as abs_x_rank,
                                                -- 4.统计维度内元素个数
                                                count(*) over (partition by t2.dim ) as dim_n
                                          from (
                                                select
                                                      t1.dim,
                                                      t1.element,
                                                      t1.before,
                                                      t1.after,
                                                      t1.pre_sum,
                                                      t1.aft_sum,
                                                      -- 2. 先计算每个element的绝对值变化量
                                                      abs(t1.after - t1.before) as abs_x
                                                from  base_info t1
                                          ) t2
                                    ) t3
                              ) t4
                        ) t5
                  ) t6
                  -- 11. 获取基尼系数最大的top5维度 且 根据单个元素EP阈值，过滤出大于0.2的元素
                  where G_rank <= 5 and abs(t6.ep) >= 0.2
            ) t7
      ) t8
) t9
-- 15. 整体EP(单维度下)（波动贡献率）的筛选：意味着只要选中元素贡献率之和超过60%，就已经能够解释大部分波动原因了
-- 在根据总EP阈值批量筛选时:包含第一个大于总EP阈值的元素:lag_ep_sum是为了处理这种情况的
-- 0.5  0.5
-- 0.9  0.5
-- 2025年8月2日21:36:34 :临时加了一个 (t6.ep_sum > 0.8 and t6.lag_ep_sum = 1)条件
-- 2025年8月3日08:14:20 :临时增加了一个t6.ep_sum >= 1.0
-- before	after	pre_sum	aft_sum    ep
-- 86	      72	    88	    75         1.076923076923  发现贡献度有大于1的情况
where t9.ep_sum <= 0.8 or (t9.ep_sum > 0.8 and t9.lag_ep_sum < 0.8) or (t9.ep_sum > 0.8 and t9.lag_ep_sum = 1) or t9.ep_sum >= 1.0



