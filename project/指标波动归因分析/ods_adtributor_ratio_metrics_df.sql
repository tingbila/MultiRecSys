with base_info as (
	 select
	 	   if(dim like '%分子%','分子','分母') as dim,
	 	   element,
	 	   cast(before as bigint)  as before,
	 	   cast(after  as bigint)  as after,
	 	   'on' as join_column
	 from  starx_ods.ods_adtributor_add_metrics_df
	 where dt = '${dt}'
)




SELECT
      -- 维度：Aij代表after、 Fij代表before
      element,
      -- 分子
      m1_before,
      m1_after,
      m1_pre_sum,
      m1_aft_sum,
      m1_p,
      m1_q,
      m1_surprise,
      m1_ep,
      -- 分母
      m2_before,
      m2_after,
      m2_pre_sum,
      m2_aft_sum,
      m2_p,
      m2_q,
      m2_surprise,
      m2_ep,
      -- 9. Adtributor算法中用于计算率值指标（如点击率、转化率等）贡献度（EP, Expected Point） 的核心公式之一。相比可加性指标的 EP 简单差值
      -- 除以总量，这个公式专门用于处理比率类指标的归因问题，它考虑了分子和分母两个部分的变化对整体比值的影响。
      -- 绝对变化值贡献EP
      ROUND(((m1_after - m1_before) * m2_pre_sum - (m2_after - m2_before) * m1_pre_sum) / (m2_pre_sum * (m2_pre_sum + m2_after - m2_before)), 12) as m1_m2_ep,
      -- 计算惊讶度:分子的S+分母的S:整体结构变化 = 分子结构变动程度 + 分母结构变动程度
      ROUND(COALESCE(m1_surprise, 0) + COALESCE(m2_surprise, 0), 12) AS m1_m2_surprise
from (
      SELECT
            -- 维度：Aij代表after、 Fij代表before
            element,
            -- 分子
            MAX(IF(dim = '分子', before, null))                      AS m1_before,
            MAX(IF(dim = '分子', after, null))                       AS m1_after,
            MAX(IF(dim = '分子', pre_sum, null))                     AS m1_pre_sum,
            MAX(IF(dim = '分子', aft_sum, null))                     AS m1_aft_sum,
            MAX(IF(dim = '分子', p, null))                           AS m1_p,
            MAX(IF(dim = '分子', q, null))                           AS m1_q,
            MAX(IF(dim = '分子', surprise, null))                    AS m1_surprise,
            MAX(IF(dim = '分子', ep, null))                          AS m1_ep,
            -- 分母
            MAX(IF(dim = '分母', before, null))                      AS m2_before,
            MAX(IF(dim = '分母', after, null))                       AS m2_after,
            MAX(IF(dim = '分母', pre_sum, null))                     AS m2_pre_sum,
            MAX(IF(dim = '分母', aft_sum, null))                     AS m2_aft_sum,
            MAX(IF(dim = '分母', p, null))                           AS m2_p,
            MAX(IF(dim = '分母', q, null))                           AS m2_q,
            MAX(IF(dim = '分母', surprise, null))                    AS m2_surprise,
            MAX(IF(dim = '分母', ep, null))                          AS m2_ep
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
                  ROUND(0.5 * (p * LN(2 * p / (p + q)) / LN(10) + q * LN(2 * q / (p + q)) / LN(10)), 12)  as surprise,
                  -- 4. 计算贡献率EP:即每个元素波动对于总体波动的贡献，以A渠道为例，A渠道的EP=（A渠道活动后销售额-A渠道活动前销售额）/（总体活动后销售额-总体活动前销售额）。
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
                              dim,
                              sum(before) as pre_sum,
                              sum(after)  as aft_sum,
                              'on' as join_column
                        from  base_info
                        group by dim
                  ) t2
                  on t1.dim = t2.dim
            ) t3
      ) t4
      group by element
) t5





