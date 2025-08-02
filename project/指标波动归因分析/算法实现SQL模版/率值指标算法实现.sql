-- Adtributor算法用于率值指标贡献度和惊讶度计算
-- Adtributor算法的率值指标下的计算逻辑与可加性指标（如曝光量、点击数等）不同，对于率值类指标（如点击率、转化率等），Adtributor 算法在计算惊讶度（Surprise）和贡献度（EP）时需要同时考虑分子和分母的变化，这是它的关键差异点。
-- 但除此之外，整体计算流程与可加性指标保持一致：包括元素粒度的结构差异分析、排序、累计 EP 到阈值、维度影响力排序等步骤。这使得算法具备统一性和可迁移性，同时兼顾了不同类型指标的解释能力。


with base_info as (
	 select
	 	   dim,      -- 用户分类_分母  {维度}_{分子分母}拼接方式
	 	   element,
	 	   before,
	 	   after,
	 	   -- 1. 先计算活动前和活动后销售额的总体数据--对于每个维度pre_sum和aft_sum大部分情况应该是一样的，但是有的时候可能不相同。
	 	   pre_sum,
	 	   aft_sum
	 from  starx_ads.ads_sm_ug_new_device_retention_ratio_adtributor_di
	 where dt = '${dt}'
),
m1_and_m2 as (
     SELECT
           -- 维度：Aij代表after、 Fij代表before
           dim,
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
           m1_m2_ep,                        -- 分子分母的联合贡献度（可能为负数）
           m1_m2_surprise as surprise,      -- 分子分母的联合惊讶度
           -- 6. 分子分母的联合贡献度进行归一化处理（可能为负数）
           ROUND(m1_m2_ep / ROUND(sum(m1_m2_ep) over (partition by dim),12),12) as ep  -- 分子分母的联合贡献度（归一化的结果）
     from (
           SELECT
                 -- 维度：Aij代表after、 Fij代表before
                 dim,
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
                 -- 5. Adtributor算法中用于计算率值指标（如点击率、转化率等）贡献度（EP, Expected Point） 的核心公式之一。相比可加性指标的 EP 简单差值
                 -- 除以总量，这个公式专门用于处理比率类指标的归因问题，它考虑了分子和分母两个部分的变化对整体比值的影响。
                 -- 计算贡献率:分子分母的ep
                 ROUND(((m1_after - m1_before) * m2_pre_sum - (m2_after - m2_before) * m1_pre_sum) / (m2_pre_sum * (m2_pre_sum + m2_after - m2_before)), 12) as m1_m2_ep,
                 -- 计算惊讶度:分子的S+分母的S 整体结构变化 = 分子结构变动程度 + 分母结构变动程度
                 ROUND(COALESCE(m1_surprise, 0) + COALESCE(m2_surprise, 0), 12) AS m1_m2_surprise
           from (
                 SELECT
                       -- 维度：Aij代表after、 Fij代表before   -- 用户分类、ele
                       regexp_extract(dim, '^(.*)_[^_]+$', 1) as dim,
                       element,
                       -- 分子
                       MAX(IF(dim rlike '分子', before, null))                      AS m1_before,
                       MAX(IF(dim rlike '分子', after, null))                       AS m1_after,
                       MAX(IF(dim rlike '分子', pre_sum, null))                     AS m1_pre_sum,
                       MAX(IF(dim rlike '分子', aft_sum, null))                     AS m1_aft_sum,
                       MAX(IF(dim rlike '分子', p, null))                           AS m1_p,
                       MAX(IF(dim rlike '分子', q, null))                           AS m1_q,
                       MAX(IF(dim rlike '分子', surprise, null))                    AS m1_surprise,
                       MAX(IF(dim rlike '分子', ep, null))                          AS m1_ep,
                       -- 分母
                       MAX(IF(dim rlike '分母', before, null))                      AS m2_before,
                       MAX(IF(dim rlike '分母', after, null))                       AS m2_after,
                       MAX(IF(dim rlike '分母', pre_sum, null))                     AS m2_pre_sum,
                       MAX(IF(dim rlike '分母', aft_sum, null))                     AS m2_aft_sum,
                       MAX(IF(dim rlike '分母', p, null))                           AS m2_p,
                       MAX(IF(dim rlike '分母', q, null))                           AS m2_q,
                       MAX(IF(dim rlike '分母', surprise, null))                    AS m2_surprise,
                       MAX(IF(dim rlike '分母', ep, null))                          AS m2_ep
                 from (
                       select
                             t2.dim,
                             t2.element,
                             t2.before,
                             t2.after,
                             t2.pre_sum,
                             t2.aft_sum,
                             t2.p,
                             t2.q,
                             -- JS散度公式s = 0.5 * (p * math.log10(2 * p / (p + q)) + q * math.log10(2 * q / (p + q)))
                             -- 3. 惊讶度（Surprise，用S表示）是一个用来衡量指标结构前后变化程度的指标，回答的是"哪个元素的波动最让人惊讶"的问题。
                             -- JS散度要求概率非负且0~1之间，加绝对值避免负值导致log计算出错
                             coalesce(ROUND(0.5 * (p * LN(2 * p / (p + q)) / LN(10) + q * LN(2 * q / (p + q)) / LN(10)), 12),0)  as surprise,
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
                                   t1.pre_sum,
                                   t1.aft_sum,
                                   -- 2. 计算活动前销售额占比p和活动后销售额占比q
                                   -- 加绝对值可以避免负值导致后续JS散度计算出错:p 和 q 表示概率或占比，理论上是非负且小于等于1的数值。它们是活动前后某元素销售额占总销售额的比例，不应出现负值。
                                   ROUND(ABS(t1.before) / ABS(t1.pre_sum), 12) AS p,
                                   ROUND(ABS(t1.after)  / ABS(t1.aft_sum), 12) AS q
                             from  base_info t1
                       ) t2
                 ) t3
                 group by regexp_extract(dim, '^(.*)_[^_]+$', 1),element   -- 用户分类、ele
           ) t4
     ) t5
)




select
      -- 维度：Aij代表after、 Fij代表before
      dim,
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
      -- ep&s
      surprise,
      surprise_rank,
      ep,
      ep_sum,
      lag_ep_sum,
      surprise_sum,
      overall_dim_surprise_rank
from (
      select
            -- 维度：Aij代表after、 Fij代表before
            dim,
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
            -- ep&s
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
                  -- 维度：Aij代表after、 Fij代表before
                  dim,
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
                  -- ep&s
                  surprise,
                  surprise_rank,
                  ep,
                  ep_sum,
                  lag_ep_sum,
                  -- 5. 在每个维度下汇总各元素的S值，得到各维度S值的汇总结果。
                  sum(surprise) over (partition by dim) as surprise_sum
            from (
                  select
                        -- 维度：Aij代表after、 Fij代表before
                        dim,
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
                        -- ep&s
                        surprise,
                        surprise_rank,
                        ep,
                        ep_sum,
                        -- 取当前行的上一行（1）的 ep_sum 值；如果没有上一行（例如是第一行），就使用默认值 ep_sum（即当前行的值）
                        lag(ep_sum,1,ep_sum) over (partition by dim order by surprise_rank asc) as lag_ep_sum
                  from (
                        select
                              -- 维度：Aij代表after、 Fij代表before
                              dim,
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
                              -- ep&s
                              surprise,
                              surprise_rank,
                              ep,
                              -- 3. 筛选完单个元素EP值之后，在对每个维度下通过筛选的元素EP值进行累加
                              -- 这里额外也添加了一个绝对值
                              -- 这里用绝对值累加，是想统计所有元素贡献度的大小和，忽略正负方向:这样设计是对的，因为你想选出贡献总量达到阈值的元素集。
                              sum(abs(ep)) over (partition by dim order by surprise_rank asc rows between unbounded preceding and current row ) as ep_sum
                        from (
                              select
                                   -- 维度：Aij代表after、 Fij代表before
                                   dim,
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
                                   -- ep&s
                                   surprise,
                                   ep,
                                   -- 1. 在每个维度内将元素按照惊讶度S从高到低对数据进行排序
                                   row_number() over (partition by dim order by surprise desc) as surprise_rank
                              from m1_and_m2
                        ) t1
                        -- 2. 根据设定的单个元素EP阈值，遍历所有元素的EP值是否高于0.2，如果高于，则通过筛选
                        where abs(t1.ep) >= 0.2   -- 这里加了一个绝对值
                  ) t2
            ) t3
            -- 4. 整体EP(单维度下)（波动贡献率）的筛选：意味着只要选中元素贡献率之和超过60%，就已经能够解释大部分波动原因了
            -- 在根据总EP阈值批量筛选时:包含第一个大于总EP阈值的元素:lag_ep_sum是为了处理这种情况的
            -- 0.5  0.5
            -- 0.9  0.5
            where t3.ep_sum <= 0.8 or (t3.ep_sum > 0.8 and t3.lag_ep_sum < 0.8)
      ) t4
) t5
-- 6. 假设我们最终的目标是筛选影响最大的top2的维度进行原因定位
where t5.overall_dim_surprise_rank <= 2








