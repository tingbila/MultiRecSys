import pandas as pd
import random

def add_random_history_columns(input_file, output_file, max_history_length=5):
    # 原始列名
    column_names = ["uid", "user_city", "item_id", "author_id", "item_city", "channel",
                    "finish", "like", "music_id", "device", "time", "duration_time",
                    "genres", "actors"]

    # 读取 tab 分隔的原始文件
    data = pd.read_csv(input_file, sep='\t', names=column_names, header=None)

    # 筛选出 item_id 和 item_city 中有效的（非负数）的值
    all_item_ids = data['item_id'].dropna()
    all_item_ids = all_item_ids[all_item_ids >= 0].astype(int).astype(str).tolist()

    all_item_citys = data['item_city'].dropna()
    all_item_citys = all_item_citys[all_item_citys >= 0].astype(int).astype(str).tolist()

    history_item_ids = []
    history_citys = []

    for _ in range(len(data)):
        history_len = random.randint(1, max_history_length)

        item_seq = random.sample(all_item_ids, k=min(history_len, len(all_item_ids)))
        city_seq = random.sample(all_item_citys, k=min(history_len, len(all_item_citys)))

        history_item_ids.append(','.join(item_seq))
        history_citys.append(','.join(city_seq))

    # 添加新列
    data['history_item_ids'] = history_item_ids
    data['history_citys'] = history_citys

    # 写入文件，保持 \t 分隔，不带 header
    data.to_csv(output_file, sep='\t', index=False, header=False, quoting=3)
    print(f"✅ 历史列（无负数）已添加并保存到: {output_file}")



if __name__ == "__main__":
    add_random_history_columns("train_2.csv", "train_2_with_history.csv", 5)
