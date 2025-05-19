import pandas as pd
import random

def add_random_history_columns_v2(input_file, output_file, max_history_length=5,
                                   item_id_range=(1, 50), city_id_range=(1, 50)):
    """
    向原始数据中添加随机生成的历史 item_id 和 item_city 序列列，格式为逗号分隔的字符串。

    :param input_file: 输入文件路径（tsv 格式）
    :param output_file: 输出文件路径
    :param max_history_length: 每条样本历史序列最大长度
    :param item_id_range: tuple，历史 item_id 的生成范围（默认 1~49）
    :param city_id_range: tuple，历史 item_city 的生成范围（默认 1~49）
    """
    # 原始列名
    column_names = ["uid", "user_city", "item_id", "author_id", "item_city", "channel",
                    "finish", "like", "music_id", "device", "time", "duration_time",
                    "genres", "actors"]

    data = pd.read_csv(input_file, sep='\t', names=column_names, header=None)

    history_item_ids = []
    history_citys = []

    for _ in range(len(data)):
        history_len = random.randint(1, max_history_length)

        item_seq = [str(random.randint(item_id_range[0], item_id_range[1] - 1)) for _ in range(history_len)]
        city_seq = [str(random.randint(city_id_range[0], city_id_range[1] - 1)) for _ in range(history_len)]

        history_item_ids.append(','.join(item_seq))
        history_citys.append(','.join(city_seq))

    data['history_item_ids'] = history_item_ids
    data['history_citys'] = history_citys

    data.to_csv(output_file, sep='\t', index=False, header=False, quoting=3)
    print(f"✅ 随机历史列已添加并保存到: {output_file}")


if __name__ == "__main__":
    add_random_history_columns_v2("train_2.csv", "train_2_with_history.csv", max_history_length=5)
