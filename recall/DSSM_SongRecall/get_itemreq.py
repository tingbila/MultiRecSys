# coding = utf-8
import csv

if __name__ == "__main__":
    items_dict = dict()
    i = 0
    with open("./data/dssm_data_u2i.txt", 'r') as f:
        for line in f.readlines():
            if i % 100000 == 0:
                print(i)
            i += 1
            try:
                tmp = line.strip().split("\t")[1]
                items = tmp.split(",")
                # print(items)
                for item in items:
                    if item not in items_dict.keys():
                        items_dict.setdefault(item, 0)
                    items_dict[item] += 1
            except:
                continue
    # print(items_dict)
    print("read file end")

    with open("./data/item_freq.csv", "w") as w_f:
        writer = csv.writer(w_f)
        for k, v in items_dict.items():
            writer.writerow([k, v])
