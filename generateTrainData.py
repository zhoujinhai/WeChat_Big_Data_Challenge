import pandas as pd
import os

# 存储数据的根目录
ROOT_PATH = "./data"
# 比赛数据集路径
DATASET_PATH = os.path.join(ROOT_PATH, "wechat_algo_data1")

# 数据集
FEED_EMBEDDINGS_FILE = os.path.join(DATASET_PATH, "feed_embeddings.csv")
USER_ACTION = os.path.join(DATASET_PATH, "user_action.csv")
TEST_FILE = os.path.join(DATASET_PATH, "test_a.csv")

STAGE_END_DAY = {"online_train": 14, "offline_train": 12, "evaluate": 13, "submit": 15}

# 组合online_train所有数据，合成一个df
ONLINE_PATH = os.path.join(ROOT_PATH, "online_train")

# 组合offline_train所有数据，合成一个df
OFFLINE_PATH = os.path.join(ROOT_PATH, "offline_train")

# 最终数据保存路径
DATA_SETS_PATH = os.path.join(ROOT_PATH, "data_sets")

# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar",  "forward"]
# 复赛待预测行为列表
# ACTION_LIST = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]


def merge_action():
    online_file_list = []
    offline_file_list = []
    for action in ACTION_LIST:
        stage = "online_train"
        online_file_list.append(os.path.join(ONLINE_PATH, "online_train_" + action + "_" + str(STAGE_END_DAY[stage]) + "_concate_sample.csv"))
    for action in ACTION_LIST:
        stage = "offline_train"
        offline_file_list.append(os.path.join(OFFLINE_PATH, "offline_train_" + action + "_" + str(STAGE_END_DAY[stage]) + "_concate_sample.csv"))

    online_df = pd.read_csv(online_file_list[0])
    for i in range(1, len(online_file_list)):
        action_df = pd.read_csv(online_file_list[i])
        online_df = online_df.append(action_df)

    online_df = online_df.fillna(0)
    print(online_df.columns)
    print(online_df.shape)
    online_df = online_df.drop_duplicates(subset=['userid', 'feedid'], keep='last')
    print(online_df.shape)

    file_name = os.path.join(ONLINE_PATH, "online_train" + "_all_" + str(STAGE_END_DAY["online_train"]) + "_concate_sample.csv")
    print('Save to: %s' % file_name)
    online_df.to_csv(file_name, index=False)

    offline_df = pd.read_csv(offline_file_list[0])
    for i in range(1, len(offline_file_list)):
        action_df = pd.read_csv(offline_file_list[i])
        offline_df = offline_df.append(action_df)

    offline_df = offline_df.fillna(0)
    print(offline_df.columns)
    print(offline_df.shape)
    offline_df = offline_df.drop_duplicates(subset=['userid', 'feedid'], keep='last')
    print(offline_df.shape)

    file_name = os.path.join(OFFLINE_PATH, "offline_train" + "_all_" + str(STAGE_END_DAY["offline_train"]) + "_concate_sample.csv")
    print('Save to: %s' % file_name)
    offline_df.to_csv(file_name, index=False)


if __name__ == "__main__":
    # merge_action()

    print("start read feed embedding file: ")
    feed_embed_feature = pd.read_csv(FEED_EMBEDDINGS_FILE)
    embedding = feed_embed_feature["feed_embedding"].str.split(' ', expand=True)
    n_cols = embedding.shape[1] - 1
    for i in range(n_cols):
        col_name = "feed_emb_" + str(i + 1)
        feed_embed_feature[col_name] = pd.to_numeric(embedding[i])
    del feed_embed_feature['feed_embedding']
    feed_embed_feature = feed_embed_feature.set_index(["feedid"])
    print(feed_embed_feature.shape)
    print(feed_embed_feature.head())
    # print(feed_embed_feature.columns)
    # feed_embed_feature["feed_embedding"] = embedding
    # print(feed_embed_feature.shape)
    # print(feed_embed_feature.head())
    # print(feed_embed_feature.columns)

    stages = ["offline_train", "online_train", "evaluate", "submit"]
    for stage in stages:
        sample_path = os.path.join(ROOT_PATH, stage, stage + "_all_" + str(STAGE_END_DAY[stage]) + "_concate_sample.csv")
        save_path = os.path.join(DATA_SETS_PATH, stage + ".csv")

        stage_df = pd.read_csv(sample_path)
        print(stage_df.shape)

        stage_df = stage_df.join(feed_embed_feature, on="feedid", how="left").fillna(0.0)

        print(stage_df.shape)
        print(stage_df.head())

        stage_df.to_csv(save_path)
        print("saved to %s" % save_path)