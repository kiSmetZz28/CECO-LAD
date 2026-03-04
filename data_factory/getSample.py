import pandas as pd
import csv

def getDataSample(file_path, save_path, fraction=0.1):

    with open(file_path, 'r') as file:
         lines = [line.strip() for line in file.readlines()]

    # Convert the list of lines into a DataFrame
    df = pd.DataFrame(lines)

    # Sample fraction of the lines
    sampled_df = df.sample(frac=fraction, random_state=42)  # random_state ensures reproducibility

    sampled_df.to_csv(save_path, index=False, header=False, quoting=csv.QUOTE_NONE, sep="\n")

    print(f"Sampled {len(sampled_df)} lines out of {len(df)}.")


if __name__ == '__main__':
     # getDataSample('./dataset/OpenStack/test_normal.txt', './dataset/OpenStack/sample/test_normal_sample_001.txt', 0.01)
     # getDataSample('./dataset/OpenStack/test_abnormal.txt', './dataset/OpenStack/sample/test_abnormal_sample_001.txt', 0.01)

     # getDataSample('./dataset/OpenStack/test_normal.txt', './dataset/OpenStack/sample/test_normal_sample_002.txt', 0.02)
     # getDataSample('./dataset/OpenStack/test_abnormal.txt', './dataset/OpenStack/sample/test_abnormal_sample_002.txt', 0.02)

     # getDataSample('./dataset/OpenStack/test_normal.txt', './dataset/OpenStack/sample/test_normal_sample_005.txt', 0.05)
     # getDataSample('./dataset/OpenStack/test_abnormal.txt', './dataset/OpenStack/sample/test_abnormal_sample_005.txt', 0.05)
     
     getDataSample('./dataset/HDFS/hdfs_test_normal.txt', './dataset/HDFS/sample/hdfs_test_normal_sample_0001.txt', 0.001)
     getDataSample('./dataset/HDFS/hdfs_test_abnormal.txt', './dataset/HDFS/sample/hdfs_test_abnormal_sample_0001.txt', 0.001)

     # getDataSample('./dataset/HDFS/hdfs_test_normal.txt', './dataset/HDFS/sample/hdfs_test_normal_sample_002.txt', 0.02)
     # getDataSample('./dataset/HDFS/hdfs_test_abnormal.txt', './dataset/HDFS/sample/hdfs_test_abnormal_sample_002.txt', 0.02)

     # getDataSample('./dataset/HDFS/hdfs_test_normal.txt', './dataset/HDFS/sample/hdfs_test_normal_sample_005.txt', 0.05)
     # getDataSample('./dataset/HDFS/hdfs_test_abnormal.txt', './dataset/HDFS/sample/hdfs_test_abnormal_sample_005.txt', 0.05)