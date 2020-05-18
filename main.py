import argparse
import pandas as pd
import numpy as np
from generator import GAN
from preprocess import preprocess
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_recall_curve, average_precision_score, roc_curve, auc, confusion_matrix, mean_squared_error,classification_report


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("gradient_penalty_weight", help="梯度惩罚权重", type=float, default=0.01)
    parser.add_argument("epoch", help="训练轮次", type=int, default=1000)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    trainset = pd.read_csv('./data/KDDTrain+.txt', sep=",", header=None)
    testset = pd.read_csv('./data/KDDTest+.txt', sep=",", header=None)

    #下面部分是GAN训练并且生成数据
    processor = preprocess()
    normal_df, R2L_df, R2L_df_train, R2L_df_test = processor.create_df(df_train=trainset, df_test=testset)
    GAN_train = np.asarray( R2L_df.iloc[:GAN.MAX_SIM, :-1])
    print("GAN训练....")
    gan = GAN(train_df=GAN_train, GRADIENT_PENALTY_WEIGHT=args.gradient_penalty_weight, MAX_EPOCH=args.epoch)
    gan.compile()
    print("已生成数据...")

    #以下是将GAN生成的数据加入训练集中，得到最终检测结果
    generated_R2L = pd.read_csv('./generated_samples/GPWGAN_generated.csv', sep=",", header=None)
    R2L_df_generated = processor.gererated_preprocess(generated_R2L)
    train_set_addGAN_df = processor.merge_df(R2L_df_train, R2L_df_generated)
    X_train,y_train = processor.split_df(train_set_addGAN_df)
    X_test,y_test = processor.split_df(R2L_df_test)

    # print(X_test.head(5))
    # all features
    print("开始训练决策树...")
    # print(X_train["Flag"])
    # print(R2L_df.shape[0], R2L_df_train.shape[0], R2L_df_test.shape[0], X_train.shape[0])
    clf_R2L = DecisionTreeClassifier(random_state=0)
    clf_R2L.fit(X_train, y_train.astype('int'))
    """训练DT;在sklearn 模型训练是出现如下报错：‘ValueError: Unknown label type: ‘unknown’’该怎么解决？
    以GBDT为例：train_y后加上astype(‘int’)即可"""
    y_pred = clf_R2L.predict(X_test)  # 预测结果
    # Create confusion matrix
    y_ture = y_test
    print("实验结果...")
    print("accuracy:",accuracy_score(y_ture, y_pred))
if __name__ == "__main__":
    main()
