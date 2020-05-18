import pandas as pd
import numpy as np
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             roc_curve, auc, confusion_matrix, mean_squared_error,
                             classification_report)
class preprocess:
    lbl = ["Duration", "Protocol_type", "Service", "Flag", "Src_bytes", "Dst_bytes",
           "Land", "Wrong_fragment", "Urgent", "Hot", "Num_failed_logins", "Logged_in",
           "Num_compromised", "Root_shell", "Su_attempted", "Num_root", "Num_file_creations",
           "Num_shells", "Num_access_files", "Num_outbound_cmds", "Is_hot_login",
           "Is_guest_login", "Count", "Srv_count", "Serror_rate", "Srv_serror_rate",
           "Rerror_rate", "Srv_rerror_rate", "Same_srv_rate", "Diff_srv_rate",
           "Srv_diff_host_rate", "Dst_host_count", "Dst_host_srv_count", "Dst_host_same_srv_rate",
           "Dst_host_diff_srv_rate", "Dst_host_same_src_port_rate", "Dst_host_srv_diff_host_rate",
           "Dst_host_serror_rate", "Dst_host_srv_serror_rate", "Dst_host_rerror_rate",
           "Dst_host_srv_rerror_rate", "attack_type", "Class"]

    dos_attacks = ["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2",
                   "udpstorm", "processtable", "worm"]
    probe_attacks = ["Satan", "Ipsweep", "Nmap", "Portsweep", "Mscan", "Saint"]
    r2l_attacks = ["guess_Password", "ftp_write", "imap", "phf", "multihop", "warezmaster",
                   "warezclient", "spy", "xlock", "xsnoop", "snmpguess", "snmpgetattack",
                   "httptunnel", "sendmail", "named"]
    u2l_attacks = ["buffer_overflow", "loadmodule", "rootkit", "perl", "sqlattack",
                   "xterm", "ps"]

    def __init__(self):
        pass

    def convertstringtonumber(self, df: pd.DataFrame, lst):
        """ 字符串转为数字型"""
        for n in range(len(lst)):
            df = df.replace(lst[n], n)
        return df

    def scalex(self, X):
        """ 数值标准化"""
        nmin, nmax = 0.0, 1.0
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (nmax - nmin) + nmin
        return X_scaled

    def calcrmse(self, X_train: pd.DataFrame, gensamples: pd.DataFrame):
        """计算均方误差"""
        max_column = X_train.shape[1]
        rmse_lst = []
        for col in range(max_column):
            rmse_lst.append(np.sqrt(mean_squared_error(X_train[:, col], gensamples[:, col])))
        return np.sum(rmse_lst) / max_column

    def create_df(self, df_train, df_test):
        """为GAN提供预处理后的输入数据"""
        df_train.columns = self.lbl
        df_test.columns = self.lbl
        df_train = df_train.drop(columns=['Class'])
        df_test = df_test.drop(columns=['Class'])
        # convert strings to numbers
        protocol_type = ['icmp', 'tcp', 'udp']
        service = ['IRC', 'X11', 'Z39_50', 'aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf',
                   'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs',
                   'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http',
                   'http_2784', 'http_443', 'http_8001', 'imap4', 'iso_tsap', 'klogin', 'kshell',
                   'ldap', 'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns',
                   'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2',
                   'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell', 'smtp',
                   'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i',
                   'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois']
        flag = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']

        df_train = self.convertstringtonumber(df_train, protocol_type)
        df_train = self.convertstringtonumber(df_train, service)
        df_train = self.convertstringtonumber(df_train, flag)
        df_test = self.convertstringtonumber(df_test, protocol_type)
        df_test = self.convertstringtonumber(df_test, service)
        df_test = self.convertstringtonumber(df_test, flag)

        for n in range(len(self.lbl) - 2):  # df_train标准化
            m = self.lbl[n]
            if (np.max(df_train[m]) > 1):
                if (len(np.unique(df_train[m])) > 1):
                    df_train[m] = self.scalex(df_train[m])
                else:
                    df_train[m] = np.int64(1)

        for n in range(len(self.lbl) - 2):  # df_test标准化
            m = self.lbl[n]
            if (np.max(df_test[m]) > 1):
                if (len(np.unique(df_test[m])) > 1):
                    df_test[m] = self.scalex(df_test[m])
                else:
                    df_test[m] = np.int64(1)

        labeldf_train = df_train['attack_type']
        labeldf_test = df_test['attack_type']
        newlabeldf_train = labeldf_train.replace(
            {'normal': 0, 'ftp_write': 1, 'guess_passwd': 1, 'imap': 1, 'multihop': 1, 'phf': 1, 'spy': 1,'warezclient': 1, 'warezmaster': 1, 'sendmail': 1, 'named': 1, 'snmpgetattack': 1, 'snmpguess': 1, 'xlock': 1, 'xsnoop': 1, 'httptunnel': 1,
             'ipsweep': 2, 'nmap': 2, 'portsweep': 2, 'satan': 2, 'mscan': 2, 'saint': 2,
             'neptune': 3, 'back': 3, 'land': 3, 'pod': 3, 'smurf': 3, 'teardrop': 3, 'mailbomb': 3, 'apache2': 3,'processtable': 3, 'udpstorm': 3, 'worm': 3,
             'buffer_overflow': 4, 'loadmodule': 4, 'perl': 4, 'rootkit': 4, 'ps': 4, 'sqlattack': 4, 'xterm': 4})
        newlabeldf_test = labeldf_test.replace(
            {'normal': 0, 'ftp_write': 1, 'guess_passwd': 1, 'imap': 1, 'multihop': 1, 'phf': 1, 'spy': 1,
             'warezclient': 1, 'warezmaster': 1, 'sendmail': 1, 'named': 1, 'snmpgetattack': 1, 'snmpguess': 1,
             'xlock': 1, 'xsnoop': 1, 'httptunnel': 1,
             'ipsweep': 2, 'nmap': 2, 'portsweep': 2, 'satan': 2, 'mscan': 2, 'saint': 2,
             'neptune': 3, 'back': 3, 'land': 3, 'pod': 3, 'smurf': 3, 'teardrop': 3, 'mailbomb': 3, 'apache2': 3,
             'processtable': 3, 'udpstorm': 3, 'worm': 3,
             'buffer_overflow': 4, 'loadmodule': 4, 'perl': 4, 'rootkit': 4, 'ps': 4, 'sqlattack': 4, 'xterm': 4})
        df_train['attack_type'] = newlabeldf_train
        df_test['attack_type'] = newlabeldf_test

        to_drop_normal =[1, 2, 3, 4]
        to_drop_R2L = [0, 2, 3, 4]
        to_drop_R2L_train_test =[2, 3, 4]
        normal_df = df_train[~df_train['attack_type'].isin(to_drop_normal)] # normal_df中只包含训练集中的normal，为的是在使用SMOTE时更方便
        R2L_df = df_train[~df_train['attack_type'].isin(to_drop_R2L)] # R2L_df是GAN的输入，只包含R2L攻击
        R2L_df_train = df_train[~df_train['attack_type'].isin(to_drop_R2L_train_test)] # R2L_df_train中包含训练集中的R2L攻击和normal
        R2L_df_test = df_test[~df_test['attack_type'].isin(to_drop_R2L_train_test)] # R2L_df_test中包含测试集中的R2L攻击和normal

        return normal_df, R2L_df, R2L_df_train, R2L_df_test

    def gererated_preprocess(self, generated_R2L: pd.DataFrame):
        """为GAN生成的数据加上attack_type"""
        df = generated_R2L
        df.columns = self.lbl[:-2]
        df["attack_type"] = pd.Series(["1"]*len(df), index=df.index)

        return df

    def split_df(self, df):
        #将df拆分为X和y，为机器学习做准备数据
        X_df = df.drop('attack_type', 1)
        y_df = df.attack_type

        return X_df, y_df

    def merge_df(self,df1, df2):
        df = df1.append(df2)

        return  df
