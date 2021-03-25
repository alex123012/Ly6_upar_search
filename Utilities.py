import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Добавление митионина
def add_M(x):
    return x if x.startswith('M') else 'M' + x


# Нахождение последовательности с неизвестными аминокислотами
def find_X(x):
    return None if 'xxxxxxx' in x.lower() else x


# Функция для раздедения датасета на две группы без ортологов в разных выборках
def split_f(x, df2):
    if x[:4] in df2:
        return 1
    else:
        return 0


# Для NB показывает вероятность данной фичи встретится в данном классе
# Для SVC - координата вектора разделения
def show_most_informative_features(pipeline, n=20, x=0):
    feature_names = pipeline['tokenizer'].get_feature_names()
    coefs_with_fns = sorted(zip(pipeline['model'].coef_[x], feature_names))
    top = coefs_with_fns[-n:] if n > 0 else coefs_with_fns[:-n]
    return top


# Функция выделения LU-домена значков "#"
def find_LU(x, n):
    res = x.replace('\n', '') + '\n'
    z = 0
    for i, j in enumerate(res):
        if i < z:
            continue
        if j == 'C':
            tmp = res[i:i + 100]
            if tmp.count('C') >= 8:
                res += '_' * (i - z - 3) + '#' * 103
                z = i + 100
    return n + '\n' + res + '\n' * 4


# Конвертация fasta в DataFrame
def fasta_to_frame(fasta):
    with open(fasta, 'r') as f:
        seqs = []
        ids = []
        for record in SeqIO.parse(fasta, "fasta"):
            seq = ''.join(record.seq)
            if 'xxxxxxx' in seq.lower():
                continue
            seqs.append(seq)
            ids.append(record.id)
            df = pd.DataFrame({'genes': ids, 'sequence': seqs})
            df.set_index('genes', inplace=True)
    return df


# Функция для создания выборки из определленого семейства белков
def find_proteins(df, pr, count, x=1):
    tmp = df[df['classification'] == pr].copy()
    tmp.classification = 'NORES'
    ind = np.random.randint(0, tmp.shape[0], count * x)
    return tmp.iloc[ind]


# Рисовашка для метрик после GridSearch
def ax_plotter(ax, j, y, label, indexes, line='r', i=2):
    ax[j].set_title(label, size=15)
    if i == 0:
        ax[j].set_title(label[3:], size=15)

    ax[j].set_xlabel('ngarm params')
    ax[j].set_ylabel('score')
    ax[j].set_xticks(range(0, len(indexes), 5))
    ax[j].set_xticklabels(indexes[::5], rotation=45)
    ax[j].set_ylim([0.0, 1.05])
    ax[j].set_yticks(np.arange(0, 1.01, 0.1))
    if j == 4:
        ax[j].set_ylim([0.0, 11])
        ax[j].set_yticks(np.arange(0, 10.05, 1))
    ax[j].plot(range(0, len(y)), y, f'{line}', label=label[:3])


# Функция возвращает 1 если в белке присутствует
# последовательность длины 100 в которую входит минимум 8 цистеинов
# (Условный LU-мотиф)
def find_pattern(x):
    for i, j in enumerate(x):
        if j == 'C':
            tmp = x[i:i + 80]
            if tmp.count('C') >= 8:
                return 1
    return 0


# Функция разбиения групп ортологов/гомологов на разные выборки
def train_test(dfx, df_ly6, test_size=0.4):
    df_ly6 = df_ly6.copy()
    list_suff = {}
    for i in df_ly6.index.tolist():
        list_suff[i[:4]] = list_suff.get(i[:4], 0) + 1

    index = sorted(list_suff, key=lambda x: list_suff[x], reverse=True)
    df1, df2 = {}, {}
    for i in range(0, len(index) - 1, 2):
        df1[index[i]] = list_suff[index[i]]
        df2[index[i + 1]] = list_suff[index[i + 1]]
    df2[index[-1]] = list_suff[index[-1]]

    df_ly6['split'] = df_ly6.index.map(lambda x: split_f(x, df2))
    testx = df_ly6[df_ly6.split == 1].sequence
    trainx = df_ly6[df_ly6.split == 0].sequence
    testy = df_ly6[df_ly6.split == 1].classification
    trainy = df_ly6[df_ly6.split == 0].classification

    X_train, X_test, y_train, y_test = train_test_split(dfx['sequence'],
                                                        dfx['classification'],
                                                        test_size=0.4,
                                                        random_state=1)
    X_train = pd.concat([trainx, X_train], axis=0)
    X_test = pd.concat([testx, X_test], axis=0)
    y_train = pd.concat([trainy, y_train])
    y_test = pd.concat([testy, y_test])
    return X_train, X_test, y_train, y_test
