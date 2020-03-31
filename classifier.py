import sys

import inline as inline

assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os

# To plot pretty figures
#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

#Where to save the CSV files
DATA_PATH = os.path.join("datasets", "students_data")

'''
#download the data files
import zipfile
import urllib.request
DOWNLOAD_ROOT = "https://analyse.kmi.open.ac.uk/open_dataset/download"
if not os.path.isdir(DATA_PATH):
    os.makedirs(DATA_PATH)
tgz_path = os.path.join(DATA_PATH, "student.zip")
urllib.request.urlretrieve(DOWNLOAD_ROOT, tgz_path)
#tarfile.getnames(tgz_path)
student_file = zipfile.ZipFile(tgz_path)
student_file.extractall(path=DATA_PATH)
student_file.close()
'''
# Where to save the figures
IMAGES_PATH = os.path.join(".", "images", "data_figures")
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

#get the data
import os
import tarfile


#if not os.path.isdir(STUDENT_PATH):
        #os.makedirs(STUDENT_PATH)

import pandas as pd
def load_student_data(student_path=DATA_PATH):
    csv_path = os.path.join(".", student_path, "studentInfo.csv")
    return pd.read_csv(open(csv_path, encoding='utf-8', errors='ignore'))

student = load_student_data()
#show the data
def create_figure (data):
    head = data.head()
    for h in head:
        if h == "id_student" or h == "region" or h == "imd_band" or h == "num_of_prev_attempts":
            continue
        student[h].hist(bins=20, figsize=(8,8))
        save_fig(h)
        plt.close()
#print("head", head)
#student.info()
#create_figure(student)
#a = student["code_module"].value_counts()
#des = student.describe()
#save_fig("studentInfo")
#plt.show()

np.random.seed(42)
import numpy as np
test_ratio = 0.2
'''
#out of order
def split_set(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_set(student, test_ratio)
'''
#in order
import hashlib
def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return bytearray(hash(np.int64(identifier)).digest())[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

student_with_id = student.reset_index()
train_set, test_set = split_train_test_by_id(student_with_id, 0.2, "index")

train_set.to_csv("train.csv")
test_set.to_csv("test.csv")
order = train_set.sort_values('studied_credits',inplace=False,ascending=True)
t = train_set["code_module"].describe()
#conbime two data sets
bbb0 = train_set.loc[(train_set["code_module"] == 'BBB')]
bbb1 = train_set.loc[(train_set["code_module"] == 'AAA')]
bbb=pd.concat([bbb0, bbb1],axis=0,ignore_index=False)
c = train_set["code_module"].value_counts()
uni = train_set["code_module"].unique()
print(train_set["code_module"].describe())
print(t.dtype)
for pra in uni:
    print(pra+": ", c[pra])
print(bbb)
print(bbb.loc[(bbb["final_result"] == "Pass")]["final_result"].count())
print(c)
print(len(test_set))
head = test_set.head()
for h in head:
    print(h)

def gini_impurity_for_root (data, decision, type):
    gini_group = []
    gini_impurity = 0
    if type == object:
        total_num = len(data)
        count_num = data[decision].value_counts()
        uni_list = data[decision].unique()
        for choice in uni_list:
            final_list = data.loc[(data[decision] == choice)]
            pass_num = final_list.loc[(final_list["final_result"] == "Pass")]["final_result"].count()
            unpass_num = count_num[choice] - pass_num
            gini_impurity = 1 - (pass_num/count_num[choice])**2 - (unpass_num/count_num[choice])**2
            gini_group.append([final_list,(count_num[choice]/total_num) * gini_impurity])
        for member in gini_group:
            gini_impurity += member[1]
        return [gini_group[0][0], gini_group[1][0], gini_impurity]

def choose_the_root_with_gini_impurity (data):
    compare_group = ["", 1]
    for head in data.head():
        if head == "final_result":
            break
        gini_impurity = gini_impurity_for_root (data, head, type)
        if gini_impurity < compare_group[2]:
            compare_group = [head, gini_impurity, gini_impurity[0], gini_impurity[1]]
    result = []
