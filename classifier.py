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
'''
train_set.to_csv("train.csv")
test_set.to_csv("test.csv")
print(len(train_set))
print(len(test_set))