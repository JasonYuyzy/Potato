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

# Where to save the figures
PROJECT_ROOT_DIR = "."
DATA_FIGURE_FOLDER = "data_figures"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", DATA_FIGURE_FOLDER)
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
STUDENT_PATH = os.path.join("datasets", "student")

#if not os.path.isdir(STUDENT_PATH):
        #os.makedirs(STUDENT_PATH)

import pandas as pd
def load_student_data(student_path=STUDENT_PATH):
    csv_path = os.path.join(".", student_path, "studentInfo.csv")
    return pd.read_csv(open(csv_path, encoding='utf-8', errors='ignore'))

student = load_student_data()
#show the data
head = student.head()
for h in head:
    if h == "id_student" or h == "region" or h == "imd_band" or h == "num_of_prev_attempts":
        continue
    student[h].hist(bins=20, figsize=(8,8))
    save_fig(h)
    plt.close()
#print("head", head)
#student.info()
a = student["code_module"].value_counts()
des = student.describe()
print(des)
import matplotlib.pyplot as plt
student.hist(bins=50, figsize=(20,15))
#save_fig("studentInfo")
plt.show()
