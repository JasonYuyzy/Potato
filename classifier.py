import sys
#to draw the decision tree
import pydot
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
order = order.reset_index()
order = order.drop(["index", "level_0"], axis=1)
print("new", order.loc[:0])
t = train_set["highest_education"].describe()
#conbime two data sets
bbb0 = train_set.loc[(train_set["code_module"] == 'BBB')]
bbb1 = train_set.loc[(train_set["code_module"] == 'AAA')]
bbb=pd.concat([bbb0, bbb1],axis=0,ignore_index=False)
high = train_set["highest_education"].value_counts()
high_uni = train_set["highest_education"].unique()
#print(train_set["highest_education"].describe())
for pra in high_uni:
    print(pra+": ", high[pra])

#'''
#need to be fixed
print("\n", train_set["imd_band"].describe())
imd = train_set["imd_band"].value_counts()
#print(imd)
imd_uni = train_set["imd_band"].unique()
print(imd_uni)
for pra1 in imd_uni:
    if type(pra1) == float:
        continue
    print(str(pra1)+": ", imd[str(pra1)])

#age band data
print("\n", train_set["age_band"].describe())
age = train_set["age_band"].value_counts()
#print(age)
age_uni = train_set["age_band"].unique()
print(age_uni)
for pra2 in age_uni:
    if type(pra2) == float:
        continue
    print(str(pra2)+": ", age[str(pra2)])

#final result data
print("\n", train_set["final_result"].describe())
final = train_set["final_result"].value_counts()
#print(final)
final_uni = train_set["final_result"].unique()
print(final_uni)
for pra3 in final_uni:
    if type(pra3) == float:
        continue
    print(str(pra3)+": ", final[str(pra3)])

#code presentation data
print("\n", train_set["code_presentation"].describe())
c_p = train_set["code_presentation"].value_counts()
#print(c_p)
c_p_uni = train_set["code_presentation"].unique()
print(c_p_uni)
for pra4 in c_p_uni:
    if type(pra4) == float:
        continue
    print(str(pra4)+": ", c_p[str(pra4)])

#print(bbb.drop(["code_module"],axis=1))
#print(bbb.loc[(bbb["final_result"] == "Pass")]["final_result"].count())
#print(len(test_set))
head = test_set.head()
for h in head:
    print(h)
#'''

def gini_impurity_for_leaf (left_side, total):
    gini = 1 - (left_side/total)**2 - (1-(left_side/total))**2
    return gini

def gini_impurity_for_float (ordered, number):
    total = len(ordered)
    first_half = ordered.loc[:number]
    first_half_pass_num = first_half.loc[(first_half["final_result"] == "Pass")]["final_result"].count()
    first_half_length = len(first_half)
    first_half_gini = 1 - (first_half_pass_num/first_half_length) ** 2 - (1 - (first_half_pass_num/first_half_length)) ** 2
    second_half = ordered.loc[:number]
    second_half_pass_num = second_half.loc[(second_half["final_result"] == "Pass")]["final_result"].count()
    second_half_length = len(second_half)
    second_half_gini = 1 - (second_half_pass_num / second_half_length) ** 2 - (1 - (second_half_pass_num / second_half_length)) ** 2
    gini_impurity = (first_half_length/total) * first_half_gini + (second_half_length/total) * second_half_gini
    gini_group = [[first_half, str(first_half_pass_num)+"/"+str(first_half_length), first_half_gini], [second_half, str(first_half_pass_num)+"/"+str(second_half_length), second_half_gini]]

    return gini_impurity, gini_group

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
            #unpass_num = count_num[choice] - pass_num
            #gini = gini_impurity_for_leaf (pass_num, count_num[choice])
            gini = 1 - (pass_num / count_num[choice]) ** 2 - (1 - (pass_num / count_num[choice])) ** 2
            #gini = 1 - (pass_num/count_num[choice])**2 - (unpass_num/count_num[choice])**2
            #leaf_group.append([str(pass_num)+"/"+str(total_num-pass_num), gini])
            gini_group.append([final_list, str(pass_num)+"/"+str(total_num-pass_num), gini, (count_num[choice]/total_num) * gini])
        for member in gini_group:
            gini_impurity += member[3]
        return gini_group, gini_impurity, False
    elif type == float:
        ordered = data.sort_values(decision, inplace=False, ascending=True)
        total_num = len(ordered)
        gini_impurity = 1
        for num in range(0, total_num-2):
            f_option = decision + " < " + str((data[decision][num]+data[decision[num+1]])/2)
            new_gini_impurity, new_gini_group = gini_impurity_for_float (ordered, num)
            if new_gini_impurity < gini_impurity:
                gini_impurity = new_gini_impurity
                gini_group = new_gini_group

        return gini_group, gini_impurity, f_option

#                                            option
def choose_the_root_with_gini_impurity (data, leaf, gini):
    new_group = [leaf, gini]
    for head in data.head():
        if head == "final_result":
            break
        info = data[head].describe()
        gini_group, gini_impurity, f_option = gini_impurity_for_root (data, head, info.dtype)
        if gini_impurity < new_group[1]:
            new_group = [head, gini_impurity]

    info = data[new_group[0]].describe()
    if info.dtype == float:
        new_group[0] = f_option

#####################
#####################
###HEAD PROBLEM######
#####################
#####################

    if new_group[0] == leaf and new_group[1] == gini:
        return [], []
    else:
        if leaf == "":
            #draw the first head
            new_head = pydot.Node(new_group[0], label="<"+new_group[0]+">", style="filled", fillcolor="grey")
            graph.add_node(new_head)
        else:
            #draw the new leaf
            new_leaf = pydot.Node(new_group[0], label="<" + new_group[0] + ">", style="filled", fillcolor="grey")
            graph.add_node(new_leaf)
            graph.add_edge(pydot.Edge(leaf, new_leaf))

        result = []
        for re in gini_group:
            ###########         rest data,                     option, gini
            result.append([re[0].drop([new_group[0]], axis=1), re[1], re[2]])
        return result, new_group

def connect_tree (new_root, rest):
    tree = 0
    ###   head
    if new_root[0] not in tree:
        draw_head = 0

    for re in rest:
        leaf = pydot.Node(re[1], label="<" + re[1] + ">", style="filled", fillcolor="grey")
        graph.add_node(leaf)
        graph.add_edge(pydot.Edge(new_root[0], leaf))

graph = pydot.Dot(graph_type='graph')
node_CON = pydot.Node("CON%d" % final, label="<CON>", style="filled", fillcolor="red")
graph.add_node(node_CON)
graph.add_edge(pydot.Edge("CON%d" % final, node_con))

'''
data = train_set
waiting_list = [[data, "", 1]]
while True:
    if waiting_list != []:
        next = waiting_list[0]
        rest, new_root = choose_the_root_with_gini_impurity(next[0], next[1], next[2])
        for re in rest:
            connect_tree (new_root, rest)
        for res in rest:
            waiting_list.append(res)
'''