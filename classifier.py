import hashlib
import os
import math
import zipfile
import warnings
import pandas as pd
import numpy as np
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
#from sklearn.grid_search import GridSearchCV

#ignoring the filter difference
warnings.filterwarnings("ignore")

DATA_PATH = ("./student_data")
#download the data files
DOWNLOAD_ROOT = "https://analyse.kmi.open.ac.uk/open_dataset/download"
if not os.path.isdir(DATA_PATH):
    os.makedirs(DATA_PATH)
tgz_path = os.path.join(DATA_PATH, "student.zip")
urllib.request.urlretrieve(DOWNLOAD_ROOT, tgz_path)
student_file = zipfile.ZipFile(tgz_path)
student_file.extractall(path=DATA_PATH)
student_file.close()

IMAGES_PATH = ("./data_figures")
os.makedirs(IMAGES_PATH, exist_ok=True)
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

#load the data
studentInfo_data = pd.read_csv('./student_data/studentInfo.csv')
studentAssessment = pd.read_csv('./student_data/studentAssessment.csv')
assessment = pd.read_csv('./student_data/assessments.csv')
studentVle = pd.read_csv('./student_data/studentVle.csv')
courses = pd.read_csv('./student_data/courses.csv')
studentRegistration = pd.read_csv('./student_data/studentRegistration.csv')

#mix the studentAssessment data and assessment data into one data set to form up a score average in the final data set easily
new_mix_score = pd.merge(studentAssessment, assessment,  left_on='id_assessment', right_on='id_assessment')



#shows the information for each student ID
studentInfo_data["id_student"].value_counts()

# sum up and average the score and create a new csv file (with the score)
need_head = True
studentInfo_module_AAA = studentInfo_data.loc[(studentInfo_data["code_module"] == 'AAA')]
student_module_code_list = studentInfo_data["code_module"].unique()
for module_code in student_module_code_list:
    student_data_with_the_module_code = studentInfo_data.loc[(studentInfo_data["code_module"] == module_code)]
    code_presentation_list = student_data_with_the_module_code["code_presentation"].unique()
    for presentation_code in code_presentation_list:
        student_data_with_the_presentation_code = student_data_with_the_module_code.loc[
            (student_data_with_the_module_code["code_presentation"] == presentation_code)]
        data_of_the_assessments_id_with_score = new_mix_score.loc[
            (new_mix_score["code_module"] == module_code) & (new_mix_score["code_presentation"] == presentation_code)]
        data_of_the_sum_click = studentVle.loc[
            (studentVle["code_module"] == module_code) & (studentVle["code_presentation"] == presentation_code)]
        data_of_the_unregistration = studentRegistration.loc[(studentRegistration["code_module"] == module_code) & (
                    studentRegistration["code_presentation"] == presentation_code)]

        student_id_with_certain_module_presentation = student_data_with_the_presentation_code["id_student"].unique()
        new_score_data = {'id_student': [], 'score': []}
        new_vle_data = {'id_student': [], 'sum_click': []}
        new_registration_data = {'id_student': [], 'register': []}
        # print(student_id_with_certain_module_presentation)
        for student in student_id_with_certain_module_presentation:
            # print(student)
            new_score_data['id_student'].append(student)
            new_vle_data['id_student'].append(student)
            new_registration_data['id_student'].append(student)
            total_score = 0
            sum_click = 0
            score_data = data_of_the_assessments_id_with_score.loc[
                (data_of_the_assessments_id_with_score['id_student'] == student)]
            click_data = data_of_the_sum_click.loc[(data_of_the_sum_click['id_student'] == student)]
            if math.isnan(data_of_the_unregistration.loc[(data_of_the_unregistration['id_student'] == student)][
                              'date_unregistration']):
                new_registration_data['register'].append(1)
            else:
                new_registration_data['register'].append(0)
            new_score_data['score'].append(round(score_data['score'].sum() / len(score_data), 3))
            new_vle_data['sum_click'].append(click_data['sum_click'].sum())

        new_score_data = pd.DataFrame(new_score_data)
        new_vle_data = pd.DataFrame(new_vle_data)
        new_registration_data = pd.DataFrame(new_registration_data)
        new_with_score = pd.merge(student_data_with_the_presentation_code, new_score_data, left_on='id_student',
                                  right_on='id_student')
        new_with_click = pd.merge(new_with_score, new_vle_data, left_on='id_student', right_on='id_student')
        new_with_register = pd.merge(new_with_click, new_registration_data, left_on='id_student', right_on='id_student')

        new_with_register.to_csv('./student_data/new_total_studentInfo.csv', mode='a', index=False, header=need_head)
        need_head = False


#Load the new data (with the score and vle)
new_total_studentInfo = pd.read_csv('./student_data/new_total_studentInfo.csv')
new_total_studentInfo.sample(5)


import matplotlib.pyplot as plt
import matplotlib as mpl
#Show the scatter diagram of the sum_click and score data
click_score = new_total_studentInfo.drop(columns=["code_module","code_presentation", "id_student", "gender", "region", "highest_education", "imd_band", "age_band", "num_of_prev_attempts", "studied_credits", "disability", "final_result"])
click_score.sample(5)
click_score.plot(kind="scatter", x="sum_click", y="score", alpha=0.5, sharex=False)
save_fig("sum_click_and_score")


#delete the useless data
new_total_studentInfo.drop([ 'id_student', 'num_of_prev_attempts'], axis = 1, inplace=True)
new_total_studentInfo.sample(5)

#Label Encoding
from sklearn.preprocessing import LabelEncoder
for col in new_total_studentInfo.columns:
    if new_total_studentInfo[col].nunique() == 2:
        le = LabelEncoder()
        le.fit(new_total_studentInfo[col])
        new_total_studentInfo[col] = le.transform(new_total_studentInfo[col])
new_total_studentInfo.sample(5)


new_total_studentInfo = new_total_studentInfo.fillna(0)
onehotencoding_data = ['highest_education', 'code_module', 'code_presentation', 'age_band', 'final_result', 'imd_band']
def LabelEncoder(data_encoding, outfile):
    lables = outfile[data_encoding].unique().tolist()
    outfile[data_encoding] = outfile[data_encoding].apply(lambda n: lables.index(n))
    return outfile

for data_set in onehotencoding_data:
    first_decration_data_set = LabelEncoder(data_set, new_total_studentInfo)


first_decration_data_set.sample(5)


#OneHotEncoding
final_data = pd.get_dummies(first_decration_data_set)
#Transfer the final result to the original data
final_data['final_result'].replace(0,'Pass',inplace= True)
final_data['final_result'].replace(1,'Withdrawn',inplace= True)
final_data['final_result'].replace(2,'Fail',inplace= True)
final_data['final_result'].replace(3,'Distinction',inplace= True)
final_data.sample(5)


#Devided in to to groups train group and test group
from sklearn.model_selection import train_test_split
y = final_data['final_result'].values
X = final_data.drop(['final_result'], axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
print("X_train_shape:", X_train.shape, " y_train_shape:", y_train.shape)
print("X_test_shape:", X_test.shape,"  y_test_shape:", y_test.shape)


# DECISION TREE
# Through the max depth to train the data
def depth_score_D(d):
    clf = DecisionTreeClassifier(max_depth=d)
    clf.fit(X_train, y_train)

    return (clf.score(X_train, y_train), clf.score(X_test, y_test))


# DECISION TREE
def find_the_best_depth_D(depth_list):
    depths = np.arange(depth_list[0], depth_list[1])
    scores = [depth_score_D(d) for d in depths]
    tr_scores = [s[0] for s in scores]
    te_scores = [s[1] for s in scores]

    tr_best_index = np.argmax(tr_scores)
    te_best_index = np.argmax(te_scores)

    return [depths[tr_best_index], tr_scores[tr_best_index], depths[te_best_index], te_scores[te_best_index]]


# DECISION TREE
# list each best depth result in each setting range
# group test
depth_group = [[1, 50], [1, 30], [1, 15], [1, 10], [1, 5], [1, 4], [5, 10], [5, 9], [2, 8], [3, 7]]
for depth_list in depth_group:
    result = find_the_best_depth_D(depth_list)
    print(depth_list)
    print("BestDepthForTR:", result[0], " bestdepth_train_score:", result[1])
    print("BestDepthForTE:", result[2], " bestdepth_test_score:", result[3])

#Random Forest
depths = np.arange(1, 10)
scores = [depth_score_D(d) for d in depths]
tr_scores = [s[0] for s in scores]
te_scores = [s[1] for s in scores]
#find the highest score
tr_best_index = np.argmax(tr_scores)
print("bestdepth:", depths[tr_best_index], " bestdepth_train_score:", tr_scores[tr_best_index], '\n')
te_best_index = np.argmax(te_scores)
print("bestdepth:", depths[te_best_index], " bestdepth_test_score:", te_scores[te_best_index], '\n')

#draw and save the image
%matplotlib inline
from matplotlib import pyplot as plt
#depths = np.arange(1,30)
plt.figure(figsize=(6,4), dpi=120)
plt.grid()
plt.xlabel('max depth of decison tree')
plt.ylabel('Scores')
plt.plot(depths, te_scores, label='test_scores')
plt.plot(depths, tr_scores, label='train_scores')
plt.legend()
save_fig("depth_for_D")

# DECISION TREE
# using the mini impurity
from sklearn import metrics


def minsplit_score_D(val):
    clf = DecisionTreeClassifier(min_impurity_decrease=val)
    clf.fit(X_train, y_train)

    return (clf.score(X_train, y_train), clf.score(X_test, y_test))


# DECISION TREE
def find_the_best_min_D(test_range):
    vals = np.linspace(test_range[0], test_range[1], test_range[2])
    scores = [minsplit_score_D(v) for v in vals]
    tr_scores = [s[0] for s in scores]
    te_scores = [s[1] for s in scores]

    bestmin_index = np.argmax(te_scores)
    bestscore_te = te_scores[bestmin_index]
    bestscore_tr = tr_scores[bestmin_index]

    return [vals[bestmin_index], bestscore_tr, bestscore_te]

#DECISION TREE
#list each best min impurity in each estting range
#group test
range_list = [[0, 0.9, 1000], [0, 0.5, 1000], [0, 0.3, 1000], [0, 0.2, 1000], [0, 0.1, 1000], [0, 0.05, 1000]]

for test_range in range_list:
    result = find_the_best_min_D(test_range)
    print(test_range)
    print("best_min:", result[0], "best_train_score:", result[1],"best_test_score", result[2], "\n")

#Decision Tree
#return the value as the min impurity decrease val
#single test
#produce 50 numbers that between 0 and 0.9 and save it as a gini group
vals = np.linspace(0, 0.2, 1000)
scores = [minsplit_score_D(v) for v in vals]
tr_scores = [s[0] for s in scores]
te_scores = [s[1] for s in scores]

bestmin_index = np.argmax(te_scores)
bestscore_te = te_scores[bestmin_index]
bestscore_tr = tr_scores[bestmin_index]
print("bestmin:", vals[bestmin_index])
print("best_train_score:", bestscore_tr)
print("best_test_score:", bestscore_te)

#draw and save the image
plt.figure(figsize=(6,4), dpi=120)
plt.grid()
plt.xlabel("min_impurity_decrease")
plt.ylabel("Scores")
plt.plot(vals, te_scores, label='test_scores')
plt.plot(vals, tr_scores, label='train_scores')

plt.legend()
save_fig('min_D')


#DECISION TREE
#the last reuslt
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=8, min_impurity_decrease=0.0003)
model.fit(X_train, y_train)

print("train score:", model.score(X_train, y_train))
print("test_score:", model.score(X_test, y_test))

y_pred = model.predict(X_test)

print("precision:",metrics.precision_score(y_test, y_pred, average="micro"))
print("recall:",metrics.recall_score(y_test, y_pred, average="micro"))
print("F1_score:",metrics.f1_score(y_test, y_pred, average="micro"))

#RANDOM FOREST
#shows the original result
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
clf2 = RandomForestClassifier()
clf2.fit(X_train, y_train)
print("train score:", clf2.score(X_train, y_train))
print("test_score:", clf2.score(X_test, y_test))

y_pred = clf2.predict(X_test)

print("precision:",metrics.precision_score(y_test, y_pred, average="micro"))
print("recall:",metrics.recall_score(y_test, y_pred, average="micro"))
print("F1_score:",metrics.f1_score(y_test, y_pred, average="micro"))

# RANDOM FOREST
# Through the max depth to train the data
from sklearn.ensemble import RandomForestClassifier
def depth_score_R(d):
    clf2 = RandomForestClassifier(max_depth=d)
    clf2.fit(X_train, y_train)

    return (clf2.score(X_train, y_train), clf2.score(X_test, y_test))


# RANDOM FOREST
def find_the_best_depth_R(depth_list):
    depths = np.arange(depth_list[0], depth_list[1])
    scores = [depth_score_R(d) for d in depths]
    tr_scores = [s[0] for s in scores]
    te_scores = [s[1] for s in scores]

    tr_best_index = np.argmax(tr_scores)
    te_best_index = np.argmax(te_scores)

    return [depths[tr_best_index], tr_scores[tr_best_index], depths[te_best_index], te_scores[te_best_index]]


#RANDOM FOREST
#list each best depth result in each setting range
#group test
depth_group = [[1,50], [1,30], [1,15], [1,10], [10,25], [10,20], [10,19], [10,15]]
for depth_list in depth_group:
    result = find_the_best_depth_R(depth_list)
    print(depth_list)
    print("BestDepthForTR:", result[0], " bestdepth_train_score:", result[1])
    print("BestDepthForTE:", result[2], " bestdepth_test_score:", result[3])


#Random Forest
depths = np.arange(5, 35)
scores = [depth_score_R(d) for d in depths]
tr_scores = [s[0] for s in scores]
te_scores = [s[1] for s in scores]
#find the highest score
tr_best_index = np.argmax(tr_scores)
print("bestdepth:", depths[tr_best_index], " bestdepth_train_score:", tr_scores[tr_best_index], '\n')
te_best_index = np.argmax(te_scores)
print("bestdepth:", depths[te_best_index], " bestdepth_test_score:", te_scores[te_best_index], '\n')

#draw and save the image
%matplotlib inline
from matplotlib import pyplot as plt
#depths = np.arange(1,30)
plt.figure(figsize=(6,4), dpi=120)
plt.grid()
plt.xlabel('max depth of decison tree')
plt.ylabel('Scores')
plt.plot(depths, te_scores, label='test_scores')
plt.plot(depths, tr_scores, label='train_scores')
plt.legend()
save_fig("depth_for_R")

# RANDOM FOREST
# using the mini impurity
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
def minsplit_score_R(val):
    clf3 = RandomForestClassifier(min_impurity_decrease=val)
    clf3.fit(X_train, y_train)

    return (clf3.score(X_train, y_train), clf3.score(X_test, y_test))


# RANDOM FOREST
def find_the_best_min_R(test_range):
    vals = np.linspace(test_range[0], test_range[1], test_range[2])
    scores = [minsplit_score_R(v) for v in vals]
    tr_scores = [s[0] for s in scores]
    te_scores = [s[1] for s in scores]

    bestmin_index = np.argmax(te_scores)
    bestscore_te = te_scores[bestmin_index]
    bestscore_tr = tr_scores[bestmin_index]

    return [vals[bestmin_index], bestscore_tr, bestscore_te]

#RANDOM FOREST
#list each best min impurity in each estting range
#group test
range_list = [[0, 0.9, 1000], [0, 0.5, 1000], [0, 0.3, 1000], [0, 0.2, 1000], [0, 0.1, 1000], [0, 0.05, 1000]]

for test_range in range_list:
    result = find_the_best_min_R(test_range)
    print(test_range)
    print("best_min:", result[0], "best_train_score:", result[1],"best_test_score", result[2], "\n")


#Random Forest
#return the value as the min impurity decrease val
#single test
#produce 50 numbers that between 0 and 0.9 and save it as a gini group
vals = np.linspace(0, 0.2, 50)
scores = [minsplit_score_R(v) for v in vals]
tr_scores = [s[0] for s in scores]
te_scores = [s[1] for s in scores]

bestmin_index = np.argmax(te_scores)
bestscore_te = te_scores[bestmin_index]
bestscore_tr = tr_scores[bestmin_index]
print("bestmin:", vals[bestmin_index])
print("best_train_score:", bestscore_tr)
print("best_test_score:", bestscore_te)

#draw and save the image
plt.figure(figsize=(6,4), dpi=120)
plt.grid()
plt.xlabel("min_impurity_decrease")
plt.ylabel("Scores")
plt.plot(vals, te_scores, label='test_scores')
plt.plot(vals, tr_scores, label='train_scores')

plt.legend()
save_fig('min')
#RANDOM FOREST
#final result
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
clf2 = RandomForestClassifier(max_depth=18, min_impurity_decrease=0.00003)
clf2.fit(X_train, y_train)
print("train score:", clf2.score(X_train, y_train))
print("test_score:", clf2.score(X_test, y_test))

y_pred = clf2.predict(X_test)

print("precision:",metrics.precision_score(y_test, y_pred, average="micro"))
print("recall:",metrics.recall_score(y_test, y_pred, average="micro"))
print("F1_score:",metrics.f1_score(y_test, y_pred, average="micro"))