"""

Custom XAI Library with functions like Feature Importance Score & Plot, PDP, Plotting Decision Tree & Explaining it,
Detection of Outlier data points, Anomaly Detection of Time-Series etc.

Code Author : Arunava Das & Piyush Nandi

"""
import pandas as pd
import numpy as np
from sklearn import metrics, tree
from matplotlib import pyplot
import warnings
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os
import sqlite3
from scipy.signal import find_peaks,peak_prominences,argrelmax,argrelmin,peak_widths,savgol_filter
from scipy.stats import entropy
from sklearn.cluster import KMeans
from graphviz import Source
from skater.core.explanations import Interpretation
from skater.model import InMemoryModel
from math import floor,inf,sqrt,ceil,log
from pyod.models.knn import KNN
import xml.etree.cElementTree as ET
import ast
import sys
from statsmodels.graphics.tsaplots import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import category_encoders
import re
from skater.core.local_interpretation.lime.lime_tabular import LimeTabularExplainer
from collections import OrderedDict, Counter
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
import random
import pickle
import json
from numpy.random import seed
from time import time
from sklearn.tree.export import export_text, export_graphviz
import genfact as gf


warnings.filterwarnings('ignore')
pyplot.style.use('fivethirtyeight')


class ModelPerformance:
    def __init__(self):
        self.metrics_text = []

    def get_metrics(self, true_labels, predicted_labels, problem_type):
        def mean_absolute_percentage_error(y_true, y_pred):
            y_true_new, y_pred_new = [], []
            for i, x in enumerate(y_true):
                if x != 0:
                    y_true_new.append(x)
                    y_pred_new.append(y_pred[i])
            y_true = y_true_new
            y_pred = y_pred_new

            y_true, y_pred = np.array(y_true), np.array(y_pred)

            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        if 'classification' in problem_type:
            self.metrics_text.append('<br><b>Accuracy: </b>'+ str(np.round(metrics.accuracy_score(true_labels, predicted_labels),4)))
            self.metrics_text.append('<br><b>Precision: </b>'+ str(np.round(
                metrics.precision_score(true_labels,
                                        predicted_labels,
                                        average='weighted'),
                4)))
            self.metrics_text.append('<br><b>Recall: </b>'+ str(np.round(
                metrics.recall_score(true_labels,
                                     predicted_labels,
                                     average='weighted'),
                4)))
            self.metrics_text.append('<br><b>F1 Score: </b>'+ str(np.round(
                metrics.f1_score(true_labels,
                                 predicted_labels,
                                 average='weighted'),
                4)))
        if 'regression' in problem_type:
            self.metrics_text.append('<br><b>R2 Score: </b>'+ str(np.round(
                metrics.r2_score(true_labels, predicted_labels),
                4)))
            self.metrics_text.append('<br><b>Explained Variance Score: </b>'+ str(np.round(
                metrics.explained_variance_score(true_labels,
                                                 predicted_labels),
                4)))
            self.metrics_text.append('<br><b>Mean Absolute Error: </b>'+ str(np.round(
                metrics.mean_absolute_error(true_labels,
                                            predicted_labels),
                4)))
            self.metrics_text.append('<br><b>Mean Absolute Percentage Error: </b>' + str(np.round(
                mean_absolute_percentage_error(true_labels,
                                            predicted_labels),
                4)))
            self.metrics_text.append('<br><b>Mean Squared Error: </b>'+ str(np.round(
                metrics.mean_squared_error(true_labels,
                                           predicted_labels),
                4)))

    def display_confusion_matrix(self, true_labels, predicted_labels, classes=[1, 0]):

        total_classes = len(classes)
        level_labels = [total_classes * [0], list(range(total_classes))]

        cm = metrics.confusion_matrix(y_true=true_labels, y_pred=predicted_labels,
                                      labels=classes)
        cm_frame = pd.DataFrame(data=cm,
                                columns=pd.MultiIndex(levels=[['Predicted:'], classes],
                                                      codes=level_labels),
                                index=pd.MultiIndex(levels=[['Actual:'], classes],
                                                    codes=level_labels))

        self.metrics_text.append('<br><b>Prediction Confusion Matrix: </b>')
        self.metrics_text.append('-' * 30)
        self.metrics_text.append(cm_frame)

    def display_classification_report(self, true_labels, predicted_labels, classes=[1, 0]):

        report = metrics.classification_report(y_true=true_labels,
                                               y_pred=predicted_labels,
                                               labels=classes)
        self.metrics_text.append('<br><b>Model Classification report: </b>')
        self.metrics_text.append('-' * 30)
        self.metrics_text.append(report)

    def display_model_performance_metrics(self, true_labels, predicted_labels, classes=[1, 0], problem_type='classification', method = 'Global'):
        self.metrics_text.append('<br><b>Model Performance metrics for {} model: </b>'.format(method))
        self.metrics_text.append('-' * 30)
        self.get_metrics(true_labels=true_labels, predicted_labels=predicted_labels, problem_type=problem_type)

        if 'classification' in problem_type:
            self.display_classification_report(true_labels=true_labels, predicted_labels=predicted_labels,
                                          classes=classes)
            self.display_confusion_matrix(true_labels=true_labels, predicted_labels=predicted_labels,
                                     classes=classes)

        return self.metrics_text


class ExplainTree:
    def __init__(self,file_name,data, data_disp, problem_type, freq, top_n = None, total_samples = None):
        self.file_name = file_name
        self.nodelist = []
        self.tree = {}
        self.paths = []
        self.data = data
        self.data_disp = data_disp
        self.top_n = top_n
        self.problem_type = problem_type
        self.freq = freq
        self.response_var = self.file_name.split("\\")[-1].replace('.svg', '').replace("dtree_structure_","")
        self.filter_dict = {}
        if total_samples is None:
            self.total_samples = int(data.shape[0] * 0.7)
        else:
            self.total_samples = total_samples

    def explain(self):
        self.nodelist = self.create_node_info(self.file_name)
        self.tree = self.Create_tree_from_nodes(self.nodelist).generate_tree()
        self.paths = self.calculate_critical_paths(self.generate_paths(self.tree))
        html = self.simple_NLP(self.paths)

        return html, self.filter_dict

    class Create_tree_from_nodes:
        def __init__(self, nodelist):
            self.nodelist = nodelist

        def insert_node(self, parent):
            if parent['isLeafNode']:
                return {'data': parent, 'left': None, 'right': None}
            return {'data': parent, 'left': self.insert_node(self.nodelist.pop(0)),
                    'right': self.insert_node(self.nodelist.pop(0))}

        def generate_tree(self):
            return self.insert_node(self.nodelist.pop(0))

    def create_node_info(self,file_name):
        tree = ET.ElementTree(file=file_name)
        root = tree.getroot()

        parent_graph = list(root)

        nodes = list(parent_graph[0])
        nodelist = []

        oppositeSignDict = {'≤': '>', '>': '≤', '≥': '<', '<': '≥'}
        convertSign = {'>': '>', '<=': '≤', '<': '<', '>=': '≥','≤': '≤', '≥': '≥'}

        for node in nodes:
            if node.tag.split('}')[-1] == 'g' and 'node' in node.get('id'):
                attributes = list(node)
                nodelist.append({'id': int(node.get('id').lstrip('node'))})
                text_attributes = list(filter(lambda x: x.tag.split('}')[-1] == 'text', attributes))
                temp_textlist = [x.text for x in text_attributes]
                nodelist[-1]['#'] = int(temp_textlist[0].split()[-1].strip('#'))

                if 'gini =' in temp_textlist[1] or 'mse =' in temp_textlist[1] or 'entropy =' in temp_textlist[1] \
                        or 'mae =' in temp_textlist[1]:
                    nodelist[-1]['isLeafNode'] = True
                    if 'classification' in self.problem_type:
                        nodelist[-1]['mse'] = 0
                        nodelist[-1]['value'] = ast.literal_eval(", ".join(temp_textlist[3:-1]).split("=")[-1].strip())
                        max_val = max(nodelist[-1]['value'])
                        nodelist[-1]['value'] = round((max_val / sum(nodelist[-1]['value'])) * 100, 2)
                        nodelist[-1]['class'] = temp_textlist[-1].split("=")[-1].strip()
                        nodelist[-1]['sample'] = temp_textlist[2].split("=")[-1].strip()
                        if '%' in nodelist[-1]['sample']:
                            nodelist[-1]['sample'] = float(nodelist[-1]['sample'].replace("%", ""))
                        else:
                            nodelist[-1]['sample'] = round((int(nodelist[-1]['sample']) / self.total_samples) * 100, 1)
                    else:
                        nodelist[-1]['value'] = 0
                        nodelist[-1]['mse'] = float(temp_textlist[1].split("=")[-1].strip())
                        nodelist[-1]['class'] = float(temp_textlist[-1].split("=")[-1].strip())
                        nodelist[-1]['sample'] = temp_textlist[-2].split("=")[-1].strip()
                        if '%' in nodelist[-1]['sample']:
                            nodelist[-1]['sample'] = float(nodelist[-1]['sample'].replace("%", ""))
                        else:
                            nodelist[-1]['sample'] = round((int(nodelist[-1]['sample']) / self.total_samples) * 100, 1)
                else:
                    nodelist[-1]['isLeafNode'] = False
                    feature_string = temp_textlist[1].strip().split()
                    nodelist[-1]['featureIndex'] = " ".join(feature_string[:-2])
                    nodelist[-1]['leftSign'], nodelist[-1]['range'] = convertSign[feature_string[-2]], feature_string[-1]
                    nodelist[-1]['range'] = float(nodelist[-1]['range'])
                    nodelist[-1]['rightSign'] = oppositeSignDict[nodelist[-1]['leftSign']]

        nodelist.sort(key=lambda x: x['id'])
        return nodelist

    def generate_paths(self,tree):
        paths = []

        def DFS(root, previous_conditions=None):
            if not previous_conditions:
                previous_conditions = []

            if root['data']['isLeafNode']:
                paths.append({'class': root['data']['class'], 'probability': root['data']['value'],
                              'condition': previous_conditions, 'sample': root['data']['sample'], 'mse': root['data']['mse']})
                return
            DFS(root['left'],
                previous_conditions + [(root['data']['featureIndex'], root['data']['leftSign'], root['data']['range'])])
            DFS(root['right'], previous_conditions + [
                (root['data']['featureIndex'], root['data']['rightSign'], root['data']['range'])])

        DFS(tree)
        return paths

    def calculate_critical_paths(self,paths):
        def sort_by_difference(a):
            min_diff = inf
            i1, i2 = 0, 1
            for i in range(0, len(a) - 1):
                for j in range(i + 1, len(a)):
                    if abs(a[i]['class'] - a[j]['class']) < min_diff:
                        min_diff = abs(a[i]['class'] - a[j]['class'])
                        i1, i2 = (i, j) if a[i]['class'] < a[j]['class'] else (j, i)
            a[0], a[i1] = a[i1], a[0]
            a[1], a[i2] = a[i2], a[1]

            for j in range(2, len(a)):
                last = a[j - 1]['class']
                min_diff = inf
                for i in range(j, len(a)):
                    if abs(a[i]['class'] - last) < min_diff:
                        min_diff = abs(a[i]['class'] - last)
                        k = i
                a[j], a[k] = a[k], a[j]

            return a

        mirror_sign = {'≥': '≤', '>': '<'}

        temp_paths = []
        path_track = {}
        path_index = 0
        if 'regression' in self.problem_type and self.top_n > 2 :
            if self.top_n > len(paths):
                self.top_n = len(paths)
            else:
                paths = sort_by_difference(paths)
        if 'classification' in self.problem_type:
            for i,path in enumerate(paths):
                if path['class'] not in path_track:
                    path_track[path['class']] = path_index
                    temp_paths.append(path)
                    path_index += 1
                else:
                    old_path = temp_paths[path_track[path['class']]]
                    old_path['probability'] = (old_path['probability'] + path['probability'])/2
                    old_path['condition'] += path['condition']
                paths = temp_paths

        for path in paths:
            old_condition = path['condition']
            new_condition = []
            condition_dict = {}
            for cond in old_condition:
                feature_index, current_sign, current_limit = cond
                if feature_index not in condition_dict:
                    condition_dict[feature_index] = {'floorSign': current_sign, 'ceilingSign': current_sign,
                                                     'ceilingValue': (
                                                         current_limit if current_sign == '≤' or current_sign == '<' else inf),
                                                     'floorValue': (
                                                         current_limit if current_sign == '≥' or current_sign == '>' else -inf)}
                else:
                    check_condition = condition_dict[feature_index]
                    if current_limit <= check_condition['ceilingValue'] and current_sign in '≤<':
                        if current_limit == check_condition['ceilingValue'] and \
                                check_condition['ceilingSign'] == '≤' and current_sign == '<':
                            check_condition['ceilingSign'] = '<'
                        else:
                            if current_limit > check_condition['floorValue']:
                                check_condition['ceilingValue'] = current_limit
                                check_condition['ceilingSign'] = current_sign
                    if current_limit >= check_condition['floorValue'] and current_sign in '≥>':
                        if current_limit == check_condition['floorValue'] and \
                                check_condition['floorSign'] == '≥' and current_sign == '>':
                            check_condition['floorSign'] = '>'
                        else:
                            if current_limit < check_condition['ceilingValue']:
                                check_condition['floorValue'] = current_limit
                                check_condition['floorSign'] = current_sign

                    if current_limit == check_condition['ceilingValue'] and current_sign in '≥>':
                        check_condition['ceilingValue'] = inf
                        check_condition['ceilingSign'] = current_sign
                    if current_limit == check_condition['floorValue'] and current_sign in '≤<':
                        check_condition['floorValue'] = -inf
                        check_condition['floorSign'] = current_sign

            for cond in condition_dict:
                check_condition = condition_dict[cond]
                if abs(check_condition['ceilingValue']) == inf and abs(check_condition['floorValue']) == inf:
                    continue
                if check_condition['ceilingValue'] > check_condition['floorValue']:
                    if check_condition['ceilingValue'] != inf and check_condition['floorValue'] != -inf:
                        new_condition.append(
                            (check_condition['floorValue'], mirror_sign[check_condition['floorSign']], cond,
                             check_condition['ceilingSign'], check_condition['ceilingValue']))
                    elif check_condition['floorValue'] == -inf:
                        new_condition.append((cond, check_condition['ceilingSign'], check_condition['ceilingValue']))
                    else:
                        new_condition.append((cond, check_condition['floorSign'], check_condition['floorValue']))

            path['condition'] = new_condition

        final_paths = []
        for path in paths:
            if path['condition']:
                final_paths.append(path)

        return final_paths

    def multiclass_format(self, feature_name):

        map_dict = {}

        if 'category' in str(self.data_disp[feature_name].dtype) or 'object' in str(self.data_disp[feature_name].dtype):

            for i,x in enumerate(self.data_disp[feature_name]):
                map_dict[self.data[feature_name].iloc[i]] = x

            return map_dict
        return None

    def multiclass_summarise(self,map_dict, feature_index, primary_sign, primary_value, secondary_sign=None, secondary_value=None):
        if not map_dict:
            return None

        class_list = [[], []]
        for i,sign in enumerate([primary_sign, secondary_sign]):
            if sign and sign in '≤<':
                if sign == "≤":
                    class_list[i] += [map_dict[x] for x in map_dict if x <= float(primary_value if i==0 else secondary_value)]
                else:
                    class_list[i] += [map_dict[x] for x in map_dict if x < float(primary_value if i==0 else secondary_value)]
            elif sign and sign in '≥>':
                if sign == "≥":
                    class_list[i] += [map_dict[x] for x in map_dict if x >= float(primary_value if i==0 else secondary_value)]
                else:
                    class_list[i] += [map_dict[x] for x in map_dict if x > float(primary_value if i==0 else secondary_value)]
        if primary_sign == '=':
            class_list[0] = [] if int(primary_value) not in map_dict else [map_dict[int(primary_value)]]

        class_list = set(class_list[0]).intersection(set(class_list[1])) if class_list[0] and class_list[1] else set(class_list[0])
        temp_freq = []
        th = 0
        for f,freq in enumerate(self.freq[feature_index]):
            if freq[0] in class_list:
                temp_freq.append(freq[0])
                th+=1
                if th==5:
                    break

        if feature_index not in self.filter_dict:
            self.filter_dict[feature_index] = temp_freq
        else:
            self.filter_dict[feature_index] += temp_freq

        class_list = temp_freq + ['etc.'] if len(temp_freq)>5 else temp_freq

        return class_list

    def simple_NLP(self,paths):
        stmnt_dir = "Statements"
        os.makedirs(stmnt_dir, exist_ok=True)

        explain_list = []
        mirror_sign = {'<': '>', '≤': '≥'}

        if len(set([path['class'] for path in paths])) > 2:

            prob_set = sorted(set([((path['probability'] + path['sample']) if 'classification' in self.problem_type else (path['sample'] - path['mse'])) for path in paths]))

            top_threshold = self.top_n if self.top_n and 0 < self.top_n <= len(prob_set) else len(prob_set)
            top_probs = prob_set[-top_threshold:] if len(prob_set) >= top_threshold else prob_set

        else:
            top_probs = [((path['probability'] + path['sample']) if 'classification' in self.problem_type else (path['sample'] - path['mse'])) for path in paths]
            top_threshold = len(top_probs)

        for path in paths:
            if 'classification' in self.problem_type and (path['probability'] + path['sample']) not in top_probs:
                continue
            if 'regression' in self.problem_type and (path['sample'] - path['mse']) not in top_probs:
                continue

            condition_string = []
            for cond in path['condition']:
                if len(cond) == 3:
                    feature_index, current_sign, current_limit = cond
                    feature_info = feature_index
                    check_multiclass = self.multiclass_format(feature_index)
                    classlist = []
                    if check_multiclass:
                        classlist = self.multiclass_summarise(check_multiclass,feature_index, current_sign, current_limit)
                        if not classlist:
                            check_multiclass = False
                    if not check_multiclass:
                        condition_string.append("""<span style="color:#ff5959"><b>{}</b></span> {} 
                        <span style="color:#18c41d"><b>{}</b></span>""".format(feature_info, current_sign, current_limit))
                    else:
                        classlist = list(map(str, classlist))
                        condition_string.append("""<span style="color:#ff5959"><b>{}</b></span> {} 
                        <span style="color:#18c41d"><b>{}</b></span>""".format(feature_info,
                              'are' if len(classlist)>1 else 'is', ", ".join(classlist)))
                elif len(cond) == 5:
                    current_lower_limit, current_left_sign, feature_index, current_right_sign, current_upper_limit, = cond
                    feature_info = feature_index
                    check_multiclass = self.multiclass_format(feature_index)
                    classlist = []
                    if check_multiclass:
                        classlist = self.multiclass_summarise(check_multiclass,feature_index, mirror_sign[current_left_sign], current_lower_limit,
                                                    current_right_sign, current_upper_limit)
                        if not classlist:
                            check_multiclass = False
                    if not check_multiclass:
                        condition_string.append("""<span style="color:#ff5959"><b>{}</b></span> {} <span style="color:#18c41d"><b>{}</b></span>
                         and {} <span style="color:#18c41d"><b>{}</b></span>"""
                                                .format(feature_info, mirror_sign[current_left_sign],
                                                        current_lower_limit,current_right_sign, current_upper_limit))
                    else:
                        condition_string.append("""<span style="color:#ff5959"><b>{}</b></span>
                         {} <span style="color:#18c41d"><b>{}</b></span>""".format(feature_info,'are' if len(classlist) > 1 else 'is', ",".join([str(a) for a in classlist])))
                else:
                    feature_index, current_limit = cond
                    feature_info = feature_index
                    check_multiclass = self.multiclass_format(feature_index)
                    classlist = []
                    if check_multiclass:
                        classlist = self.multiclass_summarise(check_multiclass,feature_index, '=', current_limit)
                        if not classlist:
                            check_multiclass = False
                    if not check_multiclass:
                        condition_string.append("""<span style="color:#ff5959"><b>{}</b></span> = 
                        <span style="color:#18c41d"><b>{}</b></span>""".format(feature_info, current_limit))
                    else:
                        if classlist:
                            condition_string.append("""<span style="color:#ff5959"><b>{}</b></span> = 
                            <span style="color:#18c41d"><b>{}</b></span>""".format(feature_info, classlist[0]))

            condition_string = ' & '.join(condition_string)
            if 'classification' in self.problem_type:
                explain_string = """If {}, Then <span style="color:#1e27d4"><b>{}</b></span>
                     will be : <span style="color:#337dff"><b>"{}"</b></span> 
                    with a probability of <span style="color:#ff0000"><b>{:.2f} %</b></span>, 
                    [ Sample size : <span style="color:#d46511"><b>{}</b> %</span> ]""".format(condition_string, self.response_var,
                                                                    path['class'], path['probability'], path['sample'])
            else:
                explain_string = """If {}, Then <span style="color:#1e27d4"><b>{}</b></span>
                    will be in range ( <span style="color:#337dff"><b>{:.2f}</b></span>, 
                    <span style="color:#337dff"><b>{:.2f}</b></span> ), 
                   [ Sample size : <span style="color:#d46511"><b>{}</b> %</span> ]""".format(condition_string, self.response_var, path['class']-sqrt(path['mse'])/2
                               , path['class']+sqrt(path['mse'])/2, path['sample'])

            explain_list.append(explain_string)

        statement = """<!DOCTYPE html>
                        <html>
                        <body>
                        <div>"""

        statement += "<br><b>Top {} Explanations : </b><br><br>".format(top_threshold)
        for i, x in enumerate(explain_list):
            statement += '<b>Explanation {} :</b> {}<br><br>'.format(i + 1, x)

        statement += """</div>
                        </body>
                        </html>"""


        with open(os.path.join(stmnt_dir, "Statements.html"), 'w+', encoding="utf-8") as sf:
            sf.write(str(statement))

        return statement


class Explainer:
    """
    #######################################################################################################################
    Parameters :
        data ( pandas.core.frame.DataFrame ) : Original data of feature variable as Pandas Dataframe

        response_var ( String ) : Name of the response variable. If response variable is not part of data, custom name for
                                response variable must be given.

        labels ( numpy.array / List ) : (Optional) ( default = None ) If labels (value of response variable) is not part of data, user has to
                                                explicitly pass labels.

        custom_class_names ( List of Strings ) : (Optional) ( default = None ) If labels are explicitly passed as the third parameter, user can
                                                    mention custom names for classes (Applicable only for classification).

        classification_threshold ( int ) : (Optional) ( default = 50 ) Threshold value to differentiate between multiclass
                                classification & regression.
                                If number of unique values is greater than threshold, it will be considered as
                                regression problem, otherwise multiclass classification.

    Return: XAI.Explainer Object
    #######################################################################################################################
    """
    def __init__(self, data, response_var, labels=None, custom_class_names=None, format_datatypes=None):
        self.data = data
        self.metrics_text = []
        self.cf_data = None
        if format_datatypes is None:
            format_datatypes = {}
        self.format_datatypes = format_datatypes
        self.clean_data()
        self.formatting_pass = False
        self.non_enc_data, self.non_enc_labels = data.copy(), None
        if not response_var and not labels:
            print("Insufficient arguments")
            return
        self.response_var = response_var if response_var else "Response_variable"
        if labels is not None and labels!="":
            self.labels = labels
        else:
            if self.response_var not in self.data.columns:
                print("Invalid name of response variable")
                return
            self.labels = self.data[self.response_var]
            self.data = self.data.drop(columns=[self.response_var])

        self.non_enc_labels = self.labels.copy()
        self.custom_class_names = custom_class_names
        self.columns = list(self.data.columns)
        self.problem_type = None
        self.xgc = None
        self.xgc_cf = None
        self.xgc_np = None
        self.objective = None
        self.predictions = None
        self.importance_data = None
        self.interpreter = None
        self.im_model = None
        self.interpreter_cf = None
        self.im_model_cf = None
        self.X_train,self.y_train, self.X_test, self.y_test = None, None, None, None
        self.X_train_cf, self.y_train_cf, self.X_test_cf, self.y_test_cf = None, None, None, None
        self.not_trained = True
        self.frequency_map = {}
        self.filter_dict = []
        self.local_exp = None

        self.data_disp, self.labels_disp = self.data.copy(), self.labels.copy()
        self.custom_data = self.data.copy()

        self.format_error = self.format_data()

    def detect_problem_category(self):
        """

        Return: List -> index 0 will be problem type, index 1 will be type of training method to be used as 'objective'
                         in XGBoost.
        """
        unique_labels = set(self.labels)
        print(len(self.labels), len(unique_labels))
        if len(unique_labels) == 2 :
            problem_type = ['Binary Classification', 'binary:logistic']

        elif 'float' in str(self.labels.dtype) or 'int' in str(self.labels.dtype):
            problem_type = ['Regression', 'reg:logistic']
        else:
            problem_type = ['Multiclass Classification', 'multi:softmax']

        return problem_type

    def get_original_labels(self):
        predictions_dict = {}
        labels = list(set(label for label in self.labels))
        labels_original = list(set(label for label in np.array(self.labels_disp)))
        for i in range(len(labels)):
            predictions_dict[labels[i]] = labels_original[i]
        return predictions_dict

    def clean_data(self):
        drop_row_list = []
        for i in range(self.data.shape[0]):
            if any([(('float' in str(type(x)) or 'int' in str(type(x))) and np.isnan(x)) or x is None for x in
                    list(self.data.loc[i])]):
                drop_row_list.append(i)
        if drop_row_list:
            self.data = self.data.drop(drop_row_list)

    def reformat_data(self):
        conversion_error = []
        for col in self.columns:
            if self.format_datatypes.get(col, None) is not None:
                datatype = self.format_datatypes.get(col)
                if datatype == 'category':
                    self.custom_data[col] = self.custom_data[col].astype('object')
                else:
                    current_type = str(self.custom_data[col].dtype)
                    if 'int' not in current_type and 'float' not in current_type:
                        try:
                            self.custom_data[col] = self.custom_data[col].astype('float64')
                            if all(floor(x) == x for i, x in enumerate(list(self.custom_data[col]))):
                                self.custom_data[col] = self.custom_data[col].astype('int64')
                        except:
                            conversion_error.append(col)

        self.non_enc_data = self.custom_data.copy()

        if self.format_datatypes.get(self.response_var, None) is not None:
            datatype = self.format_datatypes.get(self.response_var)
            if datatype == 'category':
                self.labels = self.labels.astype('object')
            else:
                current_type = str(self.labels.dtype)
                if 'int' not in current_type and 'float' not in current_type:
                    try:
                        self.labels = self.labels.astype('float64')
                        if all(floor(x) == x for i, x in enumerate(list(self.labels))):
                            self.labels = self.labels.astype('int64')
                    except:
                        conversion_error.append(self.response_var)

            self.non_enc_labels = self.labels.copy()

        return conversion_error

    def format_data(self):

        conversion_error = self.reformat_data()
        if conversion_error:
            error_mssg = ', '.join(conversion_error) + ' cannot be converted to numerical'
            print(error_mssg)
            self.formatting_pass = False
            return error_mssg

        if self.format_datatypes.get(self.response_var, None) is None:
            if 'object' in str(self.labels.dtype) or 'U15' in str(self.labels.dtype) \
                    or 'str' in str(self.labels.dtype) or 'S1' in str(self.labels.dtype) or 'category' in str(self.labels.dtype):

                self.labels = self.labels.astype('object')
                try:
                    self.labels = self.labels.astype('float64')
                except:
                    pass

            if 'bool' in str(self.labels.dtype) or ('float' in str(self.labels.dtype) and all(floor(x) == x for i, x in enumerate(list(self.labels)))):
                self.labels = np.array([int(label) for label in self.labels])

            self.labels = np.array(self.labels)

        self.problem_type = self.detect_problem_category()
        print(self.problem_type[0])
        self.labels_disp = self.labels.copy()

        if 'object' in str(self.labels.dtype):
            self.labels = pd.get_dummies(self.labels)
            self.labels = self.labels.values.argmax(1)

        for col in self.columns:
            if self.format_datatypes.get(col, None) is None:
                if 'category' in str(self.custom_data[col].dtype):
                    self.custom_data[col] = self.custom_data[col].astype('object')

        self.data_disp = self.custom_data.copy()
        self.custom_data = self.advance_encode(encode_method='m-estimator' if 'regression' in self.problem_type[0].lower() else 'leave one out')

        for col in self.columns:
            if col not in self.frequency_map:
                freq = {}
                for c in self.data_disp[col]:
                    if c in freq:
                        freq[c] += 1
                    else:
                        freq[c] = 1

                freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
                self.frequency_map[col] = freq

        if not self.custom_class_names:
            self.custom_class_names = list(self.get_original_labels().values())
            self.custom_class_names = [str(x) for i,x in enumerate(self.custom_class_names)]

        self.formatting_pass = True

    def advance_encode(self, encode_method='james stein', data=None, labels=None):
        if data is None:
            data = self.custom_data.copy()
        if labels is None:
            labels = self.labels

        cat_col_list = []

        for col in self.columns:
            if 'object' in str(data[col].dtype) or 'str' in str(data[col].dtype) or 'category' in str(data[col].dtype):
                cat_col_list.append(col)

        if encode_method == 'james stein':
            encoder = category_encoders.JamesSteinEncoder(cols=cat_col_list)
        elif encode_method == 'backward difference':
            encoder = category_encoders.BackwardDifferenceEncoder(cols=cat_col_list)
        elif encode_method == 'm-estimator':
            encoder = category_encoders.MEstimateEncoder(cols=cat_col_list)
        elif encode_method == 'one hot':
            encoder = category_encoders.OneHotEncoder(cols=cat_col_list)
        elif encode_method == 'hashing':
            encoder = category_encoders.HashingEncoder(cols=cat_col_list, max_process=1)
        elif encode_method == 'leave one out':
            encoder = category_encoders.LeaveOneOutEncoder(cols=cat_col_list)

        enc = encoder.fit(data, labels)
        transformed = enc.transform(data)

        return transformed

    def train_test(self):
        # traing the model
        data = self.custom_data.copy()
        featuredata_values_only = data.iloc[:, :].values
        data[self.response_var] = pd.Series(self.labels)
        featuredata = data.copy()
        targetclasses_indx = data.shape[1] - 1
        original_classdata = data.iloc[:, -1].values

        self.xgc = xgb.XGBClassifier(n_estimators=500, max_depth=int(self.custom_data.shape[1] * 2), base_score=0.5,
                                objective=self.problem_type[1], random_state=42)

        X_train, X_test, y_train, y_test = train_test_split(featuredata_values_only, original_classdata, test_size=0.01, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        print('Split ratio:', len(X_train), len(X_test))

        self.xgc.fit(X_train, y_train)

        self.objective = self.problem_type[1]
        self.problem_type = self.problem_type[0].lower()

        self.prepare_counterfactual_data(featuredata, targetclasses_indx)

        # making predictions
        self.predictions = self.xgc.predict(X_test)

        # model performance evaluation
        class_labels = list(set(self.labels))
        self.metrics_text = ModelPerformance().display_model_performance_metrics(true_labels=y_test,
                                              predicted_labels=self.predictions,
                                              classes=class_labels, problem_type=self.problem_type, method='XGBoost Global')

        for text in self.metrics_text:
            print(text)

        self.interpreter = Interpretation(training_data=self.X_train, training_labels=self.y_train,
                                     feature_names=list(self.columns))
        self.im_model = InMemoryModel((self.xgc.predict if 'regression' in self.problem_type else self.xgc.predict_proba),
                                      examples=self.X_train,
                                      target_names=(self.custom_class_names if 'classification' in self.problem_type else None),
                                      feature_names=list(self.columns),
                                      model_type=("regressor" if 'regression' in self.problem_type else "classifier"))

        self.not_trained = False
        self.get_performance_metric(self.metrics_text, 'global')

    def prepare_counterfactual_data(self, featuredata, targetclasses_indx):
        featuredtype = []
        for col in list(self.columns):
            if self.format_datatypes.get(col) == 'category':
                featuredtype.append('cat')
            else:
                featuredtype.append('con')

        if self.format_datatypes.get(self.response_var) == 'category':
            featuredtype.append('cat')
        else:
            featuredtype.append('con')

        factuals, counterfactuals, factclass, cfactclass, classdistribution = \
            gf.generate_counterfactuals(featuredata, featuredtype, targetclasses_indx, model=self.xgc)

        X_train = np.concatenate([factuals, counterfactuals], axis=0)
        y_train_cf = self.xgc.predict(X_train)

        self.X_train_cf = pd.DataFrame(X_train, columns=self.columns)
        self.y_train_cf = y_train_cf

    def get_performance_metric(self, metrics_text, method='global'):
        if self.not_trained:
            self.train_test()

        os.makedirs("Performance Metrics", exist_ok=True)
        with open(os.path.join("Performance Metrics", "performance_{}.txt".format(method)), 'w+') as f:
            f.write("\n".join([str(text) for text in metrics_text]))

        return "\n".join([str(text) for text in metrics_text])

    def get_feature_scores(self, plot_size=(20, 9), plot=True, filename = None, sort=True, normalize=False):
        """
        Parameter:
            plot_size ( 2-tuple ) : (Optional) (default = (18,9) ) Resolution of plot image, in scale of 100.

            plot ( boolean ) : (Optional) (default = False ) Plot image will be saved only if this parameter is True.

            filename ( String ) : (Optional) (default = None ) Custom file names for the plot image. If it is None, default
                                    file name will be used.

            sort ( boolean ) : (Optional) (default = True ) Feature scores will not be sorted if it is False.

        Return: Pandas.Dataframe -> Feature scores matrix with 'weight', 'gain', 'cover', 'total_gain' & 'total_cover'
        """
        os.makedirs("Feature Importance", exist_ok=True)
        if not self.formatting_pass:
            print('Data formatting error')
            return self.format_error

        def normalize_list(x, arr):
            return (x - min(arr)) / (max(arr) - min(arr))

        if self.not_trained:
            self.train_test()

        importance_data = {}
        wg = []
        tempw = 1

        for importance_type in ['weight', 'gain', 'cover', 'total_gain', 'total_cover']:
            score = self.xgc.get_booster().get_score(importance_type=importance_type)
            if not importance_data.get("Feature Names", None):
                importance_data["Feature Names"] = list(score.keys())
            score = list(score.values())
            if normalize and importance_type in ['weight', 'gain']:
                score = list(map(lambda x: normalize_list(x, score), score))
            importance_data[importance_type] = score
            if importance_type == 'weight':
                tempw = score
            elif importance_type == 'gain':
                wg = [tempw[i] * score[i] for i in range(len(score))]

        importance_data = pd.DataFrame(importance_data)
        importance_data['weight*gain'] = wg

        if sort:
            importance_data = importance_data.sort_values(['weight*gain'], ascending = (False,))

        if plot:
            fig, ax = pyplot.subplots(figsize=plot_size)
            x = list(importance_data["Feature Names"])[::-1]
            x = [list(self.columns)[int(val.replace('f', ''))] for val in x]
            importance_data["Feature Names"] = x[::-1]
            y = list(importance_data['weight*gain'])[::-1]
            y_norm = [yk / sum(y) for yk in y]
            ax.barh(x, y_norm)
            ax.set_title("Feature Importance Scores on " + self.response_var)

            if not filename:
                pyplot.savefig(os.path.join("Feature Importance", "FeatureImportance_XGB_{}.png".format(self.response_var)))
            else:
                pyplot.savefig(os.path.join("Feature Importance", filename))

        self.importance_data = importance_data

        return importance_data

    def pdp(self, feature_var, fig_size=(16, 9), fontsize=40, labelgap=50, max_legend=6,
            top_n_percent=30, peak_width_threshold=2.5, gap_threshold=5):
        """
        Parameter:
            feature_var ( String ) : Name of the feature variable of which PDP will be constructed.

            fig_size ( 2-tuple ) : (Optional) (default = (16,9) ) Resolution of plot image, in scale of 100.

            peak_info ( boolean ) : (Optional) (default = False ) Peaks info will be displayed if this parameter is True.

        """
        def get_left_right_index(x, n, already_taken=None):
            if not already_taken:
                already_taken = set()

            left = (x-1 if x > 0 else 0)
            right = (x+1 if x < n-1 else n-1)

            if len(already_taken) > 0:
                if left in already_taken:
                    left += 1
                if right in already_taken:
                    right -= 1

            return left, right

        if not self.formatting_pass:
            print('Data formatting error')
            return self.format_error

        if self.not_trained:
            self.train_test()

        os.makedirs(os.path.join('PDP', self.response_var), exist_ok=True)
        if 'str' in str(type(feature_var)):
            feature_var = [feature_var]

        way = len(feature_var)
        feature_var_initial = feature_var
        if way > 1:
            feature_var = [tuple(feature_var)]
        else:
            feature_var = list(feature_var)

        unique_values = set(self.data_disp[feature_var_initial[0]])

        font = {'family': 'Times New Roman',
                'size': fontsize}

        if len(unique_values) > 2:
            r = self.interpreter.partial_dependence.plot_partial_dependence(feature_var, self.im_model,
                                                                            grid_resolution=50*way,
                                                                            grid_range=(0, 1),
                                                                            n_samples=self.X_train.shape[0],
                                                                            with_variance=(True if way == 1 else False),
                                                                            figsize=fig_size, n_jobs=1, progressbar=False)


            fig, ax = pyplot.subplots(figsize=fig_size)


            pyplot.rc('font', **font)
            graph_data = []
            coords_data_x, coords_data_y = [], []
            coords_data_x_original = []

            for class_no in range(len(r[0]) // 2, len(r[0])):
                x = r[0][class_no].lines[0].get_xdata()
                y = r[0][class_no].lines[0].get_ydata()
                original_x = np.array(range(1, len(y) + 1))

                graph_data.append([x,y])
                coords_data_x.append(x)
                coords_data_x_original.append(original_x)
                coords_data_y.append(y)

            x_trans = np.transpose(np.array(coords_data_x))
            y_trans = np.transpose(np.array(coords_data_y))
            x = x_trans.mean(axis=1)
            y = y_trans.mean(axis=1)

            original_x = np.array(range(1, len(y) + 1))
            if len(r[0])//2 > 1:
                print('Smoothing the PDP Curve')
                window_len = (1 if len(x) == 2 else (len(x) // 2 if (len(x)//2) % 2 == 1 else len(x) // 2 + 1))
                polyorder = (window_len - 1 if window_len <= 3 else 3)
                try:
                    y = savgol_filter(y, window_len, polyorder)
                except:
                    y = y[~np.isnan(x)]
                    y = savgol_filter(y, window_len, polyorder)

            y = list(y)

            peaks, _ = find_peaks(y, height = 0)
            print('\nPeaks :\n')
            prom = peak_prominences(y, peaks)[0]

            wids, width_heights, left_ips, right_ips = peak_widths(y, peaks, rel_height=0.5)
            ploc = x[peaks]
            ploc_y = np.array(y)[peaks]
            print('Peaks:       ', list(ploc))
            print('Prominence:  ', list(prom))
            print('Widths:  ', wids)

            ax.plot(x, y)
            aoe = {}

            y_grad = np.gradient(y, x)

            if len(prom) > 0:
                avg_prom = sum(prom) / len(prom)
                avg_peak_point = (sum(ploc_y) + max(y)) / (len(ploc_y) + 1)

                for i, p in enumerate(ploc):
                    if prom[i] >= avg_prom and ploc_y[i] >= avg_peak_point:
                        trans_x = np.interp(left_ips[i], original_x, x)
                        trans_y = np.interp(right_ips[i], original_x, x)
                        aoe[p] = (trans_x, trans_y)
            m = max(y)

            prev_right = None
            x_range = max(x) - min(x)
            if len(ploc) == 0 or not aoe:
                already_taken = set()

                top_index = sorted(sorted(range(len(y_grad)), key=lambda i: y_grad[i])[-int(ceil((len(y) * top_n_percent) / 100)):][::-1])

                for ind_index, ind in enumerate(top_index):
                    left, right = get_left_right_index(ind, len(y_grad), already_taken)
                    if ind_index > 0 and prev_right is not None and x[left] - x[prev_right] < (x_range * gap_threshold) / 100:
                        if prev_right < left:
                            aoe[(x[prev_right] + x[left]) / 2] = (x[prev_right], x[left])

                    if left < right:
                        already_taken.add(left)
                        already_taken.add(ind)
                        already_taken.add(right)
                        aoe[x[ind]] = (x[left], x[right])
                        prev_right = right


            if not aoe:
                aoe[x[y.index(m)]] = (0.75 * x[y.index(m)] + 0.25 * min(x), x[y.index(m)])

            aoe_temp = {}
            prev_left, prev_right = None, None

            cont_flag = True
            for p in aoe:
                if prev_left is not None and prev_right is not None:
                    if prev_right >= aoe[p][0]:
                        prev_right = aoe[p][1]
                        cont_flag = True
                    else:
                        aoe_temp[(prev_left + prev_right) / 2] = (prev_left, prev_right)
                        prev_left, prev_right = aoe[p]
                        cont_flag = False
                else:
                    prev_left, prev_right = aoe[p]

            if cont_flag:
                aoe_temp[(prev_left + prev_right) / 2] = (prev_left, prev_right)
            aoe = aoe_temp

            print('Area of Effect:')
            for p in aoe:
                if aoe[p][1] - aoe[p][0] > (x_range * peak_width_threshold) / 100:
                    print(f'Peak/Slope at {p}: {aoe[p]}')

                    highlight_x, highlight_y = [], []
                    for pi, point in enumerate(x):
                        if aoe[p][0] <= point <= aoe[p][1]:
                            highlight_x.append(point)
                            highlight_y.append(y[pi])

                    highlight_y_ll = np.interp(aoe[p][0], x, y)
                    highlight_y_ul = np.interp(aoe[p][1], x, y)
                    if len(highlight_x) == 0:
                        highlight_x = [aoe[p][0], aoe[p][1]]
                        highlight_y = [highlight_y_ll, highlight_y_ul]
                    else:
                        highlight_x = [aoe[p][0]] + highlight_x + [aoe[p][1]]

                        highlight_y = [highlight_y_ll] + highlight_y + [highlight_y_ul]

                    ax.plot(highlight_x, highlight_y, color = 'red',linewidth=40, alpha = 0.5)

        else:
            fig, ax = pyplot.subplots(figsize=fig_size)
            value_keys, values = [], []
            for unique_value in unique_values:
                value_keys.append(unique_value)
                values.append(np.array(self.labels[self.custom_data[self.data_disp[feature_var_initial[0]] == unique_value].index]))

            positions = (1, 2)

            bp = ax.violinplot(values, showmeans=True, showmedians=True)
            bp['cmeans'].set(color='red',
                             linewidth=1)
            bp['cmedians'].set(color='green',
                               linewidth=1)
            pyplot.xticks(positions, value_keys)

        ax.set_title("PDP Plot - {} vs {}".format(self.response_var, "|".join(feature_var_initial)), **font)

        # adding legends
        categorical_count = []
        categorical_count.append('object' in str(self.data_disp[feature_var_initial[0]].dtype))
        categorical_count.append('object' in str(self.labels_disp.dtype))

        for ci in range(2):
            if ci == 0:
                if not categorical_count[0] or len(unique_values) <= 2:
                    continue
                test_data_disp = self.data_disp[feature_var_initial[0]]
                test_data = self.custom_data[feature_var_initial[0]]
                var_string = feature_var_initial[0]
            else:
                if not categorical_count[1]:
                    continue

                test_data_disp = self.labels_disp
                test_data = self.labels
                var_string = self.response_var

            encode_map = {}
            for i in range(test_data.shape[0]):
                if ci == 0:
                    enc_val = test_data.iloc[i]
                else:
                    enc_val = test_data[i]
                if enc_val not in encode_map:
                    encode_map[enc_val] = (test_data_disp.iloc[i] if ci == 0 else test_data_disp[i])
            encode_map = OrderedDict(sorted(encode_map.items()))

            enc_index = 0
            fake_legend = []
            enc_list = []
            encode_map_list = list(encode_map.keys())
            if len(encode_map_list) > max_legend:
                encode_map_list = random.choices(encode_map_list, k = max_legend)

            for enc_val in encode_map_list:
                fake_legend.append((Line2D([0], [0])))
                enc_list.append("{} : {}".format(round(enc_val, 2), encode_map[enc_val]))
                enc_index += 1
                if enc_index == max_legend:
                    break
            if ci == 0:
                ax.legend(fake_legend, enc_list, handlelength = 0, title = var_string, loc = 0,
                          fontsize = fontsize)
            else:
                leg = Legend(ax, fake_legend, enc_list,
                             loc = 'lower right', handlelength = 0, title = var_string, fontsize = fontsize)
                ax.add_artist(leg)

        pyplot.xlabel("_".join(feature_var_initial),fontsize = fontsize, **font)
        pyplot.ylabel(self.response_var,fontsize = fontsize, **font)
        pyplot.rc('xtick', labelsize=labelgap)
        pyplot.rc('ytick', labelsize=labelgap)
        pyplot.xticks(fontsize=fontsize, **font)
        pyplot.yticks(fontsize=fontsize, **font)
        pyplot.tight_layout()
        pyplot.savefig(os.path.join('PDP', self.response_var, "_".join(feature_var_initial) + '.png'))

    def explain_tree(self, top_n=3):
        """
        Parameter:
            top_n ( int ) : (Optional) (default = 3 ) Number of explanation statement to be displayed; based on top probability score.
        """
        if not self.formatting_pass:
            print('Data formatting error')
            return self.format_error
        os.makedirs("Decision Tree", exist_ok=True)
        if self.not_trained:
            self.train_test()

        X_train = self.X_train_cf
        y_train = self.y_train_cf

        # creating decision tree
        if 'regression' in self.problem_type:
            surrogate_explainer = tree.DecisionTreeRegressor(max_depth=3, random_state=42, splitter='best')
        else:
            surrogate_explainer = tree.DecisionTreeClassifier(max_depth=3, random_state=42,
                                                              splitter='best')

        surrogate_explainer.fit(X_train, y_train)
        print('base estimator:\n', surrogate_explainer)
        print()

        graph_string = export_graphviz(surrogate_explainer, feature_names=list(self.columns), max_depth=3, filled=True,
                                       out_file=None,
                                       node_ids=True,
                                       class_names=self.custom_class_names if 'classification' in self.problem_type else None)

        graph = Source(graph_string)

        svg_data = graph.pipe(format='svg')
        svg_filename = os.path.join("Decision Tree", 'dtree_structure_{}.svg'.format(self.response_var))
        with open(svg_filename, 'wb') as f:
            f.write(svg_data)

        html_explanation, self.filter_dict = ExplainTree(svg_filename, self.custom_data,
                                                self.data_disp, self.problem_type, self.frequency_map, top_n, total_samples=X_train.shape[0]).explain()

    def filter_dataframe(self, model_name, max_edges=200, bucket_size=10, include_labels=True):
        def normalize_list(x, arr):
            return (x - min(arr)) / (max(arr) - min(arr))

        if not self.formatting_pass:
            print('Data formatting error')
            return self.format_error
        selected_rows = None

        if not self.filter_dict:
            filtered_df = self.data_disp.copy()
        else:
            filter_cond = None
            for cat_col in self.filter_dict.keys():
                if filter_cond is None:
                    filter_cond = [val in self.filter_dict[cat_col] for val in self.data_disp[cat_col]]
                else:
                    filter_cond = filter_cond or [val in self.filter_dict[cat_col] for val in self.data_disp[cat_col]]

            selected_rows = list(set(i for i, x in enumerate(filter_cond) if x is True))

            filtered_df = self.data_disp[filter_cond]

        if include_labels:
            filtered_df[self.response_var] = np.take(self.labels_disp, selected_rows) if selected_rows is not None else self.labels_disp.copy()

        filtered_df = filtered_df.round(decimals = 2)
        os.makedirs("Filtered_data", exist_ok=True)

        filtered_df.to_csv(os.path.join("Filtered_data", "dataGraph.csv"),index = False)

        df_dict = self.summarize_filtered_dataframe(filtered_df, bucket_size)
        
        filtered_columns = list(df_dict.keys())
        edgelist_dict = {}
        edgelist = []
        nodelist = []

        jsonPath = os.path.join(os.path.dirname(os.getcwd()),'json', model_name + '.json')
        with open(jsonPath) as f:
            content = json.load(f)

        nodes = content['nodes']
        relation_list = []
        entity_dict = {}

        for node in nodes:
            for cols in node['features']:
                entity_dict[cols] = node['id']

        for i in range(len(filtered_columns)):
            col1 = filtered_columns[i]
            checklist = df_dict[col1]['Value'].tolist()
            samplesize = df_dict[col1]['Sample Size'].tolist()
            if 'object' in str(filtered_df[col1].dtype):
                for j, item in enumerate(checklist):
                    item = str(item)
                    nodelist.append([col1 + '-' + item, col1 + '-' + item, {'value': item, 'sample_size': samplesize[j]},
                                    entity_dict[col1], 1])

            else:
                for j, item in enumerate(checklist):
                    item = str(item)
                    nodelist.append([col1 + '-' + item, col1 + '-' + item,
                                    {'range': item, 'sample_size': samplesize[j]},
                                    entity_dict[col1], 1])

        nodelist_df = pd.DataFrame(nodelist, columns = ['id', 'name', 'values', 'entity', 'node_Aff'])

        for links in content['links']:
            source, target, relation_type = links['source'], links['target'], links['type']
            sourcenode = next(item for item in nodes if item["id"] == source)
            sourcefeatures = sourcenode['features']

            targetnode = next(item for item in nodes if item["id"] == target)
            targetfeatures = targetnode['features']
            relation_list.append([sourcefeatures, targetfeatures, relation_type])

        for i in range(len(filtered_columns) - 1):
            for j in range(i+1, len(filtered_columns)):
                col1 = filtered_columns[i]
                col2 = filtered_columns[j]

                is_related, relation_type = False, ''

                for unionlist in relation_list:
                    if (col1 in unionlist[0] and col2 in unionlist[1]) or (col2 in unionlist[0] and col1 in unionlist[1]):
                        is_related = True
                        relation_type = unionlist[2]

                if not is_related and col1 != self.response_var and col2 != self.response_var:
                    continue

                checklist = df_dict[col1]['Value'].tolist()
                checklist2 = df_dict[col2]['Value'].tolist()

                for k in range(filtered_df.shape[0]):
                    source = str(filtered_df[col1].iloc[k])
                    target = str(filtered_df[col2].iloc[k])
                    source_val = None
                    target_val = None

                    if 'object' in str(filtered_df[col1].dtype):
                        if source in checklist:
                            source_val = col1 + '-' + source
                    else:
                        for item in checklist:
                            item = str(item)
                            item_temp = item.split(' - ')
                            if len(item_temp) == 2:
                                minval, maxval = item_temp
                            else:
                                minval = item_temp[0]
                                maxval = minval

                            minval, maxval = float(minval), float(maxval)

                            if minval <= float(source) <= maxval:
                                source_val = col1 + '-' + item
                                break

                    if source_val is not None:
                        if 'object' in str(filtered_df[col2].dtype):
                            if target in checklist2:
                                target_val = col2 + '-' + target
                        else:
                            for item in checklist2:
                                item = str(item)
                                item_temp = item.split(' - ')
                                if len(item_temp) == 2:
                                    minval, maxval = item_temp
                                else:
                                    minval = item_temp[0]
                                    maxval = minval

                                minval, maxval = float(minval), float(maxval)

                                if minval <= float(target) <= maxval:
                                    target_val = col2 + '-' + item
                                    break

                    if source_val is not None and target_val is not None:
                        if (source_val, target_val) not in edgelist_dict:
                            edgelist_dict[(source_val, target_val)] = [1, relation_type]
                        else:
                            edgelist_dict[(source_val, target_val)][0] += 1


        if edgelist_dict:
            weight_list = [edgelist_dict[item][0] for item in edgelist_dict]
            for item in edgelist_dict:
                edge_weight = round(normalize_list(edgelist_dict[item][0], weight_list), 2)
                if edge_weight != 0:
                    edgelist.append([item[0], item[1], edgelist_dict[item][1], edge_weight])

        edgelist_df = pd.DataFrame(edgelist, columns = ['source', 'target', 'type', 'weight'])

        edgelist_df = edgelist_df.sort_values(by=['weight'], ascending=False)
        if edgelist_df.shape[0] > max_edges:

            edgelist_df = edgelist_df.head(max_edges)

            final_nodelist = set(set().union(edgelist_df['source'].tolist(), edgelist_df['target'].tolist()))
            
            t = [val in final_nodelist for val in nodelist_df['id']]
            nodelist_df = nodelist_df[t]

        result = nodelist_df.to_json(orient="records")
        parsed = json.loads(result)
        final_json = {'nodes': parsed}

        result = edgelist_df.to_json(orient="records")
        parsed = json.loads(result)
        final_json['links'] = parsed

        with open(os.path.join('Filtered_data', 'dataGraph.json'.format(model_name, self.response_var)), 'w+') as fp:
            json.dump(final_json, fp)

        with pd.ExcelWriter(os.path.join('Filtered_data', '{}_{}_nodes_and_edges.xlsx'.format(model_name, self.response_var))) as writer:
            nodelist_df.to_excel(writer, sheet_name='Node List', index=False)
            edgelist_df.to_excel(writer, sheet_name='Edge List', index=False)

        return nodelist_df, edgelist_df, filtered_df
    
    def summarize_filtered_dataframe(self, filtered_df, bucket_size=10):
        columns = filtered_df.columns
        df_dict = {}
        df_size = filtered_df.shape[0]
        bucket_limit = ceil(df_size / bucket_size)
        print('Bucket Limit :', bucket_limit)
        for col in columns:
            df_list = []
            if 'object' in str(filtered_df[col].dtype):
                current_column_data = filtered_df[col].tolist()
                freq = Counter(current_column_data)

                sorted_x = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[ : bucket_size]
                ordered_data = OrderedDict(sorted_x)
                for item in ordered_data:
                    df_list.append([str(item), ordered_data[item]])

            else:
                lastInserted = float('nan')
                current_column_data = sorted(filtered_df[col].tolist())
                current_bucket = set()
                insertion_count = 0
                for item in current_column_data:

                    if lastInserted != item:
                        lastInserted = item

                        if insertion_count + 1 > bucket_limit:
                            minval, maxval = min(current_bucket), max(current_bucket)
                            if minval != maxval:
                                inserted_range = str(minval) + ' - ' + str(maxval)
                            else:
                                inserted_range = str(minval)

                            df_list.append([inserted_range, insertion_count])
                            insertion_count = 0
                            current_bucket = set()

                    current_bucket.add(item)
                    insertion_count += 1

                if len(current_bucket) > 0:
                    minval, maxval = min(current_bucket), max(current_bucket)
                    if minval != maxval:
                        inserted_range = str(minval) + ' - ' + str(maxval)
                    else:
                        inserted_range = str(minval)

                    df_list.append([inserted_range, insertion_count])

            df_dict[col] = pd.DataFrame(df_list, columns=['Value', 'Sample Size'])

        return df_dict

    def detect_outlier(self, samples=None, testdata_only=False, crunch_factor=1):
        """
        Parameter:
            samples ( int ) : (Optional) (default = None ) Number of samples to be considered to check for outliers.
                                By default whole data will be considered.

            testdata_only ( boolean ) : (Optional) (default = False ) Only test data will be considered if it is true.
        """
        os.makedirs("Outliers", exist_ok=True)
        filtered_data = self.custom_data.copy()
        filtered_data_enc = self.data_disp.copy()
        drop_col_list = []
        for cols in self.columns:
            if filtered_data_enc[cols].dtype == 'object' or 'category' in str(filtered_data_enc[cols].dtype):
                drop_col_list.append(cols)

        if len(filtered_data.columns)==0:
            print('No numerical columns present in dataset')
            return

        X_train, X_test, y_train, y_test = train_test_split(filtered_data, self.labels, test_size=0.3,
                                                            random_state=42)
        clf = KNN()
        clf.fit(X_train)

        # get the prediction labels and outlier scores of the training data
        y_train_scores = clf.decision_scores_  # raw outlier scores

        # get the prediction on the test data
        y_test_scores = clf.decision_function(X_test)  # outlier scores

        if testdata_only and not samples:
            y_scores = np.array(list(y_test_scores))
            SD = np.std(y_scores)
            mean = np.mean(y_scores)
            ul, ll = mean + crunch_factor*SD, mean - crunch_factor*SD
            original_indices = (y_scores < ll) | (y_scores > ul)
            x_original = np.array(range(1, len(y_scores) + 1))
            y_scores = np.array(list(y_test_scores))
        else:
            y_scores = np.array(list(y_train_scores) + list(y_test_scores))
            if samples:
                SD = np.std(y_scores)
                mean = np.mean(y_scores)
                ul, ll = mean + crunch_factor*SD, mean - crunch_factor*SD
                original_indices = (y_scores < ll) | (y_scores > ul)
                x_original = np.array(range(1, len(y_scores) + 1))
                y_scores = y_scores[:samples]

        SD = np.std(y_scores)
        mean = np.mean(y_scores)
        ul, ll = mean + crunch_factor*SD, mean - crunch_factor*SD
        if not samples:
            original_indices = (y_scores < ll) | (y_scores > ul)
            x_original = np.array(range(1, len(y_scores) + 1))

        indices = (y_scores < ll) | (y_scores > ul)

        fig, ax = pyplot.subplots(figsize=(16, 9))
        x = np.array(range(1, len(y_scores) + 1))
        ax.plot(x, [ul] * len(y_scores))
        ax.plot(x, [mean] * len(y_scores), 'g--')
        ax.plot(x, [ll] * len(y_scores))

        ax.scatter(x[~indices], y_scores[~indices], color='b')
        ax.scatter(x[indices], y_scores[indices], color='r')

        ax.fill_between(x, [ul] * len(y_scores), [ll] * len(y_scores), alpha=0.10)

        ax.legend(['Upper limits', 'Mean', 'Lower limits', 'Inlier', 'Outlier'])
        ax.set_title("Scores")
        filename = os.path.join("Outliers",'Scores_{}.png'.format(self.response_var))
        if os.path.exists(filename):
            os.remove(filename)

        pyplot.savefig(filename)
        if len(x_original[original_indices])>0:
            print("Below rows are outlier:\n\t")
            print(list(x_original[original_indices]))
        else:
            print("No rows are outlier")

    def detect_anomaly(self, time_col, series_col, crunch_factor=1):
        data = self.data_disp.copy()
        if 'date' not in str(data[time_col].dtype):
            if 'object' in str(data[time_col].dtype) or 'str' in str(data[time_col].dtype) or 'category' in str(data[time_col].dtype):
                try:
                    data[time_col] = pd.to_datetime(data[time_col])
                except Exception as e:
                    print(e)
                    print('Cannot convert column : {} to datetime'.format(time_col))
                    return
            else:
                print('Cannot convert column : {} to datetime'.format(time_col))
                return
        # fraction of data to train forecaster with
        train_fraction = 1.0
        # factor of error margin over confidence interval (ucl/lcl +/- (crunch_factor * stderr))

        series = data[[time_col, series_col]]
        d = 1
        while True:
            if d:
                dS = series[series_col].diff(d)
            else:
                dS = series[series_col]
            acf_val = acf(dS.values)
            pacf_val = pacf(dS.values)

            if acf_val[10] > 0 and pacf_val[10] > 0:
                d += 1
            else:
                break

        order = (acf_val.argmax(), pacf_val.argmax(), d)
        print('ORDER:', order)
        X = series[series_col].values
        size = int(len(X) * train_fraction)
        train, test = X[0:size], X
        history = list(train)

        # predictions = []
        ucl = []
        lcl = []
        std_err = []

        for t in range(len(test)):
            model = ARIMA(history, order=order)
            model_fit = model.fit(disp=0)
            output, stderr, conf = model_fit.forecast()
            lcl.append(conf[0][0])
            ucl.append(conf[0][1])
            std_err.append(stderr)
            obs = test[t]
            history.append(obs)

        anomaly = []
        for i in range(len(test)):
            if test[i] > (ucl[i] + crunch_factor * std_err[i]) or test[i] < (lcl[i] - crunch_factor * std_err[i]):
                anomaly.append(test[i])
            else:
                anomaly.append(np.NaN)

        x = np.arange(0, len(test))

        fig, ax = pyplot.subplots(figsize=(16, 9))

        ax.plot(test, marker='.', linestyle='', color='blue')
        ax.plot(anomaly, marker='.', linestyle='', color='red')
        ax.legend(['Inlier', 'Anomaly'])
        ax.plot(ucl, linestyle='', color='silver')
        ax.plot(lcl, linestyle='', color='silver')
        ax.fill_between(x, ucl, lcl, color='silver')
        ax.set_title("Anomaly")

        os.makedirs("Anomaly", exist_ok=True)
        pyplot.savefig(os.path.join("Anomaly", "anomaly_{}_{}.png".format(time_col, series_col)))
