from flask import *
from flask_wtf import FlaskForm
from wtforms import *
from wtforms.validators import *
from flask_wtf.file import FileField, FileRequired
from werkzeug.utils import secure_filename
import mimetypes
from flask_cors import CORS
import os
import sys
import pandas as pd
import json
import re
import shutil
from zipfile import ZipFile
from collections import OrderedDict 
import requests
from random import randint
##Added-12-06-2020
import dimensions_measures_finder as dmf
##Added-25-06-2020
from XAI import Explainer
import base64

mimetypes.add_type('text/css', '.css')
mimetypes.add_type('text/javascript','.js')
basedir = os.path.abspath(os.path.dirname(__file__))
jsonDirectory = os.path.join(basedir,'json')
jsonSelectDirectory = os.path.join(basedir,'selectjson')
Form_Directory = os.path.join(basedir,'Form')
Data_Directory = os.path.join(basedir,'data')
Data_Dump_Directory = os.path.join(basedir,'Full_Data_Dumps')
UPLOAD_FOLDER = os.path.join(basedir,'upload_folder')
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'zip','xlsx','xls'])

serverName = sys.argv[1]
portDetails = sys.argv[2]
Url = "http://"+serverName+":"+portDetails+"/"
OpFileURL = Url+"xaifiles/"

if os.name == 'nt':
    replaceString = "\\"
else:
    replaceString = "/"


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def group_list(lst): 
      
    res =  [(el, lst.count(el)) for el in lst] 
    return list(OrderedDict(res).items()) 

def Load_Global_Models(Model_json_Directory,Model_Data_Directory):
    try:
        filejsontoLoad = [f for f in os.listdir(Model_json_Directory)]
        fileDatatoLoad = [f for f in os.listdir(Model_Data_Directory)]
        models = {}
        modeldata = {}
        for eachModel in filejsontoLoad:
            with open(Model_json_Directory+'/'+eachModel) as f:
                mod = json.load(f)
                modelName = eachModel.split(".")[0]
                models[modelName] = mod
            
    
        for eachModelData in fileDatatoLoad:
            fileDf = pd.read_csv(Model_Data_Directory+'/'+eachModelData)
            # fileDf = fileDf.dropna()
            modelName = eachModelData.split(".")[0]
            modeldata[modelName] = fileDf
        
    
        finRayModels = {'models':models,'modeldata':modeldata}
    
        return (True,finRayModels)
    except:
        print("There is some error in the Global model creation")
        return (False,{})

## Global Containers

global finRayModels
finRayModels = {}
Global_Data_Graph_requests = {}
Global_XaiOp = {}

## Listing the files and directory 

files_and_dir = os.listdir()
Req_dirs = list()
Valid_Req_dirs = list()
for items in files_and_dir:
    if (re.search("Req",items) != None):
        Req_dirs.append(items)

## Loading the Cache File for XAI Requests
cache_request = {}
cache_keys = list()
with open(os.path.join(basedir,"cache_request.json")) as f:
    cache_request = json.load(f)
    cache_keys = list(cache_request.keys())
    for eachReq in cache_request.keys():
        Valid_Req_dirs.append(cache_request[eachReq])

### Deleting non required caches
for dirs in Req_dirs:
    if dirs not in Valid_Req_dirs:
        try:
            shutil.rmtree(dirs)
            print("Deleted ",dirs)
        except:
            print("Unable to Delete ",dirs)
            continue

with open(os.path.join(basedir,"xai_output_cache.json")) as fc:
    Global_XaiOp = json.load(fc)

## Load Gobal Models 
(retmess,finRayModels) = Load_Global_Models(jsonDirectory,Data_Dump_Directory)

if retmess == True:
    selectJson = list()
    for eachModelOption in list(finRayModels['models'].keys()):
        option = {"name": eachModelOption, "checked" : False}
        selectJson.append(option)
    
    json.dump(selectJson,open(jsonSelectDirectory+"/selection.json","w"))


class FileUpload():
    
    prevzipfile = ""
    prevrelationFile = ""
    
    def __init__(self,zipfile,relationFile):
        self.zipfile = zipfile
        self.relationFile = relationFile
        
    def assign_filenames(self):
        FileUpload.prevzipfile = self.zipfile
        FileUpload.prevrelationFile = self.relationFile
        
    def get_File_Details(FileUpload):
        return (FileUpload.prevzipfile,FileUpload.prevrelationFile)




app = Flask(__name__)
cors = CORS(app)
app.config['SECRET_KEY'] = 'HubbaDeveloper'

class Finray():

    def __init__(self,modelName,relationFile,outputModelJson):
        self.modelName = modelName
        self.relationFile = relationFile
        self.outputModelJson = outputModelJson


    def generateInitialModel(self):
        Form_data = pd.read_excel(Form_Directory+"/"+self.relationFile)
        Form_data = Form_data.sort_values(by=['Table1','Table2'])
        File_to_Load = [f for f in os.listdir(Data_Directory+"/"+self.modelName)]
        Data_Frames =[f.split(".")[0] for f in os.listdir(Data_Directory+"/"+self.modelName)]
        All_Data = {}
        for eachFile in File_to_Load:
            df = pd.read_csv(Data_Directory+"/"+self.modelName+"/"+eachFile)
            df_name = eachFile.split(".")[0]
            All_Data[df_name] = df
        
        Pivotal_Points = {}
        for tabs in Data_Frames:
            Pivotal_Points[tabs] = 0

        Merged_df = pd.DataFrame()
        tablesAlreadyConsidered = list()
        for j in range(len(Form_data)):
            firstTab = Form_data.iloc[j,0]
            secondTab = Form_data.iloc[j,1]
            firstTabCond = Form_data.iloc[j,2]
            secondTabCond = Form_data.iloc[j,3]
            Pivotal_Points[firstTab] =  Pivotal_Points[firstTab]+1
            Pivotal_Points[secondTab] =  Pivotal_Points[secondTab]+1
            if Merged_df.empty == True:
                Merged_df = pd.merge(All_Data[firstTab],All_Data[secondTab],how="outer",left_on=firstTabCond, right_on=secondTabCond,suffixes=('_'+firstTab, '_'+secondTab))
                tablesAlreadyConsidered.append(firstTab)
                tablesAlreadyConsidered.append(secondTab)
            else:
                if secondTab not in tablesAlreadyConsidered:
                    Merged_df = pd.merge(Merged_df,All_Data[secondTab],how="outer",left_on=firstTabCond, right_on=secondTabCond,suffixes=('', '_'+secondTab))
                else:
                    Merged_df = pd.merge(Merged_df,All_Data[firstTab],how="outer",left_on=firstTabCond, right_on=secondTabCond,suffixes=('', '_'+firstTab))
            
        Merged_df.to_csv(Data_Dump_Directory+'/'+self.modelName+'.csv',index=False)
        Form_data.to_csv(Data_Dump_Directory+'/'+self.modelName+'_relation.csv',index=False)
        
        with open(jsonSelectDirectory+'/selection.json') as f:
            modelOptions = json.load(f)
            option = {"name": self.modelName, "checked" : False}
            modelOptions.append(option)
            json.dump(modelOptions,open(jsonSelectDirectory+"/selection.json","w"))

        nodes = list()
        for eachItem in Data_Frames:
            nodes.append({"id":eachItem,"name":eachItem,"node_Aff":Pivotal_Points[eachItem],"features":list(All_Data[eachItem].columns)})
        
        edges=list()
        for j in range(len(Form_data)):
            firstTab = Form_data.iloc[j,0]
            secondTab = Form_data.iloc[j,1]
            relation = Form_data.iloc[j,4]
            edges.append({"source":firstTab,"target":secondTab,"type":relation})
        
        graph={"nodes":nodes,"links":edges}
        json.dump(graph,open(jsonDirectory+"/"+self.outputModelJson,"w"))
    
        return 'MC'

class SelectModel():

    def __init__(self,modelName):
        self.modelName = modelName
        self.modelRelation = pd.read_csv(Data_Dump_Directory+'/'+self.modelName+'_relation.csv')
        self.old_query = None

    def set_old_query(self,query):
        self.old_query = query

    def get_old_query(self):
        return self.old_query 
        
    def filter_selection(self,query):
        #Columns name from Query: sample: ["products-UnitPrice", "products-UnitsInStock", "categories-Picture", "products-SupplierID"]
        #Get Model Data
        data = pd.read_csv(os.path.join(basedir,'Full_Data_Dumps',self.modelName+'.csv'))
        # data=finRayModels['modeldata'][self.modelName]
        orgCols = list(data.columns)
        filtered_Cols=[]
        for item in query:
            table,column=item.split('-')
            #Go Specific to General
            if(column+'_'+table in orgCols):
                filtered_Cols.append(column+'_'+table)
            else:
                filtered_Cols.append(column)
           
        #Get Subset of Data
        data = data[filtered_Cols].drop_duplicates().dropna()
        #Feature selection code
        dim_mes=dmf.get_dim_mes_keyColumns(data)
        #print(dim_mes)
        #Json object of list of measures and dim
        dim_mes_json=json.dumps({"dimensions":dim_mes["Dimensions"],"measures":dim_mes["Measures"]})
        #Return json object
        #
        return dim_mes_json
    
    def getXAIOutput(self,request_no,features,response_var,get_feature_scores=True,plot_pdp=True,pdp_cols=None,tree_explanation=True,detect_outlier=True,detect_anomaly=False,anomaly_time_col=None,anomaly_series_col=None):
        #Get Model Data
        data = pd.read_csv(os.path.join(basedir,'Full_Data_Dumps',self.modelName+'.csv'))
        # data=finRayModels['modeldata'][self.modelName]
        
        #Get current working directory/ specify processing folder here to store temp files.
        os.chdir(basedir)
        os.makedirs(request_no, exist_ok=True)
        cwd=os.chdir(basedir+'/'+request_no)
        
        #current dataset
        all_col=list(features.keys())
        all_col.append(list(response_var.keys())[0])
        print(all_col)
        dataset=data[all_col]
        # print(dataset)
        # print(response_var)
        # print(features)
        resp_var = list(response_var.keys())[0]
        features_saved_for_pdp = list(features.keys())
        features[resp_var] = response_var[resp_var]
        format_dtypes = features
        print(response_var)
        print(features)
        features = features_saved_for_pdp
        #explainer object
        #  We need to call explainer with new feature variable and response variable now 
        # 
        explainer=Explainer(data=dataset,response_var=resp_var,format_datatypes=format_dtypes)
        xai_outputs={}
        if get_feature_scores:
            ret_fet_scores=explainer.get_feature_scores(normalize=True)
            ret_fet_scores.to_csv(basedir+"/"+request_no+"/feature_scores.csv",index=False)
            # for eachCol in ret_fet_scores.columns:
            #     ret_fet_scores[eachCol] = ret_fet_scores[eachCol].astype('str')
             
            xai_outputs['feature_scores'] = OpFileURL+request_no+"%7Cfeature_scores.csv"
        
        if plot_pdp:
            
            pdp_paths=list()
            if pdp_cols is None:#none value implies all cols
                feature_Space = features
                features = features[:len(features)]
                print(features)
                for eachcol in features:
                    print(eachcol)
                    explainer.pdp(eachcol)
                    print(cwd)
                    print(eachcol)
                    path=os.path.join(request_no,'PDP',resp_var,"{}.png".format(eachcol))
                    path = path.replace(replaceString,"%7C")
                    path = OpFileURL+path
                    pdp_paths.append(path)
                         
            else:
                if type(pdp_cols) ==str:
                    pdp_cols=[pdp_cols]
                for eachcol in pdp_cols:
                    explainer.pdp(eachcol)
                    path=os.path.join(request_no,'PDP',resp_var,"{}.png".format(eachcol))
                    path = path.replace(replaceString,"%7C")
                    path = OpFileURL+path
                    pdp_paths.append(path)
                    
            features = feature_Space
            pdp_paths.append(OpFileURL+os.path.join(request_no,'Feature Importance',"FeatureImportance_XGB_{}.png".format(resp_var)).replace(replaceString,"%7C"))
            xai_outputs['pdp_img_paths']=pdp_paths
            
        if tree_explanation:
            explanations=explainer.explain_tree()
            xai_outputs['tree_explanation']={'explanations':tree_explanation,'explanations_statement':OpFileURL+os.path.join(request_no,'Statements','Statements.html').replace(replaceString,"%7C"),'tree_path':OpFileURL+os.path.join(request_no,'Decision Tree',f'dtree_structure_{resp_var}.svg').replace(replaceString,"%7C")}
            
        if detect_outlier:
            outliers=explainer.detect_outlier()
            if outliers != None:
                outliers = [str(item) for item in outliers]
            else:
                outliers = 'None'
            
            path=os.path.join(request_no,"Outliers",f'Scores_{resp_var}.png')
            path = path.replace(replaceString,"%7C")
            xai_outputs['detected_outliers']={'row_outliers':outliers,'image_path':OpFileURL+path}
            
        if detect_anomaly and anomaly_time_col is not None and anomaly_series_col is not None:
            anomaly=explainer.detect_anomaly(time_col=anomaly_time_col, series_col=anomaly_series_col)
            path=os.path.join(request_no,"Anomaly", f"anomaly_{anomaly_time_col}_{anomaly_series_col}.png")
            path = path.replace(replaceString,"%7C")
            xai_outputs['detected_anomalies']={'image_path':OpFileURL+path}

        #Json object of list of measures and dim

        ## Calling the Data graph and csv file creation module

        explainer.filter_dataframe(model_name = self.modelName)

        # Setting the datagraph path

        xai_outputs['Datagraph_path'] = OpFileURL+request_no+"%7CFiltered_data%7CdataGraph.json"
        xai_outputs['Datagraph_data'] = OpFileURL+request_no+"%7CFiltered_data%7CdataGraph.csv"
        xai_outputs['Global_Perf_Metrices'] = OpFileURL+request_no+"%7CPerformance Metrics%7Cperformance_global.txt"
        xai_outputs['Surrogate_Perf_Metrices'] = OpFileURL+request_no+"%7CPerformance Metrics%7Cperformance_surrogate.txt"

        print(xai_outputs)
        xai_outputs_json=json.dumps(xai_outputs)

        # Getting the filetered data from XAI
        # xai_outputs['Filtered_data'] = explainer.filter_dataframe()

        #Return json object
        return (str(xai_outputs_json),xai_outputs)
    
    def getDataGraph(self,query,RequestNo,mesName,filteredDataforGraph,feature_variables,response_var):
        # query = [{"Customers":["Region"]},{"Orders":["ShipName"]}]

        # New module

        

        response_keys = list(response_var.keys())
        feature_keys = list(feature_variables.keys())
        allkeys = feature_keys

        print(response_keys)
        print(allkeys)
        print(feature_keys)

        categories = list()
        
        for eachkey in feature_keys:
            if feature_variables[eachkey] == 'category':
                categories.append(eachkey)

        print(response_keys)
        print(allkeys)
        # print(categories)

        
        # filteredDataforGraph=filteredDataforGraph[allkeys].groupby(categories).sum().reset_index(level=categories).sort_values(response_keys,ascending=False)



        ##

        os.chdir(basedir)

        os.makedirs(RequestNo, exist_ok=True)
        # cwd=os.chdir(os.getcwd()+'/'+request_no)

        colls = query
        colls = [elem.split("-")[1]+"_"+elem.split("-")[0] for elem in colls]

        keys = list()
        for eachItem in query:
            keys.append(eachItem.split("-")[0])
    
        keys = list(set(keys))

        qry = {}

        for eachKey in keys:
            qry[eachKey]=[]
    
        for eachItem in query:
            entity =  eachItem.split("-")[0]
            features =  eachItem.split("-")[1]
            qry[entity] = qry[entity]+[features]
     
        entities = list(qry.keys())
        query = list()
        for i in range(len(entities)):
            query.append({entities[i]:qry[entities[i]]})

        # Form_data_Rel = pd.read_csv(os.path.join(basedir,'Full_Data_Dumps',self.modelName+'_relation.csv'))
        Form_data_Rel = self.modelRelation        
        features_selected = []
        for eachItem in query:
            features_selected = features_selected + list(eachItem.values())[0] 

        # print(features_selected)
        features_selected = list(set(features_selected))

        orgCols = list(finRayModels['modeldata'][self.modelName].columns)
        colls = [elem for elem in colls if elem in orgCols]
        colls2 = [elem.split("_")[0] for elem in colls]
        features_selected = [elem for elem in features_selected if elem in orgCols and elem not in colls2 ]
        features_selected = features_selected+colls

        data = finRayModels['modeldata'][self.modelName][features_selected].drop_duplicates().dropna()
        fullData = filteredDataforGraph.drop_duplicates()
        data = filteredDataforGraph
        # data = data.nlargest(10, mesName)
        filtered_Cols = list(data.columns)

        nodes = list()
        for eachEntity in query:
            entityName = list(eachEntity.keys())[0]
            featureSelected = eachEntity[entityName]
            for i in range(len(featureSelected)):
                if featureSelected[i] in filtered_Cols:
                    featureSelected[i] = featureSelected[i]
                else:
                    featureSelected[i] = featureSelected[i]+"_"+entityName

            filtered_Data =  data[featureSelected]
            filtered_Data = filtered_Data.dropna()
            
            filtered_Data_cols = list(filtered_Data.columns)

            for i in range(len(filtered_Data)):
                evrrow = list(filtered_Data.iloc[i,:])
                evrrow = [str(elem) for elem in evrrow]
                evrrow_val = [str(elem)+':'+str(belem) for (elem,belem) in zip(filtered_Data_cols,evrrow)]
                nodes.append({"id":"-".join(evrrow).replace("'",","),"name":evrrow[0].replace("'",","),"values":"-".join(evrrow_val).replace("'",","),"entity":entityName})

        for i in range(len(nodes)):
            nodes[i] = str(nodes[i])

        nodes = group_list(nodes)

        for i in range(len(nodes)):
            (Data,Affinity) = nodes[i]
            Data = Data.replace("'", '"')
        #     print(Data)
            Data = json.loads(Data)
            Data['node_Aff'] = Affinity
            nodes[i]=Data

        links = list()
        for i in range(len(query)):
            if i != len(query)-1:
                entity1 = query[i]
                entity2 = query[i+1]
                entity_one_Name = list(entity1.keys())[0]
                entity_two_Name = list(entity2.keys())[0]
                # entity_one_Name = list(entity1.keys())[0].lower()
                # entity_two_Name = list(entity2.keys())[0].lower()
                if Form_data_Rel[((Form_data_Rel['Table1'] == entity_one_Name) & (Form_data_Rel['Table2'] == entity_two_Name))].empty == True:
                    direction = "reverse"
                else:
                    direction = "straight"
                
                print(Form_data_Rel)
                print(entity_one_Name)
                print(entity_two_Name)
                try:
                    relation = list(dict(Form_data_Rel[((Form_data_Rel['Table1'] == entity_one_Name) & (Form_data_Rel['Table2'] == entity_two_Name)) | ((Form_data_Rel['Table1'] == entity_two_Name) & (Form_data_Rel['Table2'] == entity_one_Name))])['Relation'])[0]
                except:
                    relation = "Related to"
                        
                for j in range(len(data)):
                    entity_one_df = list(data.iloc[j,:][list(entity1.values())[0]])
                    entity_two_df = list(data.iloc[j,:][list(entity2.values())[0]])
                    entity_one_df = [str(elem) for elem in entity_one_df]
                    entity_two_df = [str(elem) for elem in entity_two_df]
                    if direction == "straight":
                        links.append({"source":"-".join(entity_one_df).replace("'",","),"target":"-".join(entity_two_df).replace("'",","),"type":relation})
                    else:
                        links.append({"target":"-".join(entity_one_df).replace("'",","),"source":"-".join(entity_two_df).replace("'",","),"type":relation})

        for k in range(len(links)):
            links[k]=str(links[k])

        links = list(set(links))

        for k in range(len(links)):
            links[k] = json.loads(links[k].replace("'", '"'))                     

        dataGraph = {"nodes":nodes,"links":links}
        json.dump(dataGraph,open(basedir+"/"+RequestNo+"/dataGraph.json","w"))
        fullData.to_csv(basedir+"/"+RequestNo+"/dataGraph.csv",index=False)

        return True   

@app.route('/modeljson/<filename>')
def getmodeljsonfile(filename):
    return send_file(jsonDirectory+'/'+filename,attachment_filename=filename)


@app.route('/xaifiles/<File_with_REQ_NO>')
def getXAIFiles(File_with_REQ_NO):
    print(File_with_REQ_NO)
    FileArr = File_with_REQ_NO.split("|")
    fileName = FileArr[len(FileArr)-1]
    print(fileName)
    filepath = "/".join(FileArr)
    return send_file(basedir+'/'+filepath,attachment_filename=fileName)
    

@app.route('/modeljsonselect')
def getselectjsonfile():
    # return send_file(jsonSelectDirectory+'/'+filename,attachment_filename=filename)
    with open(jsonSelectDirectory+'/selection.json') as f:
            modelOptions = json.load(f)
            modelOptions = str(modelOptions).replace("'",'"').replace("False","false")
            print(modelOptions)
            return modelOptions


@app.route('/getDimMesDetails',methods=['POST'])
def getDimMesDetails():
    print(eval(request.get_data()))
    req = eval(request.get_data())
    modelSelected = req['modelName']
    query = req['query']
    sm = SelectModel(modelSelected)
    sm.set_old_query(query)
    dm = sm.filter_selection(query)
    req_no_gen = 'Req_'+str(randint(100000000, 999999999))
    Global_Data_Graph_requests[req_no_gen]=sm
    return str(json.dumps({"req_no":req_no_gen,"Resp":dm}))


@app.route('/getXAIDetails',methods=['POST'])
def getXAIDetails():
    print(eval(request.get_data()))
    req = eval(request.get_data())
    Req_No = req['RequestNo']
    Selection = req['Selection']
    selection_encoded = str(Selection).encode("ascii") 
    selection_encoded = str(base64.b64encode(selection_encoded))
    if selection_encoded in cache_keys:
        if req['cache_required'] == 'true':
            print("Cache found and firing from cache")
            Req_No = cache_request[selection_encoded]
            return str(json.dumps({"req_no":Req_No,"Resp":"True"}))
        else:
            print("Cache found, but refreshing cache")
            Req_tobe_deleted = cache_request[selection_encoded]
            del cache_request[selection_encoded]
            del Global_XaiOp[Req_tobe_deleted]
            sm = Global_Data_Graph_requests[Req_No]
        
            (xaiop,xaiop_with_data) = sm.getXAIOutput(Req_No,Selection['feature_var'],Selection['response_var'])
        
            Global_XaiOp[Req_No] = xaiop
            cache_keys.append(selection_encoded)
            cache_request[selection_encoded] = Req_No
            json.dump(cache_request,open(basedir+"/cache_request.json","w"))
            json.dump(Global_XaiOp,open(basedir+"/xai_output_cache.json","w"))
            return str(json.dumps({"req_no":Req_No,"Resp":"True"}))    

    else:
        print("Cache not found, so genering explaination")
        sm = Global_Data_Graph_requests[Req_No]
        
        (xaiop,xaiop_with_data) = sm.getXAIOutput(Req_No,Selection['feature_var'],Selection['response_var'])
        
        Global_XaiOp[Req_No] = xaiop
        cache_keys.append(selection_encoded)
        cache_request[selection_encoded] = Req_No
        json.dump(cache_request,open(basedir+"/cache_request.json","w"))
        json.dump(Global_XaiOp,open(basedir+"/xai_output_cache.json","w"))
        return str(json.dumps({"req_no":Req_No,"Resp":"True"}))


@app.route('/getXAIOutput',methods=['POST'])
def getXaiOp():
    # return send_file(jsonDirectory+'/'+filename,attachment_filename=filename)
    print(eval(request.get_data()))
    req = eval(request.get_data())
    req_No = req["req_No"]
    xaiOp = Global_XaiOp[req_No]
    # del Global_Data_Graph_requests[req_No]
    # del Global_XaiOp[req_No]
    return str(json.dumps(xaiOp))

# @app.route('/genrateDataGraph',methods=['POST'])
# def generateDataGraph():
#     # return send_file(jsonDirectory+'/'+filename,attachment_filename=filename)
#     print(eval(request.get_data()))
#     req = eval(request.get_data())
#     modelSelected = req['modelName']
#     query = req['query']
#     sm = SelectModel(modelSelected)
#     dg = sm.getDataGraph(query)
#     req_no_gen = 'Req_'+str(randint(100000000, 999999999))
#     Global_Data_Graph_requests[req_no_gen]=dg
#     return str(json.dumps({"req_no":req_no_gen}))

# @app.route('/getDataGraph',methods=['POST'])
# def getDataGraph():
#     # return send_file(jsonDirectory+'/'+filename,attachment_filename=filename)
#     print(eval(request.get_data()))
#     req = eval(request.get_data())
#     req_No = req["req_No"]
#     dg = Global_Data_Graph_requests[req_No]
#     del Global_Data_Graph_requests[req_No]
#     return str(json.dumps(dg))
    


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            flash('File successfully uploaded')
            return redirect('/')
        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return redirect(request.url)
            

@app.route("/getModelName",methods=['POST'])
def getModelName():
    data = request.json
    print(data)
    jsonResp = data
    modelName = data['ModelName']
    gu = FileUpload("dummy.zip","dummy.relation")
    (Zipfile,relationFile) = gu.get_File_Details()
    zip = ZipFile(UPLOAD_FOLDER+"/"+Zipfile)
    zip.extractall(Data_Directory+"/"+modelName)
    fr = Finray(modelName,relationFile,modelName+'.json')
    mess = fr.generateInitialModel()
    if mess == 'MC':
        return str(jsonResp)
        # render_template('index.html')

@app.route('/python-flask-files-upload', methods=['POST'])
def upload_file_ajax():
    zipFileName = ""
    RelationFileName = ""
    # check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    
    files = request.files.getlist('files[]')
    
    errors = {}
    success = False
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)
            if filename.split(".")[1] == "zip":
                zipFileName = filename
            
            if filename.split(".")[1] == 'xlsx' or filename.split(".")[1] == 'xls':
                RelationFileName = filename
                file.save(os.path.join(Form_Directory, filename))

            file.save(os.path.join(UPLOAD_FOLDER, filename))
            success = True
        else:
            errors[file.filename] = 'File type is not allowed'
    
    if success and errors:
        errors['message'] = 'File(s) successfully uploaded'
        resp = jsonify(errors)
        resp.status_code = 206
        return resp
    if success:
        resp = jsonify({'message' : 'Files successfully uploaded'})
        fu = FileUpload(zipFileName,RelationFileName)
        fu.assign_filenames()
        resp.status_code = 201
        return resp
    else:
        resp = jsonify(errors)
        resp.status_code = 400
        return resp



@app.route('/')
def index():
    (retmess,finRayModels) = Load_Global_Models(jsonDirectory,Data_Dump_Directory)

    if retmess == True:
        selectJson = list()
        for eachModelOption in list(finRayModels['models'].keys()):
            option = {"name": eachModelOption, "checked" : False}
            selectJson.append(option)
    
        json.dump(selectJson,open(jsonSelectDirectory+"/selection.json","w"))
        return render_template('index.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=portDetails,debug=True)
