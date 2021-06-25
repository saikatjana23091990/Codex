class SelectModel():

    def __init__(self,modelName):
        self.modelName = modelName
        self.modelRelation = pd.read_csv(Data_Dump_Directory+'/'+self.modelName+'_relation.csv')
        
    def filter_selection(self,query):
        #Columns name from Query: sample: ["products-UnitPrice", "products-UnitsInStock", "categories-Picture", "products-SupplierID"]
        #Get Model Data
        data=finRayModels['modeldata'][self.modelName]
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
        return str(dim_mes_json)
    
    def getXAIOutput(self,features,response_var,get_feature_scores=True,plot_pdp=True,pdp_cols=None,tree_explanation=True,detect_outlier=True,detect_anomaly=False,anomaly_time_col=None,anomaly_series_col=None):
        #Get Model Data
        data=finRayModels['modeldata'][self.modelName]
        
        #Get current working directory/ specify processing folder here to store temp files.
        cwd=os.getcwd()
        
        #current dataset
        all_col=features
        all_col.append(response_var[0])
        # print(all_col)
        dataset=data[all_col]
        
        #explainer object
        explainer=Explainer(data=dataset,response_var=response_var[0])
        xai_outputs={}
        if get_feature_scores:
            xai_outputs['feature_scores']=explainer.get_feature_scores()
        
        if plot_pdp:
            
            pdp_paths=list()
            if pdp_cols is None:#none value implies all cols
                features = features[:len(features)-1]
                print(features)
                for eachcol in features:
                    print(eachcol)
                    explainer.pdp(eachcol)
                    path=os.path.join(cwd,'PDP',eachcol)
                    pdp_paths.append(path)
                         
            else:
                if type(pdp_cols) ==str:
                    pdp_cols=[pdp_cols]
                for eachcol in pdp_cols:
                    explainer.pdp(eachcol)
                    path=os.path.join(cwd,'PDP',eachcol)
                    pdp_paths.append(path)
                    
            xai_outputs['pdp_img_paths']=pdp_paths
            
        if tree_explanation:
            explanations=explainer.explain_tree()
            xai_outputs['tree_explanation']={'explanations':tree_explanation,'tree_path':os.path.join(cwd,'Decision Tree',f'dtree_structure_{response_var}.svg')}
            
        if detect_outlier:
            outliers=explainer.detect_outlier()
            path=os.path.join(cwd,"Outliers",f'Scores_{response_var}.png')
            xai_outputs['detected_outliers']={'row_outliers':outliers,'image_path':path}
            
        if detect_anomaly and anomaly_time_col is not None and anomaly_series_col is not None:
            anomaly=explainer.detect_anomaly(time_col=anomaly_time_col, series_col=anomaly_series_col)
            path=os.path.join(cwd,"Anomaly", f"anomaly_{anomaly_time_col}_{anomaly_series_col}.png")
            xai_outputs['detected_anomalies']={'image_path':path}

        #Json object of list of measures and dim
        xai_outputs_json=json.dumps(xai_outputs)
        #Return json object
        #
        return str(xai_outputs_json)
    
    def getDataGraph(self,query):
        # query = [{"Customers":["Region"]},{"Orders":["ShipName"]}]
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
    
            for i in range(len(filtered_Data)):
                evrrow = list(filtered_Data.iloc[i,:])
                evrrow = [str(elem) for elem in evrrow]
                nodes.append({"id":"-".join(evrrow).replace("'",","),"name":evrrow[0].replace("'",","),"values":"-".join(evrrow).replace("'",","),"entity":entityName})

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
                entity_one_Name = list(entity1.keys())[0].lower()
                entity_two_Name = list(entity2.keys())[0].lower()
                if Form_data_Rel[((Form_data_Rel['Table1'] == entity_one_Name) & (Form_data_Rel['Table2'] == entity_two_Name))].empty == True:
                    direction = "reverse"
                else:
                    direction = "straight"
                relation = list(dict(Form_data_Rel[((Form_data_Rel['Table1'] == entity_one_Name) & (Form_data_Rel['Table2'] == entity_two_Name)) | ((Form_data_Rel['Table1'] == entity_two_Name) & (Form_data_Rel['Table2'] == entity_one_Name))])['Relation'])[0]
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

        return dataGraph   




@app.route('/getDimMesDetails',methods=['POST'])
def getDimMesDetails():
    print(eval(request.get_data()))
    req = eval(request.get_data())
    modelSelected = req['modelName']
    query = req['query']
    sm = SelectModel(modelSelected)
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
    sm = Global_Data_Graph_requests[Req_No]
    xaiop = sm.getXAIOutput(Selection['dimensions'],Selection['measures'])
    return str(json.dumps(xaiop))
