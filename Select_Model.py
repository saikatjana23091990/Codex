class SelectModel():

    def __init__(self,modelName):
        self.modelName = modelName
        self.modelRelation = pd.read_csv(Data_Dump_Directory+'/'+self.modelName+'_relation.csv')


    def getDataGraph(self,query):
        # query = [{"Customers":["Region"]},{"Orders":["ShipName"]}]

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
        data = finRayModels['modeldata'][self.modelName][features_selected].drop_duplicates().dropna()

        nodes = list()
        for eachEntity in query:
            entityName = list(eachEntity.keys())[0]
            featureSelected = eachEntity[entityName]
            filtered_Data =  data[featureSelected]
            filtered_Data = filtered_Data.dropna()
    
            for i in range(len(filtered_Data)):
                evrrow = list(filtered_Data.iloc[i,:])
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
                    entity_one_df = data.iloc[j,:][list(entity1.values())[0]]
                    entity_two_df = data.iloc[j,:][list(entity2.values())[0]]
                    if direction == "straight":
                        links.append({"source":"-".join(list(entity_one_df)).replace("'",","),"target":"-".join(list(entity_two_df)).replace("'",","),"type":relation})
                    else:
                        links.append({"target":"-".join(list(entity_one_df)).replace("'",","),"source":"-".join(list(entity_two_df)).replace("'",","),"type":relation})

        for k in range(len(links)):
            links[k]=str(links[k])

        links = list(set(links))

        for k in range(len(links)):
            links[k] = json.loads(links[k].replace("'", '"'))                     

        dataGraph = {"nodes":nodes,"links":links}

        return dataGraph