
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
            modelName = eachModelData.split(".")[0]
            modeldata[modelName] = fileDf
        
    
        finRayModels = {'models':models,'modeldata':modeldata}
    
        return (True,finRayModels)
    except:
        print("There is some error in the Global model creation")
        return (False,{})



