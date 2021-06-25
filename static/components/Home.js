// src/components/App.js
var node, link, colors, svg, width,height,simulation,edgepaths,edgelabels ,dragstarted ,dragged, ticked; // global variables for d graph

export default {
  name: 'App',
  data(){
      return {
        showGraph:false,
        showSelections:false,
        selectionVal:"",
        getselectionArrRoot:[],
        getselectionArrFilter:[],
        SelectionArr:[],
        itemsSelection:[],
        AlertMsg:"",
        showAlertMsg:false,
        dialog: false,
		dialog1:false,
        loaderDialog : false,
        MeasureRadios:"",
        MeasureButtonsToggle:[],
        DimensionButtonsToggle:[],
		measureData:"",
        measureDataNew:"",
        newDimData: [],
        newMesData:[],
        isButtonSelectDisabled:false,
        panel:1,
        file_name:"",
        fileInput:null,
        customLoader:false,
        panel: [0],
        decesionTreetreePath:"",
        decesionTreeHTML:"",
        globalPerformancePath:"",
        globalPerfMetrices:"",
        surrogatePerformancePath:"",
        surrogatePerfMetrices:"",
        detectedOutliers:"",
        detectedOutliersImage:"",
        featureScores:"",
        pdpImgPaths:[],
        dialogOutlier:false,
        dialogdecesionTreepath:false,
        showAdditionalXAIdataPanel:false,
        XAIFound:false,
        alwaysTrue: true,
        AllFields : [],
        MainXAIJSON : {'feature_var' : {}, 'response_var' :{}},
        response_variable_selected : false,
      }
  },
  created(){
     this.getModelNameSelection();
  },
  methods:{
    
    getGrapthData: function(ModelName){
         this.showGraph=true;
         colors = d3.scaleOrdinal(d3.schemeCategory10);
        //console.log(d3.schemeCategory10)
         svg = d3.select("svg"),
            width = +svg.attr("width"),
            height = +svg.attr("height"),
            node,
            link;
        
        // svg.append('defs').append('marker')
        //     .attrs({'id':'arrowhead',
        //         'viewBox':'-0 -5 10 10',
        //         'refX':50,
        //         'refY':0,
        //         'orient':'auto',
        //         'markerWidth':13,
        //         'markerHeight':13,
        //         'xoverflow':'visible'})
        //     .append('svg:path')
        //     .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
        //     .attr('fill', '#999')
        //     .style('stroke','none');

         simulation = d3.forceSimulation()
            .force("link", d3.forceLink().id(function (d) {return d.id;}).distance(150))
            //.force("charge", d3.forceManyBody())
            .force("charge", d3.forceManyBody().strength(-50))
            .force("center", d3.forceCenter(width / 2, height / 2));
            
        var vm = this;
        
        fetch("modeljson/"+ModelName+".json")
        .then(function(r){return r.json()})
                                .then(function(graph){
                                    console.log(graph);
                                    graph.nodes.map(function(item){
                                        vm.getselectionArrRoot.push({
                                            "name":item.id,
                                            "features":item.features.map(function(itemFeatures){
                                                            return {"name":item.name+"-"+itemFeatures,"label":itemFeatures ,"checked":false};
                                                        })
                                        })
                                    });
                                    console.log(vm.getselectionArrRoot)
                                    vm.update(graph.links, graph.nodes);
                                })
        
    },  
    
    update:function(links, nodes) {
        link = svg.selectAll(".link")
            .data(links)
            .enter()
            .append("line")
           // .attr("class", "link")
            .attr("class",function(d,i){
                    return "link intensityLink"+d.source;
            })
            .attr('marker-end','url(#arrowhead)')

        link.append("title")
            .text(function (d) {return d.type;});

        edgepaths = svg.selectAll(".edgepath")
            .data(links)
            .enter()
            .append('path')
            .attrs({
                'class': 'edgepath',
                'fill-opacity': 0,
                'stroke-opacity': 0,
                'id': function (d, i) {return 'edgepath' + i}
            })
            .style("pointer-events", "none");

        edgelabels = svg.selectAll(".edgelabel")
            .data(links)
            .enter()
            .append('text')
            .style("pointer-events", "none")
            .attrs({
                'class': 'edgelabel',
                'id': function (d, i) {return 'edgelabel' + i},
                'font-size': 10,
                'fill': 'black'
            });

        edgelabels.append('textPath')
            .attr('xlink:href', function (d, i) {return '#edgepath' + i})
            .style("text-anchor", "middle")
            .style("pointer-events", "none")
            .attr("startOffset", "50%")
            .text(function (d) {return d.type});

        node = svg.selectAll(".node")
            .data(nodes)
            .enter()
            .append("g")
            .attr("class", "node")
            .call(d3.drag()
                    .on("start", this.dragstarted)
                    .on("drag", this.dragged)
                    //.on("end", dragended)
            );
            
        var sendARR=[];
        var vm = this;
        node.append("circle")
            .attr("r", function(d){
                //return d.node_Aff*5;
                return 12;
                
            })
            .style("fill", function (d, i) {
                // if(d.node_Aff=="1"){
                //  return "red";
                // }
                // if(d.node_Aff=="orders"){
                //  return "green";
                // }
                // if(d.node_Aff=="customers"){
                //  return "blue";
                // }    
                
                return colors(i);
                
            })
            .attr("trackid",function(d, i){
                return d.name;
            })
            .on("mouseover",function(d, i){
                console.log(d.id)
                $(".intensityLink"+d.id).each(function(){ console.log(".intensityLink"+d.id)
                    $(this).addClass("heighLightStroke");
                })
            })
            .on("mouseout",function(d, i){
                console.log(d.id)
                d3.select(this).style("cursor", "pointer");
                $(".intensityLink"+d.id).each(function(){ console.log(".intensityLink"+d.id)
                    $(this).removeClass("heighLightStroke");
                })
            })
            .on("click",function(d, i){
                vm.showSelections = true;
                if(vm.getselectionArrFilter.indexOf(d.id)==-1){
                    vm.getselectionArrFilter.push(d.id);    
                }
                console.log(vm.getselectionArrFilter)
            })

        node.append("title")
            .text(function (d) {return d.id;});

       node.append("text")
            .attr("dy", -1)
            .attr("dx", function(d){
                return (Number(d.node_Aff)*10)+2
            })
            .attrs({
                'font-size': 15,
                'fill': 'black'
            })
            .text(function (d) {return d.name+" : "+String(d.features.length);});

        simulation
            .nodes(nodes)
            .on("tick", this.ticked);

        simulation.force("link")
            .links(links);
    },

    ticked:function() {
        //console.log("called")
        link
            .attr("x1", function (d) {return  d.source.x;})
            .attr("y1", function (d) {return d.source.y;})
            .attr("x2", function (d) {return d.target.x;})
            .attr("y2", function (d) {return d.target.y;});

        node
            .attr("transform", function (d) {return "translate(" + d.x + ", " + d.y + ")";});

        edgepaths.attr('d', function (d) {
            return 'M ' + d.source.x + ' ' + d.source.y + ' L ' + d.target.x + ' ' + d.target.y;
        });

        edgelabels.attr('transform', function (d) {
            if (d.target.x < d.source.x) {
                var bbox = this.getBBox();

                var rx = bbox.x + bbox.width / 2;
                var ry = bbox.y + bbox.height / 2;
                return 'rotate(180 ' + rx + ' ' + ry + ')';
            }
            else {
                return 'rotate(0)';
            }
        });
    },

    dragstarted:function(d) {
        if (!d3.event.active) simulation.alphaTarget(0.3).restart()
        d.fx = d.x;
        d.fy = d.y;
    },

    dragged:function(d) {
        d.fx = d3.event.x;
        d.fy = d3.event.y;
    },
    
    getConnect: function(){
        this.getselectionArrRoot=[];
        this.getselectionArrFilter=[];
        this.SelectionArr=[];
        this.showSelections=false;
        
        console.log(this.selectionVal)
        $("#grapgViewPort").empty()
        this.openDialog();                      
    },
    
    Create: function(){
        this.showAlertMsg=true;
        
        var form_data = new FormData();
        var ins = document.getElementById('multiFiles').files.length;
        for (var x = 0; x < ins; x++) {
            form_data.append("files[]", document.getElementById('multiFiles').files[x]);
        }
                
        console.log(form_data);
                
        fetch("/python-flask-files-upload",{
            method:"post",
            body:form_data,
        })
        .then((r)=>{return r})
        .then((data)=>{
            console.log(this.file_name)
            
            fetch("/getModelName",{
                method:"post",
                body:JSON.stringify({"ModelName": this.file_name}),
                headers: new Headers({'content-type':'application/json'})
            })
            .then((r)=>{
                return r;}
                )
            .then((data)=>{
                    
                this.AlertMsg = "Model created Successfully!!";
                this.file_name = "";
                this.fileInput = null;
                this.getModelNameSelection(); // call model name
                
            });
            
        });
        
        
    },
    
    getModelNameSelection: function(){
                fetch("/modeljsonselect",{
                    method:"get"
                }).then((r)=>{ return r.json(); })
                  .then((res)=>{ console.log(res)
                    
                    var vm = this;
                    res.map(function(item){
                        vm.itemsSelection.push(item.name);
                    })
                    
                  });
    },
    
    getitemSelection: function(item){
        if(item.checked){
            console.log(item.name, item.checked)
            this.SelectionArr.push(item.name);
        }else{
            console.log(item.name, item.checked)
            this.SelectionArr.splice(this.SelectionArr.indexOf(item.name),1);
        }
        console.log(this.SelectionArr)
    },
    
    navigateToExport() {
            this.response_variable_selected = false;
            
			fetch("/getDimMesDetails",{
					method:"post",
					body:JSON.stringify({"query": this.SelectionArr,"modelName":this.selectionVal}),
					headers: new Headers({'content-type':'application/json'})
				})
				.then((r)=>{
					return r.json();
				})
				.then((data)=>{
					console.log(data);
					sessionStorage.setItem("req_No", data["req_no"]);
                    this.measureData = JSON.parse(data["Resp"]);
                    this.newMesData = this.measureData.measures;	
					// this.measureData = {"dimensions": ["ProductName", "CompanyName_suppliers"], "measures": ["UnitPrice_products","UnitPrice_products_new"]};
					console.log(this.measureData.measures.length);
					if(this.alwaysTrue == true){
                        if (this.measureData.measures.length == 0) {
                            var arr = {
                                "dimensions": [],
                                "measures": this.measureData.dimensions
                            }
                            this.measureData = arr;

                        }

                        var arr = {
                                "dimensions": this.measureData.dimensions,
                                "measures": this.measureData.measures
                        }

                        this.AllFields = this.measureData.dimensions.concat(this.measureData.measures);
                   
                        this.measureDataNew = this.measureData;
                        this.dialog1 = true;
                        
                        this.measureData.dimensions.forEach((t) => {
                            this.MainXAIJSON['feature_var'][t] = 'category';
                        })

                        this.measureData.measures.forEach((t) => {
                            this.MainXAIJSON['feature_var'][t] = 'numeric';
                        })

                        console.log(this.MainXAIJSON);
						
					}
					else{
                        var request_data = {}
                        request_data['Selection'] = this.measureData
                        request_data['RequestNo'] = sessionStorage.getItem("req_No")
                        this.customLoader = true;
                        this.loaderDialog = true;
                        $(".customViewCard").addClass("overlayCustom");
                        // this.showSelections = false;
                        // document.querySelector("#grapgViewPort").style.display = "none";
						fetch("/getXAIDetails",{
							method:"post",
							body:JSON.stringify(request_data),
							headers: new Headers({'content-type':'application/json'})
						})
						.then((r)=>{
							return r.json();})
						.then((data)=>{
                                this.customLoader = false;
                                this.loaderDialog = false;
                                $(".customViewCard").removeClass("overlayCustom");
                                console.log(data);
                                if (data["Resp"] == "True"){
                                    sessionStorage.setItem("req_No", data["req_no"]);
                                    // this.$router.push('/export');
                                    $("html").css({"overflow-y":"hidden"});
                                    this.getAdditionXAIdata();
                                }
                                else{
                                    console.log("Failed in XAI Generation")
                                }
						}); 
					}				
						
			});
             
    },

    goToDataPage : function(){
        sessionStorage.setItem("reloadNeeded", "true");
        this.$router.push('/export');
    },
	
	navigateToExportUpdate(cache){
		this.dialog1 = false;
        console.log(this.measureDataNew);
        var request_data = {}
        request_data['Selection'] = this.MainXAIJSON;
        request_data['cache_required'] = cache;
        request_data['RequestNo'] = sessionStorage.getItem("req_No");
        this.customLoader = true;
        this.loaderDialog = true;
        console.log(request_data);
        $(".customViewCard").addClass("overlayCustom");
        // this.showSelections = false;
        // document.querySelector("#grapgViewPort").style.display = "none";
		fetch("/getXAIDetails",{
							method:"post",
							body:JSON.stringify(request_data),
							headers: new Headers({'content-type':'application/json'})
			})
			.then((r)=>{
				return r.json();})
			.then((data)=>{
                this.customLoader = false;
                this.loaderDialog = false;
                $(".customViewCard").removeClass("overlayCustom");
                    console.log(data);
					if (data["Resp"] == "True"){
                        sessionStorage.setItem("req_No", data["req_no"]);
                        // this.$router.push('/export');
                        $("html").css({"overflow-y":"hidden"});
                        this.getAdditionXAIdata();
                    }
                    else{
                        console.log("Failed in XAI Generation")
                    }
			}); 
		
	},

    getAdditionXAIdata(){
        this.showAdditionalXAIdataPanel = true;
        var vm = this;                        
        fetch("/getXAIOutput",{
             method:"post",
                body:JSON.stringify({"req_No":sessionStorage.getItem("req_No")}),
                headers: new Headers({'content-type':'application/json'})
                }).then(function(r){return r.json()})
                                .then(function(data){
                                    data = JSON.parse(data);
                                    console.log(data);
                                    vm.XAIFound = true;
 

                                    vm.decesionTreetreePath = data.tree_explanation.tree_path;
                                    vm.globalPerformancePath = data.Global_Perf_Metrices;
                                    vm.surrogatePerformancePath = data.Surrogate_Perf_Metrices;
                                    
                                    // for decesionTreeHTML
                                        fetch(data.tree_explanation.explanations_statement,{
                                              method:"get",
                                              headers: new Headers({'content-type':'text/html'})
                                            })
                                                .then(function(r){return r.text()})
                                                .then(function(htmlData){
                                                    console.log(htmlData);
                                                    vm.decesionTreeHTML = htmlData;    
                                                    
                                            })
                                    //
                                    // for Glbal Performance Metrices
                                    fetch(data.Global_Perf_Metrices,{
                                        method:"get",
                                        headers: new Headers({'content-type':'text/html'})
                                      })
                                          .then(function(r){return r.text()})
                                          .then(function(metricesData){
                                              console.log(metricesData);
                                              vm.globalPerfMetrices = metricesData;    
                                              
                                      })

                                    // for Surrogate Performance Metrices
                                    // fetch(data.Surrogate_Perf_Metrices,{
                                    //     method:"get",
                                    //     headers: new Headers({'content-type':'text/html'})
                                    //   })
                                    //       .then(function(r){return r.text()})
                                    //       .then(function(metricesData){
                                    //           console.log(metricesData);
                                    //           vm.surrogatePerfMetrices = metricesData;    
                                              
                                    //   })
                                    //
                                    //
                                    vm.detectedOutliersImage  = data.detected_outliers.image_path;
                                    vm.detectedOutliers  = data.detected_outliers.row_outliers;
                                    console.log(vm.detectedOutliersImage)
                                    //
                                    vm.featureScores = data.feature_scores;

                                    
                                    // for  pdpImgPaths
                                    vm.pdpImgPaths = data.pdp_img_paths;
                                    
                                    $(".showfeatures").click(function(){
                                        //alert('asdasdas')
                                        $("#page-wrap1").empty();
                                        vm.loadDataTable(data.feature_scores,"page-wrap1");
                                        
                                    });
                                    
                                });
                                
        
    },


    loadDataTable: function(filePath,divId){
        // alert("Displaying Data Table");
        // alert(filePath+divId);
        d3.csv(filePath, function(error, data) {
          if (error) throw error;
         
          var sortAscending = true;
          var table = d3.select('#'+divId).append('table');
          var titles = d3.keys(data[0]);
          var headers = table.append('thead').append('tr')
                           .selectAll('th')
                           .data(titles).enter()
                           .append('th')
                           .text(function (d) {
                                return d;
                            })
                           .on('click', function (d) {
                               headers.attr('class', 'header');
                              
                               if (sortAscending) {
                                 rows.sort(function(a, b) { return b[d] < a[d]; });
                                 sortAscending = false;
                                 this.className = 'aes';
                               } else {
                                 rows.sort(function(a, b) { return b[d] > a[d]; });
                                 sortAscending = true;
                                 this.className = 'des';
                               }
                              
                           });
         
          var rows = table.append('tbody').selectAll('tr')
                       .data(data).enter()
                       .append('tr');
          rows.selectAll('td')
            .data(function (d) {
                return titles.map(function (k) {
                    return { 'value': d[k], 'name': k};
                });
            }).enter()
            .append('td')
            .attr('data-th', function (d) {
                return d.name;
            })
            .text(function (d) {
                return d.value;
            });
      });
     
    },

    openOutlier: function(){
        this.dialogOutlier = true;
    },

    openDecesionTreepath: function(){
        this.dialogdecesionTreepath = true;
    },
	
	SetnewMeasureforExport(val){ 
        console.log("Returning from Measure",val);
        this.response_variable_selected = true;

        let features = {};
        let responses = {};
        this.measureData.dimensions.forEach((f)=>{
            if (f != val){
                features[f] = 'category';
            }
            else {
                responses[f] = 'category';
            }

        });

        this.measureData.measures.forEach((m)=>{
            if (m != val){
                features[m] = 'numeric';
            }
            else {
                responses[m] = 'numeric';
            }

        });

        this.MainXAIJSON['feature_var'] = features;
        this.MainXAIJSON['response_var'] = responses;
        
        
		// var menuListLocal=[val];
		// var arr;
		// arr = this.measureData;
		// var arr = {
		// 	 "dimensions": this.measureData.dimensions,
		// 	 "measures": menuListLocal
		// }
		
		// if(val!=""){
		// this.measureDataNew = arr;
		// //console.log(arr)
		// console.log(this.measureDataNew)
        // }
        
        // if (this.measureDataNew.measures.includes(val) == true){
        //     this.MainXAIJSON['response_var'][val] = 'numeric';
        // }
        // else {
        //     this.MainXAIJSON['response_var'][val] = 'category';
        // }

        // var alreadyAssignedMeasure = Object.keys(this.MainXAIJSON['response_var']);
        // delete  this.MainXAIJSON['feature_var'][alreadyAssignedMeasure];

        console.log(this.MainXAIJSON);
		
    },
    
    SetnewCategoricalValue(val){ 
        console.log(val)
        if (val!=""){
            var mesarr = this.measureData.measures;
            var index = mesarr.indexOf(val);
            if (index > -1) {
                mesarr.splice(index, 1);
            }

            var dimarr = this.measureData.dimensions;
            dimarr.push(val);
            this.newDimData.push(val);
            this.newMesData = mesarr;
            // if (this.newMesData.length == 1){
            //     this.isButtonSelectDisabled = true;
            // }
            // else {
            //     this.isButtonSelectDisabled = false;
            // }

            this.measureData = {
                "dimensions": dimarr,
			    "measures": mesarr
            }

            this.measureDataNew = this.measureData;

            console.log(this.measureData);

            this.MainXAIJSON['feature_var'] = {};

            mesarr.forEach((m) => {
                if (Object.keys(this.MainXAIJSON['response_var']).includes(m) == false){
                    this.MainXAIJSON['feature_var'][m] = 'numeric';
                }
                else {
                    this.MainXAIJSON['response_var'][m] = 'numeric';
                }
                
            });

            dimarr.forEach((d) => {
                if (Object.keys(this.MainXAIJSON['response_var']).includes(d) == false){
                    this.MainXAIJSON['feature_var'][d] = 'category';
                }
                else {
                    this.MainXAIJSON['response_var'][d] = 'category';
                }
                
            });

            var alreadyAssignedMeasure = Object.keys(this.MainXAIJSON['response_var']);
            delete  this.MainXAIJSON['feature_var'][alreadyAssignedMeasure];

            console.log(this.MainXAIJSON);

        }
		
    },
    
    DeleteFromCategoricalValue(val){ 
        console.log(val)

        if (val!=""){
            var dimarr = this.measureData.dimensions;
            var index = dimarr.indexOf(val);
            if (index > -1) {
                dimarr.splice(index, 1);
            }

            var mesarr = this.measureData.measures;
            mesarr.push(val);

            var newArr = this.newDimData;
            var index2 = newArr.indexOf(val);
            if (index2 > -1) {
                newArr.splice(index2, 1);
            }

            this.newDimData = newArr;
            this.newMesData = mesarr;

            // if (this.newMesData.length == 1){
            //     this.isButtonSelectDisabled = true;
            // }
            // else {
            //     this.isButtonSelectDisabled = false;
            // }

            this.measureData = {
                "dimensions": dimarr,
			    "measures": mesarr
            }

            this.measureDataNew = this.measureData;

            console.log(this.measureData);


            this.MainXAIJSON['feature_var'] = {};

            mesarr.forEach((m) => {
                if (Object.keys(this.MainXAIJSON['response_var']).includes(m) == false){
                    this.MainXAIJSON['feature_var'][m] = 'numeric';
                }
                else {
                    this.MainXAIJSON['response_var'][m] = 'numeric';
                }
                
            });

            dimarr.forEach((d) => {
                if (Object.keys(this.MainXAIJSON['response_var']).includes(d) == false){
                    this.MainXAIJSON['feature_var'][d] = 'category';
                }
                else {
                    this.MainXAIJSON['response_var'][d] = 'category';
                }
                
            });

            var alreadyAssignedMeasure = Object.keys(this.MainXAIJSON['response_var']);
            delete  this.MainXAIJSON['feature_var'][alreadyAssignedMeasure];

            console.log(this.MainXAIJSON);

        }
	},
	
    openDialog(){ 
        this.dialog=true; 
        var vm = this;
        setTimeout(function(){vm.getGrapthData(vm.selectionVal);},100)
        
        }
      
  },
  
   
  template: `
  
  <v-app >
    <v-alert
      dense
      text
      type="success"
      dismissible
      v-if="showAlertMsg"
    >
    {{AlertMsg}}
    </v-alert>
    
    
            <div class="form-inline customform-inline">
                <div class="form-group">
                    <v-select
                      :items="itemsSelection"
                      label="Select your model"
                      dense
                      v-model="selectionVal"
                    ></v-select>
                </div>
                <v-btn rounded color="teal darken-1" @click="getConnect()">Connect</v-btn>
                <div style="margin:15px 0px; text-align:center;">
                or Create a Fresh Model
                </div>
                
                <div>
                    <v-text-field dense label="Model Name" v-model="file_name"></v-text-field>
                    <v-file-input chips label="Upload your Relation file and Data file" id="multiFiles" v-model="fileInput" name="files[]" multiple="multiple"></v-file-input>
                <!---<input type="file" id="multiFiles" name="files[]" multiple="multiple"/>--->
                    <v-btn rounded color="orange darken-1" @click="Create()">Create</v-btn>
                </div>
            </div>
        
            
   
   
   
   
                <v-dialog v-model="dialog" fullscreen hide-overlay transition="dialog-bottom-transition">
                  <v-card class="customViewCard">
                    <v-toolbar dark color="blue-grey darken-2">
                      <v-btn icon dark @click="dialog = false">
                        <v-icon>mdi-close</v-icon>
                      </v-btn>
                      <v-toolbar-title>Graph</v-toolbar-title>
                      <v-spacer></v-spacer> 
                      <v-btn rounded color="deep-orange darken-1" @click="navigateToExport()">Get Explaination</v-btn>
                      <v-btn rounded v-show="XAIFound" color="green darken-1" @click="goToDataPage()" style="margin:5px">Data Details</v-btn>
                    </v-toolbar>
                    
                    <div class="row" style="margin:0">
                    <div class="col-3 col-3-leftpanel">

                    
                    <v-expansion-panels focusable>
                <v-expansion-panel v-show="showSelections">
                  <v-expansion-panel-header>Selection Panel</v-expansion-panel-header>
                  <v-expansion-panel-content>
                    <div class="column_wrapper" v-for="item in getselectionArrRoot" :key="item.name" v-if="getselectionArrFilter.indexOf(item.name)!=-1">
                        <h1 class="customheadingFilter">{{item.name}}</h1><br/>
                        <span class="customfilteOptionarea" v-for="itemFilter in item.features" :key="itemFilter.label">
                            <v-checkbox
                              v-model="itemFilter.checked"
                              :label="itemFilter.label"
                              @change="getitemSelection(itemFilter);"
                            ></v-checkbox>
                        </span>
                    </div>  
                    
                  </v-expansion-panel-content>
                </v-expansion-panel>
                
            </v-expansion-panels>
            </div>       
            <div class="col-9">
                
                    <!------->
                <!---<v-expansion-panels v-model="panel"  v-show="showAdditionalXAIdataPanel" multiple>---->
                <v-expansion-panels v-show="showAdditionalXAIdataPanel">
                  
                  <v-expansion-panel>
                    <v-expansion-panel-header>Decision Tree Explainer</v-expansion-panel-header>
                    <v-expansion-panel-content>

 

                       <div class="col-xs-6 customPanelbox-wrapper"><div class="customPanelbox">
                            <v-img :src="decesionTreetreePath"  @click="openDecesionTreepath();" class="decesionTreeImgCustom"></v-img>
                       </div></div>
                       <div class="col-xs-6 customPanelbox-wrapper">
                            <div class="customPanelbox">
                                <div v-html="decesionTreeHTML"></div>
                            </div>
                       </div>
                       
                       <v-dialog
                          v-model="dialogdecesionTreepath">
                          <v-card>
                            <v-card-title
                              class="headline blue-grey darken-3"
                              primary-title
                            >
                              Decision Tree
                            </v-card-title>

 

                            <v-card-text>
                                    <v-img :src="decesionTreetreePath"></v-img>                
                            </v-card-text>

 

                            <v-divider></v-divider>

 

                            <v-card-actions>
                              <v-spacer></v-spacer>
                              <v-btn color="deep-orange darken-3" text @click="dialogdecesionTreepath = false">Close</v-btn>
                            </v-card-actions>
                          </v-card>
                        </v-dialog>
                       
                       
                    </v-expansion-panel-content>
                  </v-expansion-panel>
                  <v-expansion-panel>
                    <v-expansion-panel-header>Outlier and Anomaly</v-expansion-panel-header>
                    <v-expansion-panel-content>
                    
                      <div class="col-xs-6 customPanelbox-wrapper"><div class="customPanelbox"><v-img :src="detectedOutliersImage" @click="openOutlier();" class="outlayerImgCustom"></v-img></div></div>
                      <div class="col-xs-6 customPanelbox-wrapper">
                        <div class="customPanelbox" v-if="detectedOutliers!='None'"><v-img :src="detectedOutliers"></v-img></div>
                        <div class="customPanelbox" v-if="detectedOutliers=='None'">No anomaly detected &nbsp;&nbsp;&#128517;</div>
                      </div>
                      
                       <v-dialog
                          v-model="dialogOutlier">
                          <v-card>
                            <v-card-title
                              class="headline blue-grey darken-3"
                              primary-title
                            >
                              Outlier
                            </v-card-title>

 

                            <v-card-text>
                                    <v-img :src="detectedOutliersImage" aspect-ratio="1.7"></v-img>                
                            </v-card-text>

 

                            <v-divider></v-divider>

 

                            <v-card-actions>
                              <v-spacer></v-spacer>
                              <v-btn color="deep-orange darken-3" text @click="dialogOutlier = false">Close</v-btn>
                            </v-card-actions>
                          </v-card>
                        </v-dialog>
                      
                      
                    </v-expansion-panel-content>
                  </v-expansion-panel>
                  <v-expansion-panel>
                    <v-expansion-panel-header class="showfeatures">Model Accuracy Matrices</v-expansion-panel-header>
                    <v-expansion-panel-content>
                    <div class="col-xs-6 customPanelbox-wrapper"> <b>For Global Model :</b> 
                        <div class="customPanelbox"> 
                            <div v-html="globalPerfMetrices"></div>
                        </div>
                    </div>
                    <div class="col-xs-6 customPanelbox-wrapper"> <b>For Surrogate Model :</b> 
                        <div class="customPanelbox"> 
                            <div v-html="surrogatePerfMetrices"></div>
                        </div>
                    </div>
                    </v-expansion-panel-content>
                  </v-expansion-panel>
                  <v-expansion-panel>
                    <v-expansion-panel-header class="showfeatures">PDP Images and Feature Scores</v-expansion-panel-header>
                    <v-expansion-panel-content>
                    
                    <div>        
                      <div id="page-wrap1"></div>
                    <div>    
                    
                      <!---------->
                       <v-carousel
                            
                            height="1200"
                            hide-delimiter-background
                            show-arrows-on-hover
                          >
                        <v-carousel-item active-class="pdpImages"
                          v-for="(item,i) in pdpImgPaths"
                          :key="i"
                          :src="item"
                        ></v-carousel-item>
                      </v-carousel>
                      
                      <!----------->
                    
                    </v-expansion-panel-content>
                  </v-expansion-panel>
                </v-expansion-panels>

 

                        <!-------->
                    
                        <div style="padding-bottom:100px;height:800px;width:100%;">
                            <svg id="grapgViewPort" width="700" height="800" style="width:100%; margin-top:15px;"></svg>
                        </div>
                        </div>           
                        </div>
                  </v-card>

                    <v-dialog
                      v-model="loaderDialog"
                      width="600"
                      v-if="customLoader"
                    >
                    
                    <b style="color: white">Fetching Explainations...please wait</b>
                    
                </v-dialog>
                    

				  <!--------->
				  <v-dialog
					  v-model="dialog1"
					  width="500"
					>
						  <v-card>
							<v-card-title
							  class="headline blue-grey darken-2"
							  primary-title
							>
							  Markov Blanket preview
                            </v-card-title>
                            
                            <v-card-text>
                                      Click to set the item as categrocial <br>
                            </v-card-text>

                            <v-btn-group
                                v-model="MeasureButtonsToggle"
                                multiple
                            >
                                <v-btn class="defaultwos" v-for="item in measureData.measures" color="cyan darken-1" :disabled="isButtonSelectDisabled" @click="SetnewCategoricalValue(item)">
                                    {{ item }}
                                </v-btn>

                            </v-btn-group>

                            <v-card-text>
                                      <br> Items considered categroical (<i>click to remove if needed</i>) <br>
                            </v-card-text>

                            <v-btn-group
                                v-model="DimensionButtonsToggle"
                                multiple
                            >
                                <v-btn class="defaultws" v-for="item in measureData.dimensions" color="orange darken-1" @click="DeleteFromCategoricalValue(item)">
                                    {{ item }}
                                </v-btn>

                            </v-btn-group>

                            <v-divider></v-divider>
                            
                            <v-card-text>
                                      Please select a response variable <br>
                            </v-card-text>
                            
                            

							<v-card-text >
  
									<v-radio-group v-model="MeasureRadios" :mandatory="false">
									  <v-radio v-for="item in AllFields" :label="item" :value="item" @change="SetnewMeasureforExport(item)"></v-radio>
									</v-radio-group>

							</v-card-text>

							<v-divider></v-divider>

							<v-card-actions>
							  <v-spacer></v-spacer>
                              <v-btn
                              color="orange darken-1"
							    text
                                v-if = "response_variable_selected == true"
								@click="navigateToExportUpdate('false')"
                              >
                              Run(Without Cache)
                              </v-btn>
							  <v-btn
								color="orange darken-1"
								text
                                v-if = "response_variable_selected == true"
								@click="navigateToExportUpdate('true')"
							  >
								Run(Considering cache)
							  </v-btn>
							</v-card-actions>
						  </v-card>
                          </v-dialog>
				  
				  <!------>
				  
				  
				  
				  
                </v-dialog>
   
   </v-app>
   
  `,
};
