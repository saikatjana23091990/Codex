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
		panel:1,
		file_name:"",
		fileInput:null
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
		
		svg.append('defs').append('marker')
			.attrs({'id':'arrowhead',
				'viewBox':'-0 -5 10 10',
				'refX':50,
				'refY':0,
				'orient':'auto',
				'markerWidth':13,
				'markerHeight':13,
				'xoverflow':'visible'})
			.append('svg:path')
			.attr('d', 'M 0,-5 L 10 ,0 L 0,5')
			.attr('fill', '#999')
			.style('stroke','none');

		 simulation = d3.forceSimulation()
			.force("link", d3.forceLink().id(function (d) {return d.id;}).distance(250))
			//.force("charge", d3.forceManyBody())
			.force("charge", d3.forceManyBody().strength(-300))
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
                'fill': '#aaa'
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
				return d.node_Aff*5;
				
			})
            .style("fill", function (d, i) {
				// if(d.node_Aff=="1"){
				// 	return "red";
				// }
				// if(d.node_Aff=="orders"){
				// 	return "green";
				// }
				// if(d.node_Aff=="customers"){
				// 	return "blue";
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
			
			fetch("/genrateDataGraph",{
				method:"post",
				body:JSON.stringify({"query": this.SelectionArr,"modelName":this.selectionVal}),
				headers: new Headers({'content-type':'application/json'})
			})
			.then((r)=>{
				return r.json();})
			.then((data)=>{
					console.log("success")
					console.log(data)
					console.log(this.$router)
					sessionStorage.setItem("req_No", data["req_no"]);
					this.$router.push('/export');
			});	
			
			 
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
				<v-btn rounded color="success" @click="getConnect()">Connect</v-btn>
				<div style="margin:15px 0px; text-align:center;">
				or Create a Fresh Model
				</div>
			    
				<div>
					<v-text-field dense label="Model Name" v-model="file_name"></v-text-field>
					<v-file-input chips label="Upload your Relation file and Data file" id="multiFiles" v-model="fileInput" name="files[]" multiple="multiple"></v-file-input>
				<!---<input type="file" id="multiFiles" name="files[]" multiple="multiple"/>--->
					<v-btn rounded color="success" @click="Create()">Create</v-btn>
				</div>
			</div>
		
			
   
   
   
   
				<v-dialog v-model="dialog" fullscreen hide-overlay transition="dialog-bottom-transition">
				  <v-card>
					<v-toolbar dark color="primary">
					  <v-btn icon dark @click="dialog = false">
						<v-icon>mdi-close</v-icon>
					  </v-btn>
					  <v-toolbar-title>Graph</v-toolbar-title>
					  <v-spacer></v-spacer> 
					  <v-btn rounded color="success" @click="navigateToExport()">Export</v-btn>
					</v-toolbar>
					
					<v-expansion-panels focusable>
				<v-expansion-panel v-show="showSelections" >
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
					
					
					
						<div style="padding-bottom:100px;">
							<svg id="grapgViewPort" width="1250" height="800" style="width:100%; margin-top:15px;"></svg>
						</div>
				  </v-card>
				</v-dialog>
   
   </v-app>
   
  `,
};