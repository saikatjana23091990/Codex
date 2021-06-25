// src/components/App.js
var node, link, colors, svg, width,height,simulation,edgepaths,edgelabels ,dragstarted ,dragged, ticked; // global variables for d graph

export default {
  name: 'App',
  data(){
	  return {
		panel: [0],  
		AlertMsg:"",
		showAlertMsg:false,
		decesionTreetreePath:"",
		decesionTreeHTML:"",
		detectedOutliers:"", 
		detectedOutliersImage:"",
		featureScores:"",
		pdpImgPaths:[],
		dialogOutlier:false,
		dialogdecesionTreepath:false,
		graphData : {},
		totalNumberNodes : 0,
		consttotalNumberNodes : 0,
		totalNumberEdges : 0,
		consttotalNumberEdges : 0,
		restrcitedNodeNumber : undefined,
		restrcitedEdgeNumber : undefined,
		ListOfEntity : [],
		selectedentityArr : [],
		restrictedNodesList : [],
		restrictedEdgesList : [],
		EntityColorLegends : {}		
		
	  }
  },
  mounted(){ 
	 this.getGrapthData();
	 
  },
  methods:{

	sortnodesbySampleSize: function(nodesArr){
		nodesArr = nodesArr.sort((a,b) => {
			return Number(b.values.sample_size)-Number(a.values.sample_size);})

		return nodesArr;
	},

	generateRandomColor : function(){
		var randomColor = Math.floor(Math.random()*16777215).toString(16);
		return ("#"+randomColor);
	},

	getListofEntityInGraph: function(nodesArr){
		var tm = this;
		var entitList = [];

		nodesArr.forEach((t) => {
			entitList.push(t.entity);
		})

		entitList = [...new Set(entitList)];

		var finalEntityList = [];
		entitList.forEach((t) => {
			finalEntityList.push({'label':t,'checked':false,'color': tm.generateRandomColor()});
		})

		
		return finalEntityList;

	},

	sortEdgesByWeight: function(linkArr){
		linkArr = linkArr.sort((a,b) => {
			return Number(b.weight)-Number(a.weight);})
		
		return linkArr;
	},
	
	getGrapthData: function(initialDisplay){
		 console.log(initialDisplay);
		 colors = d3.scaleOrdinal(d3.schemeCategory10);
		 
		 var radius = d3.scaleSqrt().range([0, 6]);
		//console.log(d3.schemeCategory10)
		 svg = d3.select("svg"),
		    width = +svg.attr("width"),
		    height = +svg.attr("height"),
			node,
			link;
		
		

		 simulation = d3.forceSimulation()
		 .force('link', d3.forceLink().id(d => (d.id)).distance(function distance(d) {
			if (d.weight < 0.3){
				return 100;
			}
			else if(d.weight >=0.3 && d.weight<= 0.7){
				return 220;
			}
			else {
				return 270;
			}
			//return 1000*d.weight;
			}))
		.force('charge', d3.forceManyBody().strength(function(d){
			if (d.values.sample_size < 50){
				return -900;
			}
			else if (d.values.sample_size < 110 & d.values.sample_size >= 50){
				return -700;
			}
			else {
				return -300;
			}
		}))
		.force('center', d3.forceCenter((width / 2)-50, (height / 2)+10));
			
		
		var vm = this;
		if (initialDisplay == undefined){
			fetch("/getXAIOutput",{
				method:"post",
				body:JSON.stringify({"req_No":sessionStorage.getItem("req_No")}),
				headers: new Headers({'content-type':'application/json'})
				}).then(function(r){return r.json()})
								.then(function(data){
									data = JSON.parse(data);
									console.log(data);
									
									
									console.log(data.Datagraph_path);
									
									// for data graph
										fetch(data.Datagraph_path,{
											  method:"get",
											  headers: new Headers({'content-type':'application/json'})
											})
												.then(function(r){return r.json()})
												.then(function(graph){
													console.log(graph);
													var nodes = vm.sortnodesbySampleSize(graph.nodes);
													var links = vm.sortEdgesByWeight(graph.links);
													vm.ListOfEntity = vm.getListofEntityInGraph(graph.nodes);
													graph = {"nodes":nodes,"links":links};
													vm.graphData = graph;
													vm.totalNumberNodes = graph.nodes.length;
													vm.consttotalNumberNodes = graph.nodes.length;
													vm.totalNumberEdges = graph.links.length;
													vm.consttotalNumberEdges = graph.links.length;
													vm.restrictedNodesList = graph.nodes;
													vm.restrictedEdgesList = graph.links;
													vm.update(graph.links, graph.nodes);
											})
									//
									// for data table
									vm.loadDataTable(data.Datagraph_data,"page-wrap");
									$(".showfeatures").click(function(){
										//alert('asdasdas')
										vm.loadDataTable(data.feature_scores,"page-wrap1");
									})

									if (sessionStorage.getItem("reloadNeeded") == "true") {
										sessionStorage.setItem("reloadNeeded", "false");
										location.reload(); 
									}

									
								})
		}						
								
		
	},
	
	restrictNodesandEdgesDisplay:function(){
		console.log(this.restrcitedNodeNumber);
		console.log(this.restrcitedEdgeNumber);

		var nodelist = [];

		if (this.selectedentityArr.length != 0){
			this.graphData.nodes.forEach((t) => {
				if(this.selectedentityArr.includes(t.entity) == true){
					nodelist.push(t);
				}
			});
		}
		else {
			nodelist = this.graphData.nodes;
		}

		var rn = [];
		if (this.restrcitedNodeNumber != undefined){
			var AllSampleSize = [];
			this.graphData.nodes.forEach((t) => {
				AllSampleSize.push(t['values']['sample_size']);})
		
			AllSampleSize = AllSampleSize.sort(function(a, b){return a - b});
			AllSampleSize = AllSampleSize.slice(0,this.restrcitedNodeNumber);
			var restricedValue = AllSampleSize[AllSampleSize.length-1];
		
			rn = nodelist.slice(0,this.restrcitedNodeNumber)
		}
		else {
			rn = nodelist;
		}

		var re = [];
		if (this.restrcitedEdgeNumber != undefined){
			var AllEdgeWeight = [];
			this.graphData.links.forEach((t) => {
				AllEdgeWeight.push(t['weight']);})
		
			AllEdgeWeight = AllEdgeWeight.sort(function(a, b){return a - b});
			AllEdgeWeight = AllEdgeWeight.slice(0,this.restrcitedEdgeNumber);
			var restricedValue = AllEdgeWeight[AllEdgeWeight.length-1];
			re = this.graphData.links.slice(0,this.restrcitedEdgeNumber);

		}
		else {
			re = this.graphData.links;
		}

		console.log("Refined Edges part 1 :",re);
		

		

		var restrcitednodesList = [];
		rn.forEach((t) => {
			restrcitednodesList.push(t.id);
		});

		console.log("Refined Nodes id",restrcitednodesList);

		var refinedEdges = [];
		re.forEach((t) => {
			//console.log(restrcitednodesList.includes(t.source.id));
			//console.log(restrcitednodesList.includes(t.target.id));
			if((restrcitednodesList.includes(t.source.id) == true) && (restrcitednodesList.includes(t.target.id) == true)){
				refinedEdges.push(t);
			}
		});
		re = refinedEdges;
		var FurtherRefinedNodes = [];
		re.forEach((t) => {
			FurtherRefinedNodes.push(t.source.id);
			FurtherRefinedNodes.push(t.target.id);
		});
		FurtherRefinedNodes = [...new Set(FurtherRefinedNodes)];
		var FinalRefinedNodes = [];
		rn.forEach((t) => {
			if (FurtherRefinedNodes.includes(t.id) == true){
				FinalRefinedNodes.push(t);
			}
		});
		rn = FinalRefinedNodes;

		console.log("Refined Nodes",rn);
		console.log("Refined Edges",re);
		this.restrictedNodesList = rn;
		this.restrictedEdgesList = re;

		document.querySelector("#grapgViewPort").innerHTML = "";
		this.getGrapthData("restricted");
		this.update(re,rn);

		
		if (this.restrcitedEdgeNumber != undefined){
			this.totalNumberEdges = this.restrcitedEdgeNumber;
		}

		if (this.restrcitedNodeNumber != undefined){
			this.totalNumberNodes = this.restrcitedNodeNumber;
		}

	},

	clearNodesandEdgesRestrcition:function(){
		// document.querySelector("#grapgViewPort").innerHTML = "";
		// console.log("All Nodes",this.graphData.nodes);
		// console.log("All Links",this.graphData.links);
		// this.update(this.graphData.links,this.graphData.nodes);
		// this.totalNumberNodes = this.consttotalNumberNodes;
		// this.totalNumberEdges = this.consttotalNumberEdges;
		location.reload();
	},
	
    update:function(links, nodes) {
		var tooltip = d3.select("body")
	.append("div")
	.attr("class", "tooltip")
	.style("opacity", 0);

        link = svg.selectAll(".link")
            .data(links)
            .enter()
            .append("line")
           // .attr("class", "link")
			.attr("class",function(d,i){
					return "link intensityLink"+d.source;
			})
			.attr("dataTargetId",function(d,i){
					return d.target;
			})
			.attr('marker-end','url(#arrowhead)')
			.style("stroke", function(d){

				// var colorVal = String((1-d.weight)*100);
				// return '#FFE'+String(colorVal);
				if (d.weight <= 0.14){
					return '#8E24AA';
				}
				else if (d.weight > 0.14 && d.weight <= 0.28 ){
					return '#303F9F';
				}
				else if (d.weight > 0.28 && d.weight <= 0.42 ){
					return '#1976D2';
				}
				else if (d.weight > 0.42 && d.weight <= 0.56 ){
					return '#00E676';
				}
				else if (d.weight > 0.56 && d.weight <= 0.70 ){
					return '#FFB300';
				}
				else if (d.weight > 0.70 && d.weight <= 0.84 ){
					return '#FF3D00';
				}
				else if (d.weight > 0.84 ){
					return '#E53935';
				}

				
				// return 'rgb('+colorVal+','+colorVal+','+colorVal+')';
			});

			var linkedByIndex = {};
  links.forEach(d => {
    linkedByIndex[`${d.source.index},${d.target.index}`] = 1;
  });

  var isConnected = function (a, b) {
    return linkedByIndex[`${a.index},${b.index}`] || linkedByIndex[`${b.index},${a.index}`] || a.index === b.index;
  }

  var fade = function (opacity) {
    return d => {
      node.style('stroke-opacity', function (o) {
        const thisOpacity = isConnected(d, o) ? 1 : opacity;
        this.setAttribute('fill-opacity', thisOpacity);
        return thisOpacity;
      });

      link.style('stroke-opacity', o => (o.source === d || o.target === d ? 1 : opacity));

    };
  }

			link  
			.attr('class', 'link')
			  .on('mouseover.tooltip', function(d) {
				  tooltip.transition()
					.duration(300)
					.style("opacity", .8);
				  tooltip.html("Source:"+ d.source.id + 
							 "<p/>Target:" + d.target.id +
							"<p/>Relation :" + d.source.id + ' ' + d.type + ' ' +d.target.id )
					.style("left", (d3.event.pageX) + "px")
					.style("top", (d3.event.pageY + 10) + "px");
				})
				.on("mouseout.tooltip", function() {
				  tooltip.transition()
					.duration(100)
					.style("opacity", 0);
				})
				  .on('mouseout.fade', fade(1))
				.on("mousemove", function() {
				  tooltip.style("left", (d3.event.pageX) + "px")
					.style("top", (d3.event.pageY + 10) + "px");
				});

        // link.append("title")
		// 	.text(function (d) {return d.type;});

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
			.attr("dataTargetId",function(d,i){
					return d.target;
			 })
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
            .attr("startOffset", "50%");
            // .text(function (d) {
			// 	return d.type
			// });

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
				return (d.values.sample_size/10);
            //   // if (Number(d.node_Aff) <= 10) {
            //   //   return 5;
            //   // }
            //   // else {
            //   //   return d.node_Aff;
            //   // }
			//   return 4;
			
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
        
		// console.log(colors);
		
		// Customize the color Randomly and create JSON for name color pair
				
				// var CharJSONAlpha = {'a' : 1,'b' : 2,'c' : 3,'d' : 4,'e' : 5,'f' : 6,'g' : 7,'h' : 8,'i' : 9,'j' : 10,'k' : 11,
				// 'l' : 12,'m' : 13,'n': 14,'o': 15,'p': 16,'q': 17,'r': 18,'s': 19,'t': 20,'u': 21,'w': 22,
				// 'x': 23,'y' : 24,'z' : 25,'v' : 26};


				console.log(vm.ListOfEntity);


				// var ents = d.entity.toLocaleLowerCase();
				// var numberSum = 0;
				// for (var p=0;p<ents.length;p++){
				// 	numberSum = numberSum + CharJSONAlpha[ents.charAt(p)];
				// }

				var nodeColor = undefined;

				for (var p=0; p<vm.ListOfEntity.length; p++){
					if (d.entity == vm.ListOfEntity[p]['label']){
						nodeColor = vm.ListOfEntity[p]['color'];
					}
				}

				
				//var returnedColor = colors((d.entity.length*CharJSONAlpha[d.entity.charAt(0)])%10);
				var returnedColor = nodeColor;
				vm.EntityColorLegends[d.entity] = returnedColor; 
				return returnedColor;
				
			})
			.attr("trackid",function(d, i){
				return d.id;
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
				
				var nodes = vm.restrictedNodesList;
				var links = vm.restrictedEdgesList;

				var clicked_Node = d.id;
				var rn = [];
				nodes.forEach((t) => {
					if (t.id == clicked_Node){
						rn.push(t);
					}
				})

				var re = [];
				links.forEach((t) => {
					if (t.source.id == clicked_Node || t.target.id == clicked_Node){
						re.push(t);
					}
				})

				var frn = [];

				re.forEach((t) => {
					frn.push(t.source.id);
					frn.push(t.target.id);
				})

				frn = [...new Set(frn)];
				console.log(frn);

				var frn2 = [];
				for(var t=0;t<frn.length;t++){
					var currentNode = frn[t];
					for (var r=0;r<nodes.length;r++){
						if(nodes[r].id == currentNode){
							frn2.push(nodes[r]);
						}
					}
				}

				console.log(frn2);

				document.querySelector("#grapgViewPort").innerHTML = "";
				vm.getGrapthData("restricted");
				vm.update(re,frn2);

			})

        node.append("title")
			.text(function (d) {
				
				// return d.values+' from Table:'+d.entity;
				return 'Entity -> '+d.entity+' : Range -> '+d.values.range+' , Sample Size -> '+String(d.values['sample_size']);
			});

       node.append("text")
            .attr("dy", -1)
			.attr("dx", function(d){
				return (d.values.sample_size/10)+2;
			})
			.attrs({
                'font-size': 15,
                'fill': 'black'
            })
            .text(function (d) {
				return d.name;
			});

        simulation
            .nodes(nodes)
            .on("tick", this.ticked);

        simulation.force("link")
			.links(links);
			
		console.log(vm.EntityColorLegends);

		var ent = Object.keys(vm.EntityColorLegends);
		var legendHTML = '<span> Node Legend </span>';

		ent.forEach((u) => {
			var preparedLegendContent = '<p class="nlpara" style="margin-bottom: 0px;"><svg height="40" width="40"><circle cx="20" cy="20" r="15" stroke="black" stroke-width="1" fill="'+vm.EntityColorLegends[u]+'" /></svg><svg height="40" width="150"><text x="10" y="25" fill="rgb(75, 73, 73)">'+u+'</text></svg></p>';
			legendHTML = legendHTML+preparedLegendContent;
		})

		document.querySelector(".nodeLegends").innerHTML = legendHTML;


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
	
	clearIntermidiateSelection : function(){
		var vm = this;
		var nodes = vm.restrictedNodesList;
		var links = vm.restrictedEdgesList;


		document.querySelector("#grapgViewPort").innerHTML = "";
		vm.getGrapthData("restricted");
		vm.update(links,nodes);
	},
	
	loadDataTable: function(filePath,divId){
		console.log(filePath+divId)
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

	remove_array_element:function(array,element){

		var index = array.indexOf(element);
		if (index > -1) {
		 array.splice(index, 1);
	 	}
		return array;
	},

	entitesSelected: function(entity){
		console.log(entity);
		if (entity.checked == true){
			this.selectedentityArr.push(entity.label);
		}
		else{
			this.selectedentityArr = this.remove_array_element(this.selectedentityArr,entity.label);
		}
		// selectedentityArr
	},

	openOutlier: function(){
        this.dialogOutlier = true;
    },
    openDecesionTreepath: function(){
        this.dialogdecesionTreepath = true;
	},
	
	
	
	
	  
  },
  
   
  template: `
  
  <v-app >
				<div class="CustomhomeBtn">
					<v-btn rounded color="orange accent-4" to="/">Home</v-btn>
			    </div>

			   <v-expansion-panels v-model="panel" multiple>
				  <v-expansion-panel>
					<v-expansion-panel-header>Data Graph and Data Table</v-expansion-panel-header>
					<v-expansion-panel-content>
				
					
					<div>
						Put Restiction on Graph Display : <i>(Showing 
						<span id="nodeNumber">{{ totalNumberNodes }}</span> nodes and <span id="edgeNumber">{{ totalNumberEdges }}</span> edges)</i>  
						<v-btn rounded color="purple lighten-4" @click="restrictNodesandEdgesDisplay();">Restrict</v-btn>
						<v-btn rounded color="pink lighten-4" @click="clearNodesandEdgesRestrcition();">Clear</v-btn>
						
						<span class="graphEntityArea" v-for="entity in ListOfEntity">
							<v-checkbox
							  
                              v-model="entity.checked"
							  :label="entity.label"
							  @change="entitesSelected(entity);"
                            ></v-checkbox>
						
						</span>

					  <v-text-field
						label="Number of Nodes"
						hide-details="auto"
						v-model="restrcitedNodeNumber"
					  ></v-text-field>
					  <v-text-field label="No of Edges" v-model="restrcitedEdgeNumber"></v-text-field>
					</div>
					
				  
					  
					
				<div style="padding-bottom:100px;">
					<svg id="grapgViewPort" width="1250" height="1000" style="width:100%; margin-top:15px;"></svg>
				</div>
				<div class="container">
				<div class="row">
					<div class="edgeLegends col-sm-6"> <span>Edge Legend</span>
					<p>
						<svg height="10" width="40">
							<line x1="0" y1="5" x2="35" y2="5" style="stroke:#8E24AA;stroke-width:2" />
					  	</svg>Edge weight less or equal to 0.14 </p>
                        <p>
						<svg height="10" width="40">
							<line x1="0" y1="5" x2="35" y2="5" style="stroke:#303F9F;stroke-width:2" />
						</svg>Edge weight > 0.14 but <=0.28 </p>
                        <p>
						<svg height="10" width="40">
							<line x1="0" y1="5" x2="35" y2="5" style="stroke:#1976D2;stroke-width:2" />
						</svg>Edge weight > 0.28 but <=0.42 </p>
                        <p>
						<svg height="10" width="40">
							<line x1="0" y1="5" x2="35" y2="5" style="stroke:#00E676;stroke-width:2" />
						</svg>Edge weight > 0.42 but <=0.56 </p>
                        <p>
						<svg height="10" width="40">
							<line x1="0" y1="5" x2="35" y2="5" style="stroke:#FFB300;stroke-width:2" />
						</svg>Edge weight > 0.56 but <=0.70</p>
                        <p>
						<svg height="10" width="40">
							<line x1="0" y1="5" x2="35" y2="5" style="stroke:#FF3D00;stroke-width:2" />
						</svg>Edge weight > 0.70 but <=0.84</p>
                        <p>
						<svg height="10" width="40">
							<line x1="0" y1="5" x2="35" y2="5" style="stroke:#E53935;stroke-width:2" />
						</svg>Edge weight greater than 0.84</p>
					</div>
					<div class="nodeLegends col-sm-6"> <span> Node Legend </span>
						
					</div>
				</div>
				</div>
				
				<v-btn id="ClearSelection" rounded color="deep-orange accent-1" @click="clearIntermidiateSelection();">Clear Selection</v-btn>
				<div id="page-wrap">
				</div>
						
				</v-expansion-panel-content>
				  </v-expansion-panel>
				  
				  
				</v-expansion-panels>			
						
   
   </v-app>
  
  `,
};