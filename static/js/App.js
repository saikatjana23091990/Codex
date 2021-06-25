// src/App.js
export default {
  name: 'App',
  
  template: `
  
   <v-app >
		<v-card >
		<v-toolbar :color="'deep-purple accent-4'">
		 

		  <v-toolbar-title>
		   Test title
		  </v-toolbar-title>

		  <v-spacer></v-spacer>

		  <v-scale-transition>
		   
		  </v-scale-transition>
		 
		</v-card>
		
	<router-view/>
	
   </v-app >
   `
};