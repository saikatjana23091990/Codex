// src/App.js
export default {
  name: 'App',
  
  template: `
  
   <v-app >
		<v-card >
		<v-toolbar :color="'blue-grey darken-2'">
		 

		  <v-toolbar-title>
		   Welcome to C.O.D.E.X
		  </v-toolbar-title>

		  <v-spacer></v-spacer>

		  <v-scale-transition>
		   
		  </v-scale-transition>
		 
		</v-card>
		
	<router-view/>
	
   </v-app >
   `
};