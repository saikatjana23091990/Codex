import App from './App.js';
import Home from './components/Home.js';
import Export from './components/Export.js';

Vue.use(Vuetify);

Vue.use(VueRouter);

const routes = [{
		path: '/',
		component: Home
	},
	{
		path: '/export',
		component: Export
	}
]

const router = new VueRouter({
	routes // short for `routes: routes`
})
new Vue({
  el: '#app',
  vuetify: new Vuetify(),
  router,
  components: { App },
  template: '<App/>'
})