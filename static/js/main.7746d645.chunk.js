(this["webpackJsonpsimple-reactjs-tensorflow"]=this["webpackJsonpsimple-reactjs-tensorflow"]||[]).push([[0],[,,,,,,,,,,,,,,function(e,a,t){e.exports=t.p+"static/media/logo.5d5d9eef.svg"},,function(e,a,t){e.exports=t(36)},,,,,function(e,a,t){},,,,,,,function(e,a){},function(e,a){},function(e,a){},function(e,a){},function(e,a){},function(e,a){},function(e,a,t){},function(e,a,t){},function(e,a,t){"use strict";t.r(a);var n=t(1),r=t.n(n),c=t(12),l=t.n(c),o=(t(21),t(7)),i=t(15),s=t(2),m=t(9),u=t(13),d=t.n(u),p=t(3),f=(t(34),function(){var e=Object(n.useState)([{x:-1,y:-3},{x:0,y:-1},{x:1,y:1},{x:2,y:3},{x:3,y:5},{x:4,y:7}]),a=Object(m.a)(e,2),t=a[0],c=a[1],l=Object(n.useState)({model:null,trained:!1,predictedValue:"Click on train!",valueToPredict:1}),u=Object(m.a)(l,2),f=u[0],b=u[1],v=function(e){var a=d()(t,Object(s.a)({},e.target.dataset.index,Object(s.a)({},e.target.name,{$set:parseInt(e.target.value)})));c(a)};return r.a.createElement("div",{className:"tensorflow-example"},r.a.createElement("div",{className:"train-controls"},r.a.createElement("h2",{className:"section"},"Training Data (x,y) pairs"),r.a.createElement("div",{className:"row labels"},r.a.createElement("div",{className:"field-label column"},"X"),r.a.createElement("div",{className:"field-label column"},"Y")),t.map((function(e,a){return r.a.createElement("div",{key:a,className:"row"},r.a.createElement("input",{className:"field field-x column",value:e.x,name:"x","data-index":a,onChange:v,type:"number",pattern:"[0-9]*"}),r.a.createElement("input",{className:"field field-y column",value:e.y,name:"y","data-index":a,onChange:v,type:"number"}))})),r.a.createElement("button",{className:"button-add-example button--green",onClick:function(){c([].concat(Object(i.a)(t),[{x:1,y:1}]))}},"+"),r.a.createElement("button",{className:"button-train button--green",onClick:function(){var e=[],a=[];t.forEach((function(t,n){e.push(t.x),a.push(t.y)}));var n=p.b();n.add(p.a.dense({units:1,inputShape:[1]})),n.compile({loss:"meanSquaredError",optimizer:"sgd"});var r=p.c(e,[e.length,1]),c=p.c(a,[a.length,1]);n.fit(r,c,{epochs:250}).then((function(){b(Object(o.a)({},f,{model:n,trained:!0,predictedValue:"Ready for making predictions"}))}))}},"Train")),r.a.createElement("div",{className:"predict-controls"},r.a.createElement("h2",{className:"section"},"Predicting"),r.a.createElement("input",{className:"field element",value:f.valueToPredict,name:"valueToPredict",onChange:function(e){return b(Object(o.a)({},f,Object(s.a)({},e.target.name,[parseInt(e.target.value)])))},type:"number",placeholder:"Enter an integer number"}),r.a.createElement("br",null),r.a.createElement("div",{className:"element"},f.predictedValue),r.a.createElement("button",{className:"element button--green",onClick:function(){var e=f.model.predict(p.c([f.valueToPredict],[1,1])).arraySync()[0][0];b(Object(o.a)({},f,{predictedValue:e}))},disabled:!f.trained},"Predict")))}),b=t(14),v=t.n(b);t(35);var E=function(){return r.a.createElement("div",{className:"App"},r.a.createElement("header",{className:"App-header"},r.a.createElement("img",{src:v.a,className:"App-logo",alt:"logo"}),r.a.createElement("p",null,"Edit ",r.a.createElement("code",null,"src/App.js")," and save to reload."),r.a.createElement("a",{className:"App-link",href:"https://reactjs.org",target:"_blank",rel:"noopener noreferrer"},"Learn React")),r.a.createElement(f,null))};Boolean("localhost"===window.location.hostname||"[::1]"===window.location.hostname||window.location.hostname.match(/^127(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$/));l.a.render(r.a.createElement(E,null),document.getElementById("root")),"serviceWorker"in navigator&&navigator.serviceWorker.ready.then((function(e){e.unregister()}))}],[[16,1,2]]]);
//# sourceMappingURL=main.7746d645.chunk.js.map