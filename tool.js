var fs = require('fs');
var util = require('./util.js');

function forEachModel(fn) {
    fs.readdirSync('./models').forEach(function(dir) {
        fs.readdirSync('./models/'+dir).forEach(function(model) {
            var path = './models/'+dir+'/'+model;
            fn && fn(path, dir);
        });
    });
}

var tools = {
    bestmodel: function() {
        var bestErr = 10e8, bestPath = '';
        forEachModel(function (path, name) {
            var model = JSON.parse(fs.readFileSync(path, 'ascii'));
            console.log(path, model.errorRate);
            if (bestErr > model.errorRate) {
                bestErr = model.errorRate;
                bestPath = path;
            }        
        });
        console.log('------------------------');
        console.log('best error', bestErr);
        console.log('best model', bestPath);
    },
    fixnames: function() {
        forEachModel(function (path, name) {
            var model = JSON.parse(fs.readFileSync(path, 'ascii'));
            model.name = name;
            fs.writeFileSync(path, JSON.stringify(model));
        });
    },
    prune: function() {
        forEachModel(function (path, name) {
            
        });
    }
}

var name = process.argv[2];
if (name && tools[name]) { tools[name](); }
