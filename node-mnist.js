var fs = require('fs');
var util = require('./util.js');
var nn = require('./nn.js');

var prng = new util.prng(81237);
var random = function () { return prng.next() };

var data = nn.loadMnistTrain();
nn.scaleDataSet(data, 2/255, -1);

splitData = nn.splitDatasetTrainTest(data, 0.1, random);
var trainDataMnist = splitData.train;
var testDataMnist = splitData.test;

function experiment() {
    
    if (process.argv.length == 2) {
        
        var model = new nn.MLP({
            layers: [{size: trainDataMnist.rows*trainDataMnist.cols},
                     {size: 100,
                      activation: 'relu'},
                     {size: 100,
                     activation: 'relu'},
                     {size: 10,
                     activation: 'tanh'}],
            optimizer: 'momentum',
            alpha: 0.0005,
            theta: 0.7,
            batchSize: 8,
            random: random
        });
    } else {
        
        var params = JSON.parse(fs.readFileSync(process.argv[2], 'ascii'));
        
        if (process.argv[3]) {
            var res = {};
            try {
                res = eval('('+process.argv[3]+')');
            } catch (e) { console.log(e) }
            util.simpleExtend(params, res);            
        }
        
        var model = new nn.MLP(params);
    }
    
    var N = (model.epoch||0)+500;    
        
    console.log('alpha: ', model.alpha);
    console.log('start ep', (model.epoch||0), 'end ep', N);

    return nn.trainClassifier({
        model: model,
        trainData: trainDataMnist,
        testData: testDataMnist,
        numClasses: 10,
        numEpochs: N,
        labelTrue: 1,
        labelFalse: -1,
        alphaDecay: 0.95,
        saveParams: true,
        random: random
    });
}

experiment();

