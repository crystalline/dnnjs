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
    
    var model = new nn.MLP([{size: trainDataMnist.rows*trainDataMnist.cols},
                         {size: 50,
                          activation: 'relu'},
                         {size: 35,
                         activation: 'relu'},
                         {size: 20,
                         activation: 'relu'},
                         {size: 10,
                         activation: 'tanh'}],
                         random);
    
    model.alpha = 0.01;
    return nn.trainClassifier({
        model: model,
        trainData: trainDataMnist,
        testData: testDataMnist,
        numClasses: 10,
        numEpochs: 100,
        labelTrue: 1,
        labelFalse: -1,
        alphaDecay: 0.92,
        saveParams: true,
        random: random
    });
}

experiment();

