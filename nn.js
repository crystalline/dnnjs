
try {
    fs = require('fs');
    rle = require('./rle.js');
} catch (e) {}

//Dataset load/store

function loadLabels(data) {
    var magic = data.readUInt32BE(0),
        num = data.readUInt32BE(4),
        labels = new Uint8Array(num);
    
    for (var i = 0; i < num; i++) {
        labels[i] = data.readUInt8(8+i);
    }
    
    return labels;   
}

function loadImages(data) {
    var magic = data.readUInt32BE(0),
        num = data.readUInt32BE(4),
        rows = data.readUInt32BE(8),
        cols = data.readUInt32BE(12),
        images = new Array(rows*cols);
        
    
    console.log('reading idx file '+'magic: '+magic);
    console.log(num+' images '+rows+'x'+cols+' px');
    
    for (var i = 0; i < num; i++) {
        var baseOffset = 16+i*rows*cols;
        var img = new Uint8Array(rows*cols);
        for (var j = 0; j < rows*cols; j++) {
            img[j] = data.readUInt8(baseOffset+j);
        }
        images[i] = img;
    }
    
    return { magic: magic, num: num, rows: rows, cols: cols, images: images };
}

function loadMnist(imageFile, labelFile) {
    var res = loadImages(imageFile);
    res.labels = loadLabels(labelFile);
    return res;
}

function decodeB64rle(string) {
    var locs = util.locations("\"", string);
    if (locs.length == 2) {
        var binary = new Buffer(string.substring(locs[0]+1, locs[1]), 'base64');
        return new Buffer(rle.decode(binary));
    }
}

function loadMnistTrain() {
    var imagePath = './datasets/train-images.idx3-ubyte-b64-rle.js';
    var labelPath = './datasets/train-labels.idx1-ubyte-b64-rle.js';
    return loadMnist(
        decodeB64rle(fs.readFileSync(imagePath, 'ascii')),
        decodeB64rle(fs.readFileSync(labelPath, 'ascii'))
    );
}

function loadMnistT10K() {
    var imagePath = './datasets/t10k-images.idx3-ubyte-b64-rle.js';
    var labelPath = './datasets/t10k-labels.idx1-ubyte-b64-rle.js';
    return loadMnist(
        decodeB64rle(fs.readFileSync(imagePath, 'ascii')),
        decodeB64rle(fs.readFileSync(labelPath, 'ascii'))
    );
}


function printPattern(pattern, data) {
    str = '';
    var mean = arrayStat(pattern).mean;
    for (var i = 0; i < data.rows; i++) {
        for (var j = 0; j < data.cols; j++) {
            var val = pattern[i*data.cols+j];
            str += (val > mean ? 'X' : ' ');
        }
        str += '\n';
    }
    console.log(str);
}

function print2dArray(arr, w) {
    var m = maxmin(arr);
    var mean = arrayStat(pattern).mean;
    for (var i = 0; i < arr.length; i++) {
        if (i % w == 0) {
            str += '\n';
        } else {
            var val = arr[i];
            str += (val > mean ? 'X' : ' ');
        }
    }
    console.log(str);
}

//Array/matrix utilities

function maxmin(arr) {
    if (!arr || !arr.length) { console.log('maxmin: not an array'); return; }
    var i = 0;
    var amax = -Infinity;
    var amaxi = false;
    var amin = +Infinity;
    var amini = false;
    var val;
    var sum = 0;
    for (i=0; i<arr.length; i++) {
        val = arr[i];
        sum += val;
        if (val > amax) { amax = val; amaxi = i; }
        if (val < amin) { amin = val; amini = i; }
    }
    return {max: amax, maxindex: amaxi, min: amin, minindex: amini, mean: sum/arr.length};
}

var arrayStat = maxmin;

function makeVector(n, init) {
    var arr = new Float32Array(n);
    if (typeof init == 'function') for (var i=0; i<n; i++) { arr[i] = init(i); }
    if (typeof init == 'number') for (var i=0; i<n; i++) { arr[i] = init; }
    arr.zero = function(x) { x = x||0; var i=0; while (i<n) { this[i] = x; i++ } };
    return arr;
}

function copyVector(src, dst) {
    dst = dst || new Float32Array(src.length);
    var i;
    for (i=0; i<src.length; i++) { dst[i] = src[i] };
    dst.length = src.length;
    return dst;
}

function mget(i,j) {
    return this[i*this.w+j];
}

function mset(i,j,val) {
    this[i*this.w+j] = val;
}

function makeMatrix(w, h, init) {
    var arr = new Float32Array(w*h);
    arr.w = w;
    arr.h = h;
    arr.get = mget;
    arr.set = mset;
    arr.zero = function() { var i=0; while (i<this.w*this.h) { this.arr[i] = 0; i++ } };
    if (init) {
        var i;
        var m = arr;
        for (i=0; i<w*h; i++) {
            if (typeof init == 'number') { m[i] = init; }
            else if (typeof init == 'function') { m[i] = init(Math.floor(i/w), i%w); }
            else if (init.length) { m[i] = init[i]; }
        }
    }
    return arr;
}

function matXvec(mat, vec, result, resultIndex) {
    resultIndex = resultIndex || 0;
    result = result || makeVector(mat.h, 0);
    var i;
    var W = mat.w;
    var H = mat.h;
    for (i=0; i<H; i++) {
        var base = W*i;
        var acc = 0;
        for (j=0; j<W; j++) {
            acc += vec[j] * mat[base++];
        }
        result[i+resultIndex] = acc;
    }
    return result;
}

function mapVec(src, fun, result, resultIndex) {
    var N = src.length;
    resultIndex = resultIndex || 0;
    result = result || makeVector(N, 0);
    var i;
    var stopIndex = N+resultIndex;
    for (i=resultIndex; i<stopIndex; i++) {
        result[i] = fun(src[i]);
    }
}

function mapVec2(srcA, srcB, fun, result, resultIndex) {
    if (srcA.length != srcB.length) {
        console.log('mapVec2 vec length error '+srcA.length+' != '+srcB.length);
        throw new Error();
        return;
    }
    var N = srcA.length;
    resultIndex = resultIndex || 0;
    result = result || makeVector(N, 0);
    var i;
    var stopIndex = N+resultIndex;
    for (i=resultIndex; i<stopIndex; i++) {
        result[i] = fun(srcA[i], srcB[i]);
    }
}

function reduceVec(vec, fun, start) {
    if (start == undefined) start = 0;
    var acc = start;
    var i;
    for (i=0; i<vec.length; i++) {
        acc = fun(acc, vec[i]);
    }
    return acc;
}

function plus(a,b) { return a+b }
function minus(a,b) { return a-b }
function mul(a,b) { return a*b }

function addVec(srcA, srcB, result, resultIndex) {
    var N = srcA.length;
    resultIndex = resultIndex || 0;
    result = result || makeVector(N, 0);
    var i;
    for (i=resultIndex; i<N+resultIndex; i++) {
        result[i] = srcA[i] + srcB[i];
    }
    return result;
}

function subVec(srcA, srcB, result, resultIndex) {
    mapVec2(srcA, srcB, minus, result, resultIndex);
}

function mulVec(srcA, srcB, result, resultIndex) {
    mapVec2(srcA, srcB, mul, result, resultIndex);
}

function loadVecMat(srcObj) {
    if (srcObj.length) {
        var arr = makeVector(srcObj.length, 0);
        var N = srcObj.length;
    } else if (srcObj.w && srcObj.h) {
        var arr = makeMatrix(srcObj.w, srcObj.h, 0);
        var N = srcObj.w*srcObj.h;
    }
    var i;
    for (i=0; i<N; i++) {
        arr[i] = srcObj[i];
    }
    return arr;
}

function initWeights(random, wmax, wmin) {
    random = random || Math.random;
    wmax = wmax || 0.05;
    wmin = wmin || -wmax;
    return function () {
        return random()*(wmax-wmin)+wmin;
    }
}

function sigmoid(x) {
    return Math.tanh(x);
}

function ddxsigmoid(x) {
    var t = Math.tanh(x);
    return 1-t*t;
}

function L2distanceSquare(vecA, vecB) {
    if (vecA.length != vecB.length) return;
    var i=0, acc=0;
    for (i=0; i<vecA.length; i++) {
        var d = vecA[i] - vecB[i];
        acc += d*d
    }
    return acc;
}

function L2distance(vecA, vecB) {
    return Math.sqrt(L2distanceSquare(vecA, vecB));
}

function convolve2d(image, kernel) {
    if (kernel.w % 2 == 0 || kernel.h % 2 == 0) {
        console.log("even kernel dimensions, aborting");
        return;
    }
    
    var imW = image.w;
    var imH = image.h;
    var kW = kernel.w;
    var kH = kernel.h;
    var kWhalf = Math.round(kernel.w-1/2);
    var kHhalf = Math.round(kernel.h-1/2);
    
    var i,j;
    
}

function normal() {
    var sum = 0;
    sum += Math.random() - 0.5;
    sum += Math.random() - 0.5;
    sum += Math.random() - 0.5;
    return sum;
}

function softmax(src, dst) {
    var i;
    var sumexp = 0;
    for (i=0; i<vec.length; i++) { sumexp += Math.exp(vec[i]); }
    for (i=0; i<vec.length; i++) { dst[i] = Math.exp(vec[i])/sumexp; }
    return dst;
}

//Single layer perceptron

function SLP(inDim, outDim, random) {
    this.inDim = inDim;
    this.outDim = outDim;
    this.alpha = 0.01;
    this.decay = 0;
    this.W = makeMatrix(inDim, outDim, initWeights(random));
    this.b = makeVector(outDim, initWeights(random));
    this.out = makeVector(outDim);
    this.temp = makeVector(outDim);
    this.outg = makeVector(outDim);
}

SLP.prototype.forward = function(invec) {
    if (invec.length == this.inDim) {
        
        matXvec(this.W, invec, this.temp, 0);
        addVec(this.temp, this.b, this.temp, 0);
        mapVec(this.temp, sigmoid, this.out, 0);
        return this.out;
        
        
        /*
        for (var l=0; l<this.outDim; l++) {
            var base = this.inDim*l;
            var acc = 0;
            for (var k=0; k<this.inDim; k++) {
                acc += this.W[base+k] * invec[k];
            }
            acc += this.b[l];
            this.temp[l] = acc;
            this.out[l] = sigmoid(acc);
        }
        */
        return this.out;
    }
}

SLP.prototype.train = function(invec, outvec, verbose) { 
    if (invec.length == this.inDim && outvec.length == this.outDim) {
        this.forward(invec);
        mapVec(this.temp, ddxsigmoid, this.outg, 0);
        
        if (verbose) console.log('Err: '+L2distance(outvec, this.out));
        
        var alpha = this.alpha;
        var decay = this.decay;
        var l,k;
        for (l=0; l<this.outDim; l++) {
            var diff = (outvec[l] - this.out[l]);
            var outg = this.outg[l];
            var base = this.inDim*l;
            for (k=0; k<this.inDim; k++) {
                var w = this.W[base+k];
                this.W[base+k] += ((diff * outg * invec[k] * alpha) - decay*w);
            }
            var b = this.b[l];
            this.b[l] += ((diff * outg * alpha) - decay*b);
        }
    }
}

//Multilayer perceptron

function tanh(x) {
    var e = Math.exp(-2.0*x);
    return (1-e)/(1+e);
}

var sigmoids = {
    tanh: {
        fn: tanh,
        dx: function (x) {
            var t = tanh(x);
            return 1-t*t;
        }
    },
    invabs: {
        fn: function (x) { return x/(1+Math.abs(x)) },
        dx: function (x) { var t = 1+Math.abs(x); return 1/(t*t) }
    },
    relu: {
        fn: function (x) { return x*(x>0.0); },
        dx: function (x) { return 1.0*(x>0.0); }
    },
    leakyrelu: {
        fn: function (x) { return x*(x>0.0); },
        dx: function (x) { return 0.995*(x>0.0)+0.005; }
    }
}

function testSigmoids() {
    var N = 1000;
    var delta = 1e-4;
    var range = 10;
    var i;
    var report = {};
    for (key in sigmoids) {
        var fn = sigmoids[key].fn;
        var dx = sigmoids[key].dx;
        var avgErr = 0;
        var maxErr = 0;
        for (i=0; i<N; i++) {
           var x = (Math.random()-0.5)*2*range;
           var ngrad = (fn(x+delta)-fn(x-delta))/(2*delta);
           var grad = dx(x);
           var diff = Math.abs(ngrad-grad);
           maxErr = Math.max(maxErr, diff);
           avgErr += diff;
        }
        report[key] = {maxError: maxErr, meanError: avgErr/N};
    }
    return report;
}

function MLP(layers, random) {
    var paramsInited = false;
    if (typeof layers == 'string') {
        this.loadParams(layers);
        paramsInited = true;
    } else {
        this.layers = layers;
        this.weights = [];
        this.biases = [];
    }
    //this.wspeed = [];
    //this.bspeed = [];
    this.temp = [];
    this.gradtemp = [];
    this.error = [];
    this.alpha = 0.01;
    this.theta = 0.5;
    
    if (this.layers.length < 2) return;
    
    var i;
    for (i=0; i<this.layers.length-1; i++) {
        if (!this.layers[i].activation) this.layers[i].activation = 'tanh';
        var w = this.layers[i].size, h = this.layers[i+1].size;
        if (!paramsInited) {
            this.weights.push(makeMatrix(w, h, this.layers[i].initw || initWeights(random)));
            //this.wspeed.push(makeMatrix(w, h, 0));
            this.biases.push(makeVector(h, this.layers[i].initb || initWeights(random)));
            //this.bspeed.push(makeVector(h, 0));
        }
        this.temp.push(makeVector(h));
        this.gradtemp.push(makeVector(h));
        this.error.push(makeVector(h));
    }
}

MLP.prototype.forward = function(invec) {
    if (invec.length != this.layers[0].size) {
        console.log('MLP invec len error'+invec.length+' != '+this.layers[0].size);
        return false;
    }
    
    matXvec(this.weights[0], invec, this.temp[0], 0);
    addVec(this.temp[0], this.biases[0], this.temp[0], 0);
    mapVec(this.temp[0], sigmoids[this.layers[1].activation].dx, this.gradtemp[0], 0);
    mapVec(this.temp[0], sigmoids[this.layers[1].activation].fn, this.temp[0], 0);
    
    if (this.layers.length == 2) return this.temp[0];
    
    for (i=1; i<this.layers.length-1; i++) {
        matXvec(this.weights[i], this.temp[i-1], this.temp[i], 0);
        addVec(this.temp[i], this.biases[i], this.temp[i], 0);
        mapVec(this.temp[i], sigmoids[this.layers[i+1].activation].dx, this.gradtemp[i], 0);
        mapVec(this.temp[i], sigmoids[this.layers[i+1].activation].fn, this.temp[i], 0);
    }
    
    var out = this.temp[this.layers.length-2];
    return out;
};

MLP.prototype.backward = function(invec, outvec) {
    
    var untrainedOutVec = this.temp[this.temp.length-1];
    var i,k,l;
    
    var outError = this.error[this.error.length-1];
    var outGrad = this.gradtemp[this.gradtemp.length-1];
    
    subVec(outvec, untrainedOutVec, outError, 0);
    
    for (k=0; k<outError.length; k++) {
        outError[k] *= outGrad[k];
    }
            
    for (i=this.weights.length-1; i>0; i--) {
        var weights = this.weights[i];
        var error = this.error[i];
        var prevError = this.error[i-1];
        var prevGrad = this.gradtemp[i-1];
        
        var W = weights.w;
        var H = weights.h;
        for (k=0; k<W; k++) {
            var acc = 0;
            for (l=0; l<H; l++) {
                acc += error[l] * weights[W*l+k];
            }
            prevError[k] = prevGrad[k]*acc;
        }
        /*
        for (k=0; k<W; k++) {
            prevError[k] = 0;
        }
        for (l=0; l<H; l++) {
            var err = error[l];
            var index = W*l;
            for (k=0; k<W; k++) {
                prevError[k] += err * weights[index++];
            }
        }
        for (k=0; k<W; k++) {
            prevError[k] *= prevGrad[k];
        }
        */
    }
};

MLP.prototype.update = function(example) {
    var alpha = this.alpha;
    var theta = this.theta;
    var i,l;
    
    for (i=this.weights.length-1; i>=0; i--) {
        var weights = this.weights[i];
        var biases = this.biases[i];
        //var wspeed = this.wspeed[i];
        //var bspeed = this.bspeed[i];
        var gradtemp = this.gradtemp[i];
        var error = this.error[i];
        
        var input;
        if (i == 0) { input = example }
        else { input = this.temp[i-1] }
        
        var W = weights.w;
        var H = weights.h;
        
        for (l=0; l<H; l++) {
            var base = l*W;
            var err = error[l];
            var egrad = err * alpha;
            var stop = base+W;
            
            for (k=0; k<W; k++) {
                var index = base+k;
                //wspeed[index] = theta * wspeed[index] - egrad * input[k];
                //weights[index] += wspeed[index];
                weights[index] += egrad * input[k]
            }
            //bspeed[l] += theta * bspeed[l] + egrad;
            //biases[l] += bspeed[l];
            biases[l] += egrad;
        }
    }
};

MLP.prototype.train = function(example, correctOutput) {
    this.forward(example);
    this.backward(example, correctOutput);
    this.update(example);
};

MLP.prototype.gradcheck = function(example, correctOutput, paramAddr) {
    this.forward(example);
    this.backward(example, correctOutput);
    
    console.log("Checking model param gradient ", JSON.stringify(paramAddr));
    var error = this.error[paramAddr.layer];
    var input = paramAddr.layer == 0 ? example : this.temp[paramAddr.layer-1];
    var algError = -error[paramAddr.row]*input[paramAddr.column];
    console.log("Algorithmic: ", algError);
    
    var weights = this.weights[paramAddr.layer];
    var index = weights.w*paramAddr.row+paramAddr.column;
    var tempW = weights[index];
    var dx = 0.0001;
    
    weights[index] -= dx;
    var e1 = 0.5*L2distanceSquare(this.forward(example), correctOutput);
    weights[index] = tempW + dx;
    var e2 = 0.5*L2distanceSquare(this.forward(example), correctOutput);
    weights[index] = tempW;
    var numError = (e2-e1)*0.5/dx;
    console.log("Numeric: ", numError);
    
    return Math.abs(numError-algError/((numError+algError)*0.5));
};

MLP.prototype.saveParams = function(extraParams) {
    var params = {
        layers: this.layers,
        weights: this.weights,
        biases: this.biases.map(function(b) {
            return copyVector(b,{});
        }),
        alpha: this.alpha
    }
    extraParams = extraParams || {};
    util.simpleExtend(params, extraParams);
    return JSON.stringify(params);
};

MLP.prototype.loadParams = function(jsonParams) {
    var params = JSON.parse(jsonParams);
    util.simpleExtend(this, params);
    this.layers = params.layers;
    this.weights = params.weights.map(loadVecMat);
    this.biases = params.biases.map(loadVecMat);
};

MLP.prototype.getName = function() {
    if (this.name) { return this.name };
    var res = ''
    this.layers.forEach(function(layer) {
        res += layer.size+'_';
    });
    return res;
}

MLP.prototype.visualize = function() {
    var i;
    for (i=0; i<this.weights.length; i++) {
        showPattern(this.weights[i], {rows: this.weights[i].w, cols: this.weights[i].h});  
    }
}

//Classification, error measurement and model training utilites

function makeBinClassifier(model, thrs, label1, label0) {
    return function(input) {
        var out = model.forward(input);
        if (out[0] > thrs) {
            return label1;
        } else {
            return label0;
        }
    };
}

function makeMultiClassifier(model) {
    return function(input) {
        var out = model.forward(input);
        var stat = maxmin(out);
        var label = stat.maxindex;
        return label;
    };
}

function filterByLabel(dataset, labelsToExtract) {
    images = dataset.images;
    labels = dataset.labels;
    var i;
    var lfilter = {};
    for (i=0; i<labelsToExtract.length; i++) {
        lfilter[labelsToExtract[i]] = true;
    }
    var res = {images: [], labels: [], num: 0, rows: dataset.rows, cols: dataset.cols};
    for (i=0; i<dataset.num; i++) {
        if (lfilter[labels[i]]) {
            res.images.push(images[i]);
            res.labels.push(labels[i]);
            res.num++;
        }
    }
    return res;
}

function scaleDataSet(dataset, k, bias) {
    var i;
    for (i=0; i<dataset.num; i++) {
        var img = dataset.images[i];
        var newimg = new Float32Array(img.length);
        for (j=0; j<img.length; j++) {
            newimg[j] = img[j]*k+bias;
        }
        dataset.images[i] = newimg;
    }
}

function splitDatasetTrainTest(dataset, testDatasetSize, random) {
    random = random || Math.random;
    function swap(i,j) {
        var temp = dataset.images[i];
        dataset.images[i] = dataset.images[j];
        dataset.images[j] = temp;
        temp = dataset.labels[i];
        dataset.labels[i] = dataset.labels[j];
        dataset.labels[j] = temp;
    }
    util.abstractShuffle(dataset.num, swap, random);
    
    var limit = Math.floor(dataset.num*testDatasetSize);
    var testd = util.simpleExtend({}, dataset);
    testd.images = [];
    testd.labels = [];
    testd.num = limit;
    var traind = util.simpleExtend({}, dataset);
    traind.images = [];
    traind.labels = [];
    traind.num = dataset.num-limit;
    
    for (var i=0; i<limit; i++) {
        testd.images.push(dataset.images[i]);
        testd.labels.push(dataset.labels[i]);
    }
    for (var i=limit; i<dataset.num; i++) {
        traind.images.push(dataset.images[i]);
        traind.labels.push(dataset.labels[i]);
    }
    return {test: testd, train: traind};
}

function measureErrorOnDataset(dataset, classifier) {
    var i;
    var errors = 0;
    var err = [];
    for (i=0; i<dataset.num; i++) {
        if (classifier(dataset.images[i]) != dataset.labels[i]) {
            errors++;
            err.push(i);
        }
    }
    return {errorRate: errors/dataset.num, errors: err};
}

function trainClassifier(config) {
    var data = config.trainData;
    var model = config.model;
    console.log('Starting training of model '+(model.getName && model.getName()));
    var numClasses = config.numClasses;
    var numEpochs = config.numEpochs;
    var labelTrue = config.labelTrue || 1;
    var labelFalse = config.labelFalse || -1;
    var random = config.random || Math.random;
    //console.log('Model: ',model);
    //console.log('Data: ',data);
    if (config.saveParams === true) {
        var models = __dirname + '/models/';
        var modelDir = models + (model.name || (model.getName()+Date.now()));
        if (!fs.existsSync(models)) { fs.mkdirSync(models); }
        if (!fs.existsSync(modelDir)) { fs.mkdirSync(modelDir); }
        config.saveParams = function(params, epoch) {
            var dir = modelDir+'/e'+epoch+'.json';
            fs.writeFileSync(dir, params);
        }
    }
    var trainer = makeVector(numClasses, 0);
    var i,j;
    var classifier = makeMultiClassifier(model);
    var minError = 1e8;
    if (typeof model.errorRate == 'number') { minError = model.errorRate; }
    var bestModelParams = false;
    if (model.epoch) { j=model.epoch + 1 } else { j=0 }
    for (; j<numEpochs; j++) {
        if (config.alphaDecay) {
            model.alpha *= 0.95;
        }
        console.log("Epoch ",j);
        var t1 = Date.now();
        for (i=0; i<data.num; i++) {
            var index = Math.floor(random()*data.num);
            var img = data.images[index];
            var label = data.labels[index];
            trainer.zero(labelFalse);
            trainer[label] = labelTrue;
            model.train(img, trainer);
        }
        var t2 = (Date.now()-t1)/1000;
        var erate = measureErrorOnDataset(config.testData, classifier).errorRate;
        console.log('Error rate:', erate, '\nepoch time:', t2, 'sec', '\nt per example:', t2/data.num, 'sec');
        if (erate < minError && model.saveParams) {
            bestModelParams = model.saveParams({epoch: j, errorRate: erate});
            if (typeof config.saveParams == 'function') try {
                config.saveParams(bestModelParams, j);
            } catch (e) { console.log('saveException', e) }
        }
    }
    try{
        if (model.visualize) { model.visualize() };
    } catch (e) {};
    return bestModelParams;
}

function runTests() {
    console.log(testSigmoids());
}

var nn = {
    loadLabels: loadLabels,
    loadImages: loadImages,
    loadMnist: loadMnist,
    loadMnistTrain: loadMnistTrain,
    loadMnistT10K: loadMnistT10K,
    printPattern: printPattern,
    print2dArray: print2dArray,
    maxmin: maxmin,
    makeVector: makeVector,
    mget: mget,
    mset: mset,
    makeMatrix: makeMatrix,
    matXvec: matXvec,
    mapVec: mapVec,
    mapVec2: mapVec2,
    reduceVec: reduceVec,
    plus: plus,
    minus: minus,
    mul: mul,
    addVec: addVec,
    subVec: subVec,
    mulVec: mulVec,
    initWeights: initWeights,
    L2distanceSquare: L2distanceSquare,
    L2distance: L2distance,
    normal: normal,
    SLP: SLP,
    MLP: MLP,
    makeBinClassifier: makeBinClassifier,
    makeMultiClassifier: makeMultiClassifier,
    filterByLabel: filterByLabel,
    scaleDataSet: scaleDataSet,
    splitDatasetTrainTest: splitDatasetTrainTest,
    measureErrorOnDataset: measureErrorOnDataset,
    trainClassifier: trainClassifier,
    runTests: runTests
}

try {
    if (GLOBAL) {
        module.exports = nn;
    }
} catch (e) { console.log(e) };
