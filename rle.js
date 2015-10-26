/*
    RLE codec for byte arrays
    encoded form consists of parts of source array located one after another
    and coded with one of three byte encoding types:
    single character: 0 -> single_byte
    multiple single characters: 1 -> num_of_chars -> chars
    multiple repeating characters: 2 -> num_of_chars -> char  
*/

var S_DIFF = 0, S_EQ = 1;

function rleenc(a) {
    var i = 0;
    var c,p;
    var state;
    var counter = 0;
    var numIndex;
    var r;
    
    r = [1,0];
    numIndex = 1;
    state = S_DIFF;
    
    function countNext() { counter++; i++ }
    
    function finalize() {
        r[numIndex] = counter;
    }
    
    while (i < a.length) {
        c = a[i];
        
        if (counter == 256) {
            counter--;
            r[numIndex] = counter;
            if (state == S_DIFF) {
                var b = r.pop();
                r.push(1);
                r.push(1);
                numIndex = r.length-1;
                counter = 1;
                r.push(b);
            } else if (state == S_EQ) {
                var char = r[r.length-1];
                r.push(2);
                r.push(1);
                numIndex = r.length-1;
                counter = 1;
                r.push(char);
            }
        }
        
        if (state == S_DIFF) {
            if (c != p || c != a[i+1]) {
                r.push(c);
                countNext();
            } else {
                counter--;
                r[numIndex] = counter;
                r.pop();
                r.push(2);
                r.push(2);
                counter = 2;
                numIndex = r.length-1;
                r.push(c);
                state = S_EQ;
                i++;
            }
        } else if (state == S_EQ) {
            if (c == p) {
                countNext();
            } else {
                r[numIndex] = counter;
                r.push(1);
                r.push(1);
                counter = 1;
                numIndex = r.length-1;
                r.push(c);
                state = S_DIFF;
                i++;
            }
        }
        
        p = c;
    }
    
    finalize();
    
    return r;
}

function rledec(a) {
    var i = 0;
    var c;
    var r = [];
    var lastState = '';
    
    while (i < a.length) {
        c = a[i];
        if (c == 0) {
            r.push(a[i+1]);
            i += 2;
            lastState = 'Single';
        } else if (c == 1) {
            var N = a[i+1];
            var j;
            for (j = 0; j < N; j++) {
                r.push(a[i+2+j]);
            }
            i += N + 2;
            lastState = 'Different';
        } else if (c == 2) {
            var N = a[i+1];
            var j;
            var fill = a[i+2];
            for (j = 0; j < N; j++) {
                r.push(fill);
            }
            i += 3;
            lastState = 'Equal';
        } else {
            console.log('Error, got '+c+' at '+i+', State: '+lastState);
            return;
        }
    }
    
    return r;
}

function randInt(lim) {
    return Math.floor(Math.random()*lim);
}

function compareArrays(a,b) {
    if (a.length != b.length) return false;
    var i;
    for (i=0;i<a.length;i++) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

function testRLE(enc, dec, N, testcaseGen) {
    N = N || 1000;
    
    testcaseGen = testcaseGen || function () {
        
        var res = [];
        
        function genRand(n) { var i; for (i=0; i<n; i++) { res.push(randInt(256)) } }
        function genRepeat(n) { var i; var c = randInt(256); for (i=0; i<n; i++) { res.push(c) } }
        function genFast(n) {
            var i;
            var a = randInt(256); var b = randInt(256);
            for (i=0; i<n; i++) { if (Math.random() > 0.5) { res.push(a) } else { res.push(b) } }
        }
        var i;
        var nstages = 10+randInt(100);
        for (i=0; i<nstages; i++) {
            var rand = Math.random();
            if (rand < 0.3) {
                genRand(randInt(2000));      
            } else if (rand > 0.3 && rand < 0.6) {
                genRepeat(randInt(2000));
            } else {
                genFast(randInt(2000));
            }
        }
        
        return res;
    };
    
    var i;
    var passed = 0;
    for (i=0; i<N; i++) {
        var data = testcaseGen();
        //console.log('DATA: '+data);
        var encData = enc(data);
        if (compareArrays(dec(encData), data)) {
            passed++;
        }
    }
    console.log('Test done, '+passed+' tests of '+N+' passed');
}

module.exports = {
    encode: rleenc,
    decode: rledec,
    compareArrays: compareArrays,
    test: function () { testRLE(rleenc, rledec, 100); }
};
