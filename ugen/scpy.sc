Py : UGen {
    *new { arg code, args=(), doneAction=0;
        var convertString = { |str|
            [str.asString.size] ++ str.ascii
        };
        var convertClass = { |x|
            var className = x.class.asString;
            case
                { className == "FFT" } { className } // special case, FFT is a UGen but is handled differently
                { x.isKindOf(UGen) } { "UGen" }
                { true } { className };
        };
        var convertValue = { |x|
            if (x.class == Array,
                { [convertString.value(convertClass.value(x)), [x.size]
                    ++ x.collect({ |y| convertValue.value(y) })] },
                { [convertString.value(convertClass.value(x)), x] })
        };
        var convertArray = { |array|
            array.collect(convertValue);
        };
        var convertArgs = { |args|
            convertArray.value(args).invert.collect(convertString).invert.asKeyValuePairs.flat;
        };
        var argsList = ['control', doneAction]
            ++ convertString.value(code)
            ++ args.size
            ++ convertArgs.value(args);
        ^this.multiNewList(argsList);
    }
}
PyOnce {
    *new { arg code, args=(), doneAction=2;
        { Py(code, args, doneAction) }.play
    }
}
PyFile {
    *new { arg filename, args=(), doneAction=0;
        var file = File(filename, "r");
        var code = file.readAllString;
        file.close;
        Py(code, args, doneAction)
    }
}
PyOnceFile {
    *new { arg filename, args=(), doneAction=2;
        var file = File(filename, "r");
        var code = file.readAllString;
        file.close;
        PyOnce(code, args, doneAction)
    }
}
