FSM : UGen {
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
FSMInit {
    *new { arg code, args=(), doneAction=2;
        { FSM(code, args, doneAction) }.play
    }
}
