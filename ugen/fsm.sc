FSM : UGen {
    *new { arg code, args=(), doneAction=0;
        var convertString = { |str|
            [str.asString.size] ++ str.ascii
        };
        var convertValue = { |x|
            if (x.class == Array,
                { [convertString.value(x.class.asString), [x.size]
                    ++ x.collect({ |y| convertValue.value(y) })] },
                { [convertString.value(x.class.asString), x] })
        };
        var convertArray = { |array|
            array.postln;
            array.collect(convertValue);
        };
        var convertArgs = { |args|
            convertArray.value(args).invert.collect(convertString).invert.asKeyValuePairs.flat;
        };
        var argsList = ['control', doneAction]
            ++ convertString.value(code)
            ++ args.size
            ++ convertArgs.value(args);
        argsList.postln;
        ^this.multiNewList(argsList);
    }
}
FSMInit {
    *new { arg code, args=(), doneAction=2;
        { FSM(code, args, doneAction) }.play
    }
}
