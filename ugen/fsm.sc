FSM : UGen {
    *new { arg code, args=(), doneAction=0;
        var convertString = { |str| [str.asString.size] ++ str.ascii };
        var argsList = args.collect({|x| [convertString.value(x.class.asString), x]}).invert.collect(convertString).invert.asKeyValuePairs.flat;
        ^this.multiNewList(['control', doneAction] ++ convertString.value(code) ++ args.size ++ argsList);
    }
}
FSMInit {
    *new { arg code, args=(), doneAction=2;
        { FSM(code, args, doneAction) }.play
    }
}
