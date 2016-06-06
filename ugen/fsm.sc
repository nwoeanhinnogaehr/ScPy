FSM : UGen {
    *new { arg code, args=();
        var convertString = { |str| [str.asString.size] ++ str.ascii };
        var argsList = args.collect({|x| [convertString.value(x.class.asString), x]}).invert.collect(convertString).invert.asKeyValuePairs.flat;
        ^this.multiNewList(['control'] ++ convertString.value(code) ++ args.size ++ argsList);
    }
}
