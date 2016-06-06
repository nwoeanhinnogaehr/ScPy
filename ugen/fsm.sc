FSM : UGen {
    *new { arg code, args=();
        var convertString = { |str| [str.asString.size] ++ str.ascii };
        var argsList = args.invert.collect(convertString).invert.asKeyValuePairs.flatten;
        ^this.multiNewList(['control'] ++ convertString.value(code) ++ args.size ++ argsList);
    }
}
