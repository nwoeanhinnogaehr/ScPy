FSM : UGen {
    *new { arg code, args=();
        var convertString = { |str| [str.asString.size] ++ str.ascii };
        var argsList = args.collect({|x| [x, convertString.value(x.class.asString)]}).invert.collect(convertString).asKeyValuePairs.flat;
        ^this.multiNewList(['control'] ++ convertString.value(code) ++ args.size ++ argsList);
    }
}
