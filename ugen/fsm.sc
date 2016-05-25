FSM : UGen {
    *new { arg code;
        ^this.multiNewList(['control', code.size] ++ code.ascii);
    }
}
