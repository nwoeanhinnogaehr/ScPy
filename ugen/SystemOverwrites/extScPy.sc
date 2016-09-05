+ PV_ChainUGen {
	addCopiesIfNeeded {
		var directDescendants, frames, buf, copy;
		// find UGens that have me as an input
		directDescendants = buildSynthDef.children.select ({ |child|
			var inputs;
			child.isKindOf(PV_Copy).not and: { child.isKindOf(Unpack1FFT).not } and: { child.isKindOf(Py).not } and: {
				inputs = child.inputs;
				inputs.notNil and: { inputs.includes(this) }
			}
		});
		if(directDescendants.size > 1, {
			// insert a PV_Copy for all but the last one
			directDescendants.drop(-1).do({|desc|
				desc.inputs.do({ arg input, j;
					if (input === this, {
						frames = this.fftSize;
						frames.widthFirstAntecedents = nil;
						buf = LocalBuf(frames);
						buf.widthFirstAntecedents = nil;
						copy = PV_Copy(this, buf);
						copy.widthFirstAntecedents = widthFirstAntecedents ++ [buf];
						desc.inputs[j] = copy;
						buildSynthDef.children.postln;
						buildSynthDef.children = buildSynthDef.children.drop(-3).insert(this.synthIndex + 1, frames);
						buildSynthDef.children = buildSynthDef.children.insert(this.synthIndex + 2, buf);
						buildSynthDef.children = buildSynthDef.children.insert(this.synthIndex + 3, copy);
						buildSynthDef.children.postln;
						buildSynthDef.indexUGens;
					});
				});
			});
		});
	}
}
