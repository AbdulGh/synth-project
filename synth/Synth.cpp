#include "Synth.h"

#include "./stk/BiQuad.h"
#include "./stk/Oscillator.h"
#include "./stk/SineWave.h"
#include "./stk/BlitSaw.h"
#include "./stk/BlitSquare.h"
 
#include <functional>

std::unique_ptr<StkFrames> Synth::synthesize(unsigned int framesToGenerate) {

    std::unique_ptr<Oscillator> generator;
    switch (wave) {
    case SINE:
        generator = std::make_unique<SineWave>();
        break;
    case SAW:
        generator = std::make_unique<BlitSaw>();
        break;
    case SQUARE:
        generator = std::make_unique<BlitSquare>();
        break;
    default:
        std::cerr << "Weird wave seen";
        generator = std::make_unique<SineWave>();
    }
    generator->setFrequency(pitch);

    ADSR filterEnv = ADSR();
    filterEnv.setAllTimes(fAttackTime, fDecayTime, fSustainLevel, fReleaseTime);

    BiQuad filter = BiQuad();

    std::unique_ptr<StkFrames> outputFrames = std::make_unique<StkFrames>(framesToGenerate, 1);

    int frame;
    filterEnv.keyOn();
    for (frame = 0; frame < framesToGenerate * 3 / 4; ++frame) {
        filter.setLowPass(filterEnv.tick() * filterCutoff, filterResonance);
        (*outputFrames)[frame] = filter.tick(
            generator->tick()
        );
    }
    filterEnv.keyOff();
    for (; frame < framesToGenerate; ++frame) {
        filter.setLowPass(filterEnv.tick() * filterCutoff, filterResonance);
        (*outputFrames)[frame] = filter.tick(
            generator->tick()
        );
    }

    generator.reset();

    return outputFrames;
}

void Synth::setPitch(float newPitch) {
    pitch = newPitch;
}

void Synth::setFilterADSR(float attackTime, float decayTime, float sustainLevel, float releaseTime) {
	fAttackTime = attackTime;
	fDecayTime = decayTime;
	fSustainLevel = sustainLevel;
	fReleaseTime = releaseTime;
}

void Synth::setFilterParameters(float cutoff, float resonance) {
    filterCutoff = cutoff;
    filterResonance = resonance;
}

void Synth::setWaveForm(waveform type) {
    wave = type;
}

std::ostream& operator<<(std::ostream& os, const Synth& syn) {
    std::function<void(float)> writeParameter = [&os](float toWrite) {
        os << std::to_string(toWrite) << " ";
    };

    os << syn.waveFormEncodings[syn.wave] << " ";
    writeParameter(syn.pitch);
    writeParameter(syn.filterCutoff);
    writeParameter(syn.filterResonance);
    writeParameter(syn.fAttackTime);
    writeParameter(syn.fDecayTime);
    writeParameter(syn.fSustainLevel);
    writeParameter(syn.fReleaseTime);

    return os;
}