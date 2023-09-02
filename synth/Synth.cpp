#include "Synth.h"
#include "Config.h"

#include "./stk/BiQuad.h"
#include "./stk/Oscillator.h"
#include "./stk/SineWave.h"
#include "./stk/BlitSaw.h"
#include "./stk/BlitSquare.h"
 
#include <functional>

Synth::Synth(waveform waveType)
{
    setWaveForm(waveType);
}

std::unique_ptr<StkFrames> Synth::synthesize() 
{
    ADSR filterEnv = ADSR(); //todo no need to remake this each synthesis
    filterEnv.setAllTimes(fAttackTime, fDecayTime, fSustainLevel, fReleaseTime);

    BiQuad filter = BiQuad();

    SineWave filterLFO = SineWave(); //todo - maybe allow for different waveforms here?
    filterLFO.setFrequency(fModFreq);

    SineWave vibrato = SineWave();
    vibrato.setFrequency(pModFreq);

    std::unique_ptr<StkFrames> outputFrames = std::make_unique<StkFrames>(framesToGenerate, 1);

    std::function<void(int)> tick;
    if (filterOn) {
        tick = [&](int frame) {
            generator->setFrequency(pitch * (1 + pModInt * vibrato.tick()));
            filter.setLowPass(
                filterEnv.tick() * filterCutoff * (1 + fModInt * filterLFO.tick()),
                filterResonance
            );
            (*outputFrames)[frame] = filter.tick(
                generator->tick()
            );
        };
    }
    else {
        tick = [&](int frame) {
            generator->setFrequency(pitch * (1 + pModInt * vibrato.tick()));
            (*outputFrames)[frame] = generator->tick();
        };
    }

    int frame;
    filterEnv.keyOn();
    for (frame = 0; frame < framesToGenerate * 3 / 4; ++frame) tick(frame);
    filterEnv.keyOff();
    for (; frame < framesToGenerate; ++frame) tick(frame);

    return outputFrames;
}

void Synth::setPitch(float newPitch)
{
    pitch = newPitch;
}

void Synth::setFilterADSR(float attackTime, float decayTime, float sustainLevel, float releaseTime)
{
	fAttackTime = attackTime;
	fDecayTime = decayTime;
	fSustainLevel = sustainLevel;
	fReleaseTime = releaseTime;
}

void Synth::setFilterLFO(float frequency, float intensity)
{
    fModFreq = frequency;
    fModInt = intensity;
}

void Synth::setVibrato(float frequency, float intensity)
{
    pModFreq = frequency;
    pModInt = intensity;
}

void Synth::setFilterParameters(float cutoff, float resonance)
{
    filterCutoff = cutoff;
    filterResonance = resonance;
}

void Synth::setWaveForm(waveform type) {
    generator.reset();
    wave = type;
    switch (type) {
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
}

void Synth::setFilterOn(bool setting)
{
    filterOn = setting;
}

std::ostream& operator<<(std::ostream& os, const Synth& syn)
{
    std::function<void(float)> writeParameter = [&os](float toWrite) 
    {
        os << std::to_string(toWrite) << " ";
    };

    os << syn.waveFormEncodings[syn.wave] << " ";
    //os << syn.filterOn ? "1 " : "0 ";
    writeParameter(syn.pitch);
    writeParameter(syn.filterCutoff);
    writeParameter(syn.filterResonance);
    writeParameter(syn.fAttackTime);
    writeParameter(syn.fDecayTime);
    writeParameter(syn.fSustainLevel);
    writeParameter(syn.fReleaseTime);
    writeParameter(syn.fModFreq);
    writeParameter(syn.fModInt);
    writeParameter(syn.pModFreq);
    writeParameter(syn.pModInt);

    return os;
}