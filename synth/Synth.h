#ifndef SYNTH_H
#define SYNTH_H

#include "./stk/ADSR.h"
#include "./stk/Oscillator.h"

#include <array>
#include <iostream>

using namespace stk;

class Synth
{
public:
    enum waveform {
        SINE,
        SAW,
        SQUARE
    };

    Synth(waveform waveType);

    void setPitch(float newPitch);
    void setWaveForm(waveform type);
    void setFilterParameters(float cutoff, float resonance);
    void setFilterADSR(float attackTime, float decayTime, float sustainLevel, float releaseTime);
    void setFilterLFOParameters(float frequency, float intensity);
    void setVibrato(float frequency, float intensity);

    std::unique_ptr<StkFrames> synthesize();
    friend std::ostream& operator<<(std::ostream& os, const Synth& obj);

private:
    float pitch; //not traditionally part of a 'patch' but still
    waveform wave;
    float filterCutoff;
    float filterResonance;
    float fAttackTime;
    float fDecayTime;
    float fSustainLevel;
    float fReleaseTime;
    float fModFreq;
    float fModInt;
    float pModFreq;
    float pModInt;

    std::unique_ptr<Oscillator> generator;

    static constexpr std::array<const char*, 3> waveFormEncodings{ "1 0 0", "0 1 0", "0 0 1" };
};

#endif