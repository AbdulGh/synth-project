#include "./stk/FileWvOut.h"
#include "Synth.h"

#include <stdlib.h>
#include <time.h>

#include <iostream>
#include <fstream>
#include <filesystem>
#include <functional>
#include <cmath>

using namespace stk;

constexpr int framesToGenerate = 64000;
constexpr stk::StkFloat sampleRate = 16000.0;
constexpr float duration = framesToGenerate / sampleRate;

inline float uniformSample () {
    return static_cast<float>(rand() / static_cast<float>(RAND_MAX));
}

inline float notesFromA4(int note, int octave) {
    return 440.0f * pow(2, octave + note / 12.0f);
}

void generateRandomWaveforms(int number, std::string_view directory)
{
    for (int i = 0; i < number; ++i) {
        //choose a waveform and pitch
        float pitch = notesFromA4(rand() % 12, rand() % 3 - 2);
        Synth::waveform waveChoice = static_cast<Synth::waveform>(rand() % 3);
        
        //filter parameters
        int filterCutoff = 10 + static_cast<int>(rand() % 430);
        float filterResonance = 0.1 + uniformSample() / 9;

        //filter envelope
        float fAttack = duration * uniformSample() / 4;
        float fDecay = duration * uniformSample() / 4;
        float fSustain = uniformSample();
        float fRelease = duration * uniformSample() / 4;

        //put that all into a synth
        Synth synth{};
        synth.setPitch(pitch);
        synth.setWaveForm(waveChoice);
        synth.setFilterParameters(filterCutoff, filterResonance);
        synth.setFilterADSR(fAttack, fDecay, fSustain, fRelease);
        
        //too embarrased to admit that im using windows
        std::filesystem::path filenameBase(directory);
        filenameBase /= std::to_string(i);

        //write wave
        std::unique_ptr<StkFrames> sound = synth.synthesize(framesToGenerate);
        FileWvOut waveOut;
        waveOut.openFile(filenameBase.string() + ".wav", 1, FileWrite::FILE_WAV, Stk::STK_SINT16);
        waveOut.tick(*sound);
        waveOut.closeFile();
        sound.reset();

        //write parameters
        std::ofstream parametersOut(filenameBase.string() + ".txt");
        parametersOut << synth;
    }
}

int main()
{
    Stk::setSampleRate(sampleRate);
    Stk::showWarnings(true);
    srand(time(NULL));

    generateRandomWaveforms(10, "C:\\Users\\abdulg\\Desktop\\waves");

	return 0;
}
