#include "./stk/ADSR.h"
#include "./stk/BiQuad.h"
#include "./stk/FileWvOut.h"

//waveforms
#include "./stk/SineWave.h"
#include "./stk/BlitSaw.h"
#include "./stk/BlitSquare.h"

#include <stdlib.h>
#include <time.h>

#include <iostream>
#include <fstream>
#include <filesystem>
#include <array>
#include <functional>

using namespace stk;

constexpr int framesToGenerate = 64000;
constexpr stk::StkFloat sampleRate = 16000.0;

inline float uniformSample () {
    return static_cast<float>(rand() / static_cast<float>(RAND_MAX));
}

void generateRandomWaveforms(int number, std::string_view directory)
{
    for (int i = 0; i < number; ++i) {
        //set up oscillator waveform
        std::unique_ptr<Generator> generator;
        int waveChoice = rand() % 3;
        waveChoice = 1;
        switch (waveChoice) {
            case 0:
                generator = std::make_unique<SineWave>();
                break;
            case 1:
                generator = std::make_unique<BlitSaw>();
                break;
            default:
                generator = std::make_unique<BlitSquare>();
        }

        //set up filter
        BiQuad filter = BiQuad();
        /*int frequency = 10 + static_cast<int>(rand() % 430);
        float resonance = 0.1 + uniformSample() / 9;
        filter.setResonance(frequency, resonance, true);

        //filter envelope
        ADSR filterEnv = ADSR();
        float fAttack = uniformSample() * static_cast<float>(framesToGenerate) / 4;
        float fDecay = uniformSample() * static_cast<float>(framesToGenerate) / 4;
        float fSustain = uniformSample() * frequency;
        float fRelease = uniformSample() * static_cast<float>(framesToGenerate) / 4;
        filterEnv.setAttackRate(fAttack);
        filterEnv.setAttackRate(static_cast<float>(1) / (4 * framesToGenerate));*/
        
        //too embarrased to admit that im using windows
        std::filesystem::path filenameBase(directory);
        filenameBase /= std::to_string(i);

        //compute wave
        StkFrames frames(framesToGenerate, 1);
        for (int i = 0; i < framesToGenerate; ++i) {
            //const float filterEnvVal = filterEnv.tick(); //todo check if this is faster vectorised upfront
            filter.setResonance(500 - static_cast<float>(i) * 200 / framesToGenerate, 0.999, true);
            frames[i] = filter.tick(
                generator->tick()
            );
            if (i % 1200 == 0) std::cout << std::to_string(static_cast<float>(i) * 440 / framesToGenerate) << "\n";
        }

        //write wave
        FileWvOut waveOut;
        waveOut.openFile(filenameBase.string() + ".wav", 1, FileWrite::FILE_WAV, Stk::STK_SINT16);
        waveOut.tick(frames);
        waveOut.closeFile();

        //write features
        std::ofstream featuresOut(filenameBase.string() + ".txt");
        std::function<void(std::string)> writeFeature = [&featuresOut](std::string toWrite) {
            featuresOut << toWrite << " ";
        };

        constexpr std::array<const char*, 3> waveFormEncodings{ "1 0 0", "0 1 0", "0 0 1"};
        /*writeFeature(waveFormEncodings[waveChoice]);
        writeFeature(std::to_string(frequency));
        writeFeature(std::to_string(resonance));
        writeFeature(std::to_string(fAttack));
        writeFeature(std::to_string(fDecay));
        writeFeature(std::to_string(fSustain));
        writeFeature(std::to_string(fRelease));*/
        
        generator.reset();
    }
}

int main()
{
    Stk::setSampleRate(sampleRate);
    Stk::showWarnings(true);
    srand(time(NULL));

    generateRandomWaveforms(10, "C:\\Users\\abdulg\\source\\repos\\Synth\\backwards\\tmp");

	return 0;
}
