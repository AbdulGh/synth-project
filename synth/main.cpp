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
        //set up oscillator waveform
        std::unique_ptr<Oscillator> generator;
        int waveChoice = rand() % 3;
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

        float pitch = notesFromA4(rand() % 12, rand() % 3 - 2);
        generator->setFrequency(pitch);

        //set up filter
        BiQuad filter = BiQuad();
        int filterCutoff = 10 + static_cast<int>(rand() % 430);
        float resonance = 0.1 + uniformSample() / 9;

        //filter envelope
        ADSR filterEnv = ADSR();
        float fAttack = duration * uniformSample() * static_cast<float>(framesToGenerate) / 4;
        float fDecay = duration * uniformSample() * static_cast<float>(framesToGenerate) / 4;
        float fSustain = uniformSample() * filterCutoff;
        float fRelease = duration * uniformSample() * static_cast<float>(framesToGenerate) / 4;
        filterEnv.setAllTimes(fAttack, fDecay, fSustain, fRelease);
        filterEnv.setTarget(filterCutoff);
        
        //too embarrased to admit that im using windows
        std::filesystem::path filenameBase(directory);
        filenameBase /= std::to_string(i);

        //compute wave
        StkFrames outputFrames(framesToGenerate, 1);
        int frame;

        generator->tick(outputFrames);

        /*
        filterEnv.keyOn();
        for (frame = 0; frame < framesToGenerate * 3 / 4; ++frame) {
            filter.setLowPass(filterCutoff * filterEnv.tick(), resonance);
            outputFrames[frame] = filter.tick(
                generator->tick()
            );
            //if (frame % 1200 == 0) std::cout << std::to_string(frame) << "\n";
        }

        filterEnv.keyOff();
        for (; frame < framesToGenerate; ++frame) {
            filter.setResonance(filterEnv.tick(), resonance, true);
            outputFrames[frame] = filter.tick(
                generator->tick()
            );
            //if (i % 1200 == 0) std::cout << std::to_string(frame) << "\n";
        }
        */

        //write wave
        FileWvOut waveOut;
        waveOut.openFile(filenameBase.string() + ".wav", 1, FileWrite::FILE_WAV, Stk::STK_SINT16);
        waveOut.tick(outputFrames);
        waveOut.closeFile();

        //write features
        constexpr std::array<const char*, 3> waveFormEncodings{ "1 0 0", "0 1 0", "0 0 1" };
        std::ofstream featuresOut(filenameBase.string() + ".txt");
        std::function<void(std::string)> writeFeature = [&featuresOut](std::string toWrite) {
            featuresOut << toWrite << " ";
        };
        writeFeature(std::to_string(pitch));
        writeFeature(waveFormEncodings[waveChoice]);
        writeFeature(std::to_string(filterCutoff));
        writeFeature(std::to_string(resonance));
        writeFeature(std::to_string(fAttack));
        writeFeature(std::to_string(fDecay));
        writeFeature(std::to_string(fSustain));
        writeFeature(std::to_string(fRelease));
        
        generator.reset();
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
