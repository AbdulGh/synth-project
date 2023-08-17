#include "./stk/ADSR.h"
#include "./stk/TwoPole.h"
#include "./stk/FileWvOut.h"

//waveforms
#include "./stk/Generator.h"
#include "./stk/SineWave.h"
#include "./stk/BlitSaw.h"
#include "./stk/BlitSquare.h"

#include <stdlib.h>
#include <time.h>

#include <iostream>
#include <fstream>
#include <filesystem>
#include <array>

using namespace stk;

constexpr int framesToGenerate = 50000;
constexpr stk::StkFloat sampleRate = 16000.0;

void generateRandomWaveforms(int number, std::string_view directory)
{
    for (int i = 0; i < number; ++i) {
        std::unique_ptr<Generator> generator;
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

        //too embarrased to admit that im using windows
        std::filesystem::path filenameBase(directory);
        filenameBase /= std::to_string(i);

        //write wave
        FileWvOut waveOut;
        waveOut.openFile(filenameBase.string() + ".wav", 1, FileWrite::FILE_WAV, Stk::STK_SINT16);
        StkFrames frames(framesToGenerate, 1);
        waveOut.tick(generator->tick(frames));
        waveOut.closeFile();

        //write features
        std::ofstream featuresOut(filenameBase.string() + ".txt");
        constexpr std::array<const char*, 3> waveFormEncodings{ "1 0 0 ", "0 1 0 ", "0 0 1 "};
        featuresOut << waveFormEncodings[waveChoice];
        featuresOut.close();
        
        generator.reset();
    }
}

int main()
{
    Stk::setSampleRate(sampleRate);
    Stk::showWarnings(true);
    srand(time(NULL));

    generateRandomWaveforms(10, "C:\\Users\\abdulg\\Desktop\\tmp");

	return 0;
}
