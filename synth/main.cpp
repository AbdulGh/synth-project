#include "./stk/FileWvOut.h"
#include "Synth.h"
#include "Config.h"

#include <stdlib.h>
#include <time.h>

#include <iostream>
#include <fstream>
#include <filesystem>
#include <functional>
#include <cmath>
#include <string>

using namespace stk;

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
        float pitch = notesFromA4(rand() % 12, rand() % 2);
        Synth::waveform waveChoice = static_cast<Synth::waveform>(rand() % 3);
        
        //filter parameters
        int filterCutoff = static_cast<int>(rand() % static_cast<int>(ceil(pitch)));
        float filterResonance = uniformSample();

        //filter envelope
        float fAttack = duration * uniformSample() / 4;
        float fDecay = duration * uniformSample() / 4;
        float fSustain = (rand() % 5) ? uniformSample() : 0; //rand() because zero sustain level is very unlikely otherwise, but we'd like to have some examples of this
        float fRelease = duration * uniformSample() / 4; //todo consider setting this to some fixed value if no sustain, for the CNN

        //filter modulation
        int filterModulated = rand() % 2;
        float fModFreq = filterModulated * uniformSample() * 8;
        float fModInt = filterModulated * uniformSample() / 2;

        //vibrato
        int pitchModulated = rand() % 2;
        float pModFreq = pitchModulated * uniformSample() * 8;
        float pModInt = pitchModulated * uniformSample() / 32;

        //put that all into a synth
        Synth synth{waveChoice};
        synth.setPitch(pitch);
        synth.setWaveForm(waveChoice);
        synth.setFilterParameters(filterCutoff, filterResonance);
        synth.setFilterADSR(fAttack, fDecay, fSustain, fRelease);
        synth.setFilterLFO(fModFreq, fModInt);
        synth.setVibrato(pModFreq, pModInt);
        
        //too embarrased to admit that im using windows
        std::filesystem::path filenameBase(directory);
        filenameBase /= std::to_string(i);

        //write wave
        std::unique_ptr<StkFrames> sound = synth.synthesize();
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

inline int boole(char* zeroOrOne) {
    return zeroOrOne[0] == 0 ? 0 : 1;
}

std::unique_ptr<stk::StkFrames> invokeSynthesizer(char** args) {
    Synth::waveform wave = Synth::waveform::SINE;

    //waves go SINE, SAW, SQUARE
    for (int i = 0; i < 3; ++i) {
        if (args[i] == "1") {
            wave = static_cast<Synth::waveform>(i);
            break;
        }
    }

    Synth synth{wave};
    synth.setPitch(std::stof(args[3]));
    synth.setFilterParameters(
        std::stof(args[4]),
        std::stof(args[5])
    );
    synth.setFilterADSR(
        std::stof(args[6]),
        std::stof(args[7]),
        std::stof(args[8]),
        std::stof(args[9])
    );
    synth.setFilterLFO(
        std::stof(args[10]),
        std::stof(args[11])
    );
    synth.setVibrato(
        std::stof(args[12]),
        std::stof(args[13])
    );

    return synth.synthesize();
}

int main(int argc, char** argv)
{
    //we dont really do any error checking on these parameters
    if (argc == 3) {
        char* dir = argv[2];
        int num = std::stoi(argv[1]); //no error handling for now
        Stk::setSampleRate(sampleRate);
        Stk::showWarnings(true);
        srand(time(NULL));
        generateRandomWaveforms(num, dir);
    }
    else if (argc == 16) {
        std::unique_ptr<stk::StkFrames> sound = invokeSynthesizer(argv + 1);
        FileWvOut waveOut;
        waveOut.openFile(argv[14], 1, FileWrite::FILE_WAV, Stk::STK_SINT16);
        waveOut.tick(*sound);
        waveOut.closeFile();
        sound.reset();
    }
    else {
        std::cout << "USAGE: " << argv[0] << " numberofexamples directory\n";
        std::cout << "USAGE: " << argv[0] << " (14 parameters) outputfile\n";
        return 0;
    }

	return 0;
}
