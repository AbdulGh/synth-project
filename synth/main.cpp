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
#include <random>

using namespace std;

//random real between 0 and 0.1
inline float uniformSample () {
    return static_cast<float>(rand()) / (10.0 * RAND_MAX);
}

inline float notesFromA4(int note) {
    return 440.0f * pow(2, note / 12.0f);
}

void generateRandomWaveforms(int number, string_view directory, unique_ptr<array<float, 3>> waveProbs = NULL, unique_ptr<array<array<float, 10>,11>> paramProbs = NULL)
{
    std::random_device rd{};
    std::mt19937 generator{ rd() };

    function<float(discrete_distribution<int>*)> sampleDistribution = [&generator](discrete_distribution<int>* dist) {
        return (*dist)(generator) / 10.0f + uniformSample();
    };

    for (int i = 0; i < number; ++i) {
        //fill in unspecified probabilities with uniform ones
        if (waveProbs == NULL) {
            waveProbs = make_unique<array<float, 3>>();
            waveProbs->fill(1.0f / 3);
        }
        if (paramProbs == NULL) {
            paramProbs = make_unique<array<array<float, 10>, 11>> ();
            for (int j = 0; j < 11; ++j) {
                paramProbs->at(j) = array<float, 10>();
                paramProbs->at(j).fill(0.1f);
            }
        }

        //make numerical probabilities into a bunch of discrete distributions
        discrete_distribution<int> waveDistribution(waveProbs->begin(), waveProbs->end());
        array<discrete_distribution<int>, 11> continuousDistributions;
        for (int j = 0; j < 11; ++j) {
            continuousDistributions[j] = discrete_distribution<int>(paramProbs->at(j).begin(), paramProbs->at(j).end());
        }

        Synth::waveform waveChoice = static_cast<Synth::waveform>(waveDistribution(generator));

        //choose a waveform and pitch
        float pitch = notesFromA4(24 * sampleDistribution(&continuousDistributions[0]));

        //filter parameters
        int filterCutoff = 2 * pitch * sampleDistribution(&continuousDistributions[1]);
        float filterResonance = sampleDistribution(&continuousDistributions[2]);

        //filter envelope
        float fAttack = duration * sampleDistribution(&continuousDistributions[3]) / 4;
        float fDecay = duration * sampleDistribution(&continuousDistributions[4]) / 4;
        float fSustain = (rand() % 5) ? sampleDistribution(&continuousDistributions[5]) : 0; //rand() because zero sustain level is very unlikely otherwise, but we'd like to have some examples of this
        float fRelease = duration * sampleDistribution(&continuousDistributions[6]) / 4; //todo consider setting this to some fixed value if no sustain, for the CNN

        //filter modulation
        int filterModulated = rand() % 2;
        float fModFreq = filterModulated * sampleDistribution(&continuousDistributions[7]) * 8;
        float fModInt = filterModulated * sampleDistribution(&continuousDistributions[8]) / 2;

        //vibrato
        int pitchModulated = rand() % 2;
        float pModFreq = pitchModulated * sampleDistribution(&continuousDistributions[9]) * 8;
        float pModInt = pitchModulated * sampleDistribution(&continuousDistributions[10]) / 32;

        //put that all into a synth
        Synth synth{waveChoice};
        synth.setPitch(pitch);
        synth.setWaveForm(waveChoice);
        synth.setFilterParameters(filterCutoff, filterResonance);
        synth.setFilterADSR(fAttack, fDecay, fSustain, fRelease);
        synth.setFilterLFO(fModFreq, fModInt);
        synth.setVibrato(pModFreq, pModInt);

        //too embarrased to admit that im using windows
        filesystem::path filenameBase(directory);
        filenameBase /= to_string(i);

        //write wave
        unique_ptr<StkFrames> sound = synth.synthesize();
        FileWvOut waveOut;
        waveOut.openFile(filenameBase.string() + ".wav", 1, FileWrite::FILE_WAV, Stk::STK_SINT16);
        waveOut.tick(*sound);
        waveOut.closeFile();
        sound.reset();

        //write parameters
        ofstream parametersOut(filenameBase.string() + ".txt");
        parametersOut << synth;
    }
}

inline int boole(char* zeroOrOne) {
    return zeroOrOne[0] == 0 ? 0 : 1;
}

unique_ptr<stk::StkFrames> invokeSynthesizer(char** args) {
    Synth::waveform wave = Synth::waveform::SINE;

    //waves go SINE, SAW, SQUARE
    for (int i = 0; i < 3; ++i) {
        if (args[i][0] == '1') {
            wave = static_cast<Synth::waveform>(i);
            break;
        }
    }

    Synth synth{wave};
    synth.setPitch(stof(args[3]));
    synth.setFilterParameters(
        stof(args[4]),
        stof(args[5])
    );
    synth.setFilterADSR(
        stof(args[6]),
        stof(args[7]),
        stof(args[8]),
        stof(args[9])
    );
    synth.setFilterLFO(
        stof(args[10]),
        stof(args[11])
    );
    synth.setVibrato(
        stof(args[12]),
        stof(args[13])
    );

    return synth.synthesize();
}

int main(int argc, char** argv)
{
    //we dont really do any error checking on these parameters
    if (argc == 3) {
        char* dir = argv[2];
        int num = stoi(argv[1]); //no error handling for now
        Stk::setSampleRate(sampleRate);
        Stk::showWarnings(true);
        srand(time(NULL));
        generateRandomWaveforms(num, dir);
    }
    else if (argc == 16) {
        unique_ptr<stk::StkFrames> sound = invokeSynthesizer(argv + 1);
        FileWvOut waveOut;
        waveOut.openFile(argv[15], 1, FileWrite::FILE_WAV, Stk::STK_SINT16);
        waveOut.tick(*sound);
        waveOut.closeFile();
        sound.reset();
    }
    else {
        cout << "USAGE: " << argv[0] << " numberofexamples directory\n";
        cout << "or: " << argv[0] << " (14 parameters) outputfile\n";
        return 0;
    }

	return 0;
}
