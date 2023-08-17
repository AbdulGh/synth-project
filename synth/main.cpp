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

using namespace stk;

int main()
{
    Stk::setSampleRate(16000.0);
    Stk::showWarnings(true);

    int nFrames = 50000;
    
    srand(time(NULL));

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

    FileWvOut output;
    output.openFile("output.wav", 1, FileWrite::FILE_WAV, Stk::STK_SINT16);
    StkFrames frames(nFrames, 1);
    output.tick(generator->tick( frames ));

    //clean up
    output.closeFile();
    generator.reset();

	return 0;
}
