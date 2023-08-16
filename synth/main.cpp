#include "./stk/ADSR.h"
#include "./stk/TwoPole.h"
#include "./stk/SineWave.h"
#include "./stk/FileWvOut.h"

using namespace stk;

int main()
{
    // Set the global sample rate before creating class instances.
    Stk::setSampleRate(44100.0);
    Stk::showWarnings(true);

    int nFrames = 100000;
    SineWave sine;
    FileWvOut output;

    output.openFile("hellosine.wav", 1, FileWrite::FILE_WAV, Stk::STK_SINT16);

    sine.setFrequency(441.0);

    StkFrames frames( nFrames, 1 );
    output.tick(sine.tick( frames ));

	return 0;
}
