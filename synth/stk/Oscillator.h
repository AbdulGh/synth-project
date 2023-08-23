#ifndef STK_OSCILLATOR_H
#define STK_OSCILLATOR_H

#include "Generator.h"

namespace stk {

    /***************************************************/
    /*! \class Oscillator
        \brief Instances of Generator with an idea of frequency
    */
    /***************************************************/

    class Oscillator : public Generator
    {
    public:

        //! Class constructor.
        Oscillator(void) : Generator() {};

        virtual void setFrequency(StkFloat frequency) = 0;
    };

} // stk namespace

#endif
