# Sirith Tetrahedral Pi Pico 2 W Firmware TODO
## Description of Hardware
It is a four microphone array with the array shape of a tetrahedron (configurable, but here each microphone is exactly 50 mm from the others). The high quality MEMs microphones are processed by the ADAU7112. This will be using TDM This optionally has a GPS connection, but should be able to run standalone as a single node without GPS (with all four microphones hardwired, the need for 'perfect' timing across a network is not critical).
TX = GP12 (GPS such as M10Q, optional)
RX = GP13 (GPS, optional)
PPS = GP10 (optional)
SCL = GP19
SDA = GP18
tdm sdata = GP7
tdm bclk = GP8
tdm ws = GP9
MK4 is the top of the pyramid, above the plane of the others, and is the Left of the pair fed to the second ADAU7112. DIP switches mean either ADAU7112 can be either i2s, 1/2 or 3/4 slots, but the expected default here is TDM slot 3 is the top of the pyramid mic.
Likely the code needs to be designed to easily "rotate" the configured orientation of microphones during manual calibration. Although it would be useful to have an optional auto-orientation from the onboard LIS2MDLTR (not installation is expected to be fixed, so can filter or smooth heavily readings to make up for usual high noise in these digital compasses).
Design should generally be configured "safe" so that bugs cannot destroy the board.

