# About this Project

This project takes a string as input, converts it to it's binary representation (using ASCII encoding), and then outputs a CSV definition of a Arbitrary Waveform to use with Siglent's series of Signal Generators.

## Inputs
- **string_to_serialize**: The string you want to convert
- **slew_rate**: Requested slew rate when switching between high and low (to reduce ringing), in seconds. Minimum 1ns, maximum 100ns.
- **logic_high_level**: the logic HIGH voltage level
- **logic_low_level**: the logic LOW voltage level
- **baud_rate**: The baud rate of the output wave. Common values are 4800, 9600, 19200, 38400, 57600, 115200.

### Buad rate, data length, and frequency
The Arbitrary wave definition format uses "frequency" and "data length" as the two input variables. The defined waveform's available *points* is then spread out over the selected frequency. The number of points is equal to the data length.

Because of this, there is a inherent relationship between the selected the frequency / data length and baud rate. To compensate for this, the baud rate is converted to a equivalent frequency + data length as input.