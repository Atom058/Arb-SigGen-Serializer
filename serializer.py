#!/usr/bin/env python
# coding: utf-8

# # About this notebook
# 
# This notebook takes a string as input, converts it to it's binary representation (using ASCII encoding), and then outputs a CSV definition of a Arbitrary Waveform to use with Siglent's series of Signal Generators.
# 
# ## Inputs
# - **strings_to_serialize**: The string you want to convert
# - **slew_rate**: Requested slew rate when switching between high and low (to reduce ringing), in seconds. Minimum 500ns, maximum 1/4 of baud rate (dynamic).
# - **logic_high_level**: the logic HIGH voltage level
# - **logic_low_level**: the logic LOW voltage level
# - **baud_rate**: The baud rate of the output wave. Common values are 4800, 9600, 19200, 38400, 57600, 115200.
# 
# ### Buad rate, data length, and frequency
# The Arbitrary wave definition format uses "frequency" and "data length" as the two input variables. The defined waveform's available *points* is then spread out over the selected frequency. The number of points is equal to the data length.
# 
# Because of this, there is a inherent relationship between the selected the frequency / data length and baud rate. To compensate for this, the baud rate is converted to a equivalent frequency + data length as input.

# In[45]:


# Switching this to 'True' will enable Notebook outputs. 
# When set to 'False', the file is optimized for exporting to .py.
dev_mode = False
if dev_mode:
    dev_args = [
        '--strings_to_serialize', "$GPZDA,181813,14,10,2003,00,00*4F\r\n", "$GPRMC,161229.487,A,3723.2475,N,12158.3416,W,0.13,309.62,120598, ,*10\r\n", "$GPGGA,002153.000,3342.6618,N,11751.3858,W,1,10,1.2,27.0,M,-34.2,M,,0000*5E\r\n"
        , '--baud_rate', '4800'
        , '--logic_high_level', '3.3'
        , '--logic_low_level', '0.0'
        , '--waveform_delay', '0.175'
        , '--slew_rate', '1e-6'
        , '--encoding', '8N1'
    ]
    print (dev_args)


# In[46]:


import argparse

def enforced_positive_int(val):
    ival = int(val)
    if ival < 0:
        raise argparse.ArgumentTypeError(f"{val} is not a positive integer")
    return ival
def enforced_positive_float(val):
    ival = float(val)
    if ival < 0:
        raise argparse.ArgumentTypeError(f"{val} is not a positive float")
    return ival

parser = argparse.ArgumentParser(
    prog="arb_siggen_serializer", 
    description="""
        Converts an input string to ASCII Binary sequence for Siglent's Arbitrary Signal Generator.
        Output is a csv file that is equivalent in format to what EasyWaveX software produces.
    """
)
parser.add_argument(
    '--strings_to_serialize', '-str'
    , help='Input string(s). If multiple strings are provided, a waveform_delay is added between them.'
    , nargs='*'
    , required=True
)
parser.add_argument(
    '--baud_rate', '-br'
    , help='BAUD rate of message'
    , type=enforced_positive_int
    , required=True
)
parser.add_argument(
    '--logic_high_level', '-lh'
    , help='Logical HIGH level in volts'
    , type=float
    , required=True
)
parser.add_argument(
    '--logic_low_level', '-ll'
    , help='Logical LOW level in volts'
    , type=float
    , required=True
)
parser.add_argument(
    '--waveform_delay', '-wd'
    , help="""
        Insert a delay between two subsequent phases, in seconds.
        Will be aligned to be a whole number of symbols long.
    """
    , type=enforced_positive_float
    , required=False
    , default=0
    , dest='waveform_delay_input'
)
parser.add_argument(
    '--slew_rate', '-slr'
    , help="""
        Requested slew rate when switching between high and low (to reduce ringing), in seconds. 
        Easiest way is to provide it in scientific notation (i.e. '1e-6' for 1µs)
        Minimum: 0
        Maximum: 1/4 of baud rate (dynamic)
    """
    , type=enforced_positive_float
    , required=False
    , default=0
    , dest='slew_rate_input'
)
parser.add_argument(
    '--encoding', '-e'
    , help="Character encoding to use. Default is 8N1. Only 8 data bits are supported."
    , choices=['8N1', '8O1', '8E1']
    , required=False
    , default='8N1'
)
parser.add_argument(
    '--endian', '-end'
    , help="Endian (most or least significant bit are sen first)"
    , choices=['LSB', 'MSB']
    , required=False
    , default='MSB'
)

args = parser.parse_args() if not dev_mode else parser.parse_args(dev_args)

# Inputs
strings_to_serialize    = args.strings_to_serialize
baud_rate               = args.baud_rate
logic_high_level        = args.logic_high_level
logic_low_level         = args.logic_low_level
waveform_delay_input    = args.waveform_delay_input
slew_rate_input         = args.slew_rate_input
encoding                = args.encoding.upper()
endian                  = args.endian.upper()

if dev_mode:
    display({
        "strings_to_serialize": strings_to_serialize
        , "baud_rate": baud_rate
        , "logic_high_level": logic_high_level
        , "logic_low_level": logic_low_level
        , "waveform_delay": waveform_delay_input
        , "slew_rate_input": slew_rate_input 
        , "encoding": encoding 
        , "endian": endian 
    })


# In[47]:


import math

# Setting required variables
frequency       = 0.0
data_length     = 0
point_duration  = 0.0
symbol_duration = 0.0
waveform_delay  = 0.0
stop_bits       = int(encoding[-1])

print(f"Encoding: {encoding}")

# division by some steps for smoother curves
slew_rate_steps = 2

if not(baud_rate is None) and isinstance(baud_rate, int):    

    extra_bits_per_character = 0
    # TODO: if we want to support more encoding schemes, add logic for them here
    if encoding in ['8N1']:
        extra_bits_per_character = 1 + 0 + 1        
    elif encoding in ['8O1', '8E1']:
        extra_bits_per_character = 1 + 1 + 1

    encoded_bits = 0
    longest_string_bits = 0
    for str in strings_to_serialize:
        this_sum = len(str) * (8 + extra_bits_per_character + 1) # the last 1 is for "start of message" logic high bit
        longest_string_bits = max(longest_string_bits, this_sum)
        encoded_bits += this_sum
            
    print(f"Converted input is: {encoded_bits} symbols long")

    symbol_duration = 1 / baud_rate
    # Adjust waveform delay to be a whole number of symbols long
    waveform_delay = symbol_duration * round(waveform_delay_input / symbol_duration, 0)
    print(f"Tot. waveform delay in message: {waveform_delay * len(strings_to_serialize):.2f} ({waveform_delay:.2e} sec p. delay)")    
    print(f"Symbol duration {(symbol_duration*1e6):.2f} µs (longest string {(symbol_duration * longest_string_bits):.3f} s)")

    # Adjust the slew_rate to be a reasonable fraction of the baud rate
    if slew_rate_input != 0: 
        slew_rate = min(
            max(slew_rate_input, symbol_duration / 50),
            symbol_duration / 4
        )        
        # Harmonize so that a symbol is a whole number of symbols long
        slew_rate = (
            symbol_duration / 
            round(symbol_duration / slew_rate, 0) # Fraction of slew_rate:s per symbol_duration
        )
    print(f"Slew rate: {slew_rate:.2e} seconds")

    # When we consider a pure square wave, we can use symbol duration as the point step
    # If we use slew-rate, we want to have least 2 steps for the slew rate to do it's thing, which gives the data_length
    if slew_rate_input == 0:
        # 3 points per symbol
        data_length = 3 * int(encoded_bits + len(strings_to_serialize) * (waveform_delay / symbol_duration))
        point_duration = (1 / 3) * symbol_duration
    else:
        # points are given by the slew_rate relationship with symbols
        data_length = int((symbol_duration / slew_rate) * (encoded_bits + len(strings_to_serialize) * (waveform_delay / symbol_duration)))
        point_duration = (1 / (symbol_duration / slew_rate)) * symbol_duration
    print(f"Selected data_length: {data_length:,.0f}")

    # Frequency is a straight consequence of the baud rate and the number of symbols to send:
    frequency = baud_rate / (encoded_bits + len(strings_to_serialize) * (waveform_delay / symbol_duration))
    print(f"Selected frequency: {frequency:.6e} Hz")

    print(f"One point takes: {point_duration*1e6:.2f} µs ({symbol_duration / point_duration:.3f} points per symbol)")

else:
    raise ValueError("baud_rate is not a valid number")


# # Parsing logic
# 
# 1: Convert the string to ASCII ordinals
# 2: 

# In[48]:


s_t_s = []
for str in strings_to_serialize:
    s_t_s.append([ord(char) for char in str])
if dev_mode: display(s_t_s)

# In this code, bin(value) converts each ASCII value to a binary string, 
#   [2:] strips the 0b prefix, and zfill(8) ensures each binary sequence is 8 bits long.
# binary_values = [bin(value)[2:].zfill(8) for value in s_t_s]
# binary_values
# print(binary_values)

if dev_mode: print(f"First binary of first word: {bin(s_t_s[0][0])[2:].zfill(8)}")


# In[49]:


#Calculate standardized raise and fall sequences to append
# logic_high_level = 3.3
# logic_low_level = 0
v_steps = abs((logic_high_level - logic_low_level) / (slew_rate_steps - 1))
if(logic_high_level > logic_low_level):
    rise_sequence = [(logic_low_level + (i) * v_steps) for i in range(slew_rate_steps)]
    fall_sequence = rise_sequence.copy()
    fall_sequence.reverse()
else:
    fall_sequence = [(logic_high_level + (i) * v_steps) for i in range(slew_rate_steps)]
    rise_sequence = fall_sequence.copy()
    rise_sequence.reverse()

if dev_mode: print(rise_sequence)
if dev_mode: print(fall_sequence)


# In[50]:


# Note: point_duration and symbol_duration are set in the 'required variables' step
#   point_duration: how long each individual point approximately represents
#   symbol_duration: how long each symbol is supposed to last, given the baud rate

if dev_mode: print(f"Symbol duration: {symbol_duration:.2e}")
if dev_mode: print(f"Point duration: {point_duration:.2e}")
if dev_mode: print(f"Points per symbol: {(symbol_duration / point_duration):.1f}")
if dev_mode: print(f"Symbols per Waveform delay: {(waveform_delay / symbol_duration):.1f} ({(waveform_delay / point_duration):.1f} points)")

points_array      = [] # This is where we store all our points that we want to print
prev_bit          = (s_t_s[0][0] >> 7) & 1 # set the first bit as the starting point
point_count       = 0 # how many points have we put in so far
symbol_count      = 0 # what is the current character we are working on

# Switch how we loop through based on endianness
byte_loop_range = range(7, -1, -1) if endian == "MSB" else range(8)

# ----------------------------------
# Helper function to generate steps
# ----------------------------------
def write_symbol(bit, p_bit):

    # Careful - this function modifies global variables ;-)
    global point_count
    global symbol_count
    
    symbol_count += 1

    # Add slew_rate for sign changes
    if (slew_rate > 0):
        #Check if bit is 1 and previous was 0
        if bit & 1 and (~p_bit & 1):
            points_array.extend(rise_sequence)
            point_count += len(rise_sequence)
        #Check if bit is 0 and previous was 1
        elif (~bit & 1) and p_bit & 1:
            points_array.extend(fall_sequence)
            point_count += len(fall_sequence)
    
    # Add current symbol points
    v_target = logic_high_level if bit & 1 else logic_low_level
    while((point_count * point_duration) < (symbol_count * symbol_duration)):
        points_array.append(v_target)
        point_count += 1

# -------------------------------
# Loop through the input strings
# -------------------------------

for str in s_t_s:

    # Parse each character in str
    for char in str:

        # Start bit: logic low
        write_symbol(0, prev_bit)
        prev_bit = 0

        # Parse each bit in byte (1 character)
        bitsum = 0
        for i in byte_loop_range:
            bit = (char >> i) & 1   # Check if bit is 1 or 0
            bitsum += bit           # Add to bitsum for parity checking
            write_symbol(bit, prev_bit)
            prev_bit = bit

        # Write optional parity bits:
        if encoding[1] != 'N':

            bit = 0
            if encoding[1] == 'O':
                bit = 1 if bitsum % 2 == 0 else 0
            elif encoding[1] == 'E':
                bit = 0 if bitsum % 2 == 0 else 1            

            write_symbol(bit, prev_bit)
            prev_bit = bit
            
        # Stop bit(s): logic high
        for i in range(int(encoding[-1])):
            write_symbol(1, prev_bit)
            prev_bit = 1

    # Add a waveform_delay at the end of each string, if selected
    if waveform_delay > 0:
        # Add the amount of symbols that is the effective waveform delay
        for i in range( int(round(waveform_delay / symbol_duration, 0)) ):            
            write_symbol(1, prev_bit)
            if i == 0: prev_bit = 1

if dev_mode: display(f"Total points in output: {len(points_array):,}")
if dev_mode: display(points_array)


# In[51]:


# Fixing the last transition if needed (stopping a bit early)

#First bit of binary ASCII array is
if dev_mode: print(f"First byte: 0b{bin(s_t_s[0][0])[2:].zfill(8)}")
f_b = (s_t_s[0][0] >> 7) & 1
if dev_mode: print(f"First bit: {f_b}")

#Last bit of binary ASCII array is
if dev_mode: print(f"Last byte: 0b{bin(s_t_s[-1][-1])[2:].zfill(8)}")
l_b = (s_t_s[-1][-1] >> 0) & 1
if dev_mode: print(f"Last bit: {l_b}")

if f_b != l_b and slew_rate > 0:
    points_array = points_array[:-slew_rate_steps]
    if l_b == 1:        
        points_array.extend(fall_sequence)
    else:        
        points_array.extend(rise_sequence)


# ## Outputs
# The output is a waveform definition in .csv format, following standard set by EasyWaveX.

# In[52]:


import csv

file_path = "output.csv"

header_array = [
    ["data length", f"{ len(points_array) :.0f}"]
    , ["frequency", f"{ frequency :.6f}"]
    , ["amp", f"{ abs(logic_high_level - logic_low_level) :.6f}"]
    , ["offset", f"{ abs(logic_high_level - logic_low_level) / 2 :.6f}"]
    , ["phase", f"{0:.6f}"]
]

with open(file_path, "w", newline="") as file:
    writer = csv.writer(file, quoting=csv.QUOTE_NONE, escapechar='\\')
    writer.writerows(header_array)
    writer.writerow([]) #1
    writer.writerow([]) #2
    writer.writerow([]) #3
    writer.writerow([]) #4
    writer.writerow([]) #5
    writer.writerow([]) #6
    writer.writerow([]) #7

output_array = [
    ['xpos', 'value']
]

for index, point_val in enumerate(points_array):
    output_array.append([f"{index + 1:.0f}", f"{point_val:.6f}"])


with open(file_path, "a", newline="") as file:
    writer = csv.writer(file, quoting=csv.QUOTE_NONE, escapechar='\\')
    writer.writerows(output_array)

if dev_mode: 
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        sample_rows = [next(reader) for _ in range(20)]
        display(sample_rows)

