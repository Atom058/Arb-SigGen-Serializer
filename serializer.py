#!/usr/bin/env python
# coding: utf-8

# # About this notebook
# 
# This notebook takes a string as input, converts it to it's binary representation (using ASCII encoding), and then outputs a CSV definition of a Arbitrary Waveform to use with Siglent's series of Signal Generators.
# 
# ## Inputs
# - **string_to_serialize**: The string you want to convert
# - **slew_rate**: Requested slew rate when switching between high and low (to reduce ringing), in seconds. Minimum 500ns, maximum 1/4 of baud rate (dynamic).
# - **logic_high_level**: the logic HIGH voltage level
# - **logic_low_level**: the logic LOW voltage level
# - **baud_rate**: The baud rate of the output wave. Common values are 4800, 9600, 19200, 38400, 57600, 115200.
# 
# ### Buad rate, data length, and frequency
# The Arbitrary wave definition format uses "frequency" and "data length" as the two input variables. The defined waveform's available *points* is then spread out over the selected frequency. The number of points is equal to the data length.
# 
# Because of this, there is a inherent relationship between the selected the frequency / data length and baud rate. To compensate for this, the baud rate is converted to a equivalent frequency + data length as input.

# In[29]:


# Switching this to 'True' will enable Notebook outputs. 
# When set to 'False', the file is optimized for exporting to .py.
dev_mode = False
if dev_mode:
    dev_args = [
        '--string_to_serialize', "$GPRMC,161229.487,A,3723.2475,N,12158.3416,W,0.13,309.62,120598, ,*10\r\n"
        , '--baud_rate', '4800'
        , '--logic_high_level', '3.3'
        , '--logic_low_level', '0.0'
        , '--slew_rate', '1e-6'
    ]
    print (dev_args)


# In[27]:


import argparse

parser = argparse.ArgumentParser(
    prog="arb_siggen_serializer", 
    description="""
        Converts an input string to ASCII Binary sequence for Siglent's Arbitrary Signal Generator.
        Output is a csv file that is equivalent in format to what EasyWaveX software produces.
    """
)
parser.add_argument(
    '--string_to_serialize', '-str'
    , help='Input string'
    , type=str
    , required=True
)
parser.add_argument(
    '--baud_rate', '-br'
    , help='BAUD rate of message'
    , type=int
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
    '--slew_rate', '-slr'
    , help="""
        Requested slew rate when switching between high and low (to reduce ringing), in seconds. 
        Easiest way is to provide it in scientific notation (i.e. '1e-6' for 1µs)
        Minimum: 500ns
        Maximum: 1/4 of baud rate (dynamic)
    """
    , type=float
    , required=False
    , default=1e-5
    , dest='slew_rate_input'
)

args = parser.parse_args() if not dev_mode else parser.parse_args(dev_args)

# Inputs
string_to_serialize = args.string_to_serialize
baud_rate           = args.baud_rate
slew_rate_input     = args.slew_rate_input
logic_high_level    = args.logic_high_level
logic_low_level     = args.logic_low_level

if dev_mode:
    display({
        "string_to_serialize": string_to_serialize
        , "baud_rate": baud_rate
        , "slew_rate_input": slew_rate_input
        , "logic_high_level": logic_high_level
        , "logic_low_level": logic_low_level
    })


# In[17]:


import math

# Setting required variables
frequency       = None
data_length     = None
point_duration  = None
symbol_duration = None

# division by some steps for smoother curves
slew_rate_steps = 2

if not(baud_rate is None) and isinstance(baud_rate, int):
    encoded_bits = len(string_to_serialize) * 8
    print(f"Converted input is: {encoded_bits} bits long")

    symbol_duration = 1 / baud_rate
    print(f"Symbol duration {(symbol_duration*1e6):.2f} µs (entire message {(symbol_duration * encoded_bits):.3f} s)")
    
    slew_rate = min(
        max(slew_rate_input, 0.5e-6),
        symbol_duration / 4
    )
    print(f"Slew rate: {slew_rate:.2e} seconds")

    frequency = math.ceil(
        1 / (
            symbol_duration * encoded_bits  #Message duration
            + 2 * slew_rate                 #Headroom for slew rate
        )
        * 10
    ) / 10
    print(f"Selected frequency: {frequency} Hz")

    # We want to have least 10 steps for the slew rate to do it's thing, which gives the data_length
    data_length = math.ceil((1 / frequency) / (slew_rate / slew_rate_steps))
    print(f"Selected data_length: {data_length}")

    point_duration = (1 / frequency) / data_length
    print(f"One point takes: {point_duration:.2e} seconds")

else:
    raise ValueError("baud_rate is not a valid number")


# # Parsing logic
# 
# 1: Convert the string to ASCII ordinals
# 2: 

# In[18]:


s_t_s = [ord(char) for char in string_to_serialize]
if dev_mode: print(s_t_s)

# In this code, bin(value) converts each ASCII value to a binary string, 
#   [2:] strips the 0b prefix, and zfill(8) ensures each binary sequence is 8 bits long.
# binary_values = [bin(value)[2:].zfill(8) for value in s_t_s]
# binary_values
# print(binary_values)

if dev_mode: print(f"First binary: {bin(s_t_s[0])[2:].zfill(8)}")


# In[28]:


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


# In[46]:


# Note: point_duration and symbol_duration are set in the 'required variables' step
#   point_duration: how long each individual point approximately represents
#   symbol_duration: how long each symbol is supposed to last, given the baud rate

if dev_mode: print(f"Symbol duration: {symbol_duration}")
if dev_mode: print(f"Point duration: {point_duration}")
if dev_mode: print(f"Approx. points per symbol: {(symbol_duration / point_duration):.2f}")

points_array = []
prev_bit          = (s_t_s[0] >> 7) & 1 # set the first bit as the starting point
point_count       = 0 # how many points have we put in so far
symbol_count      = 0 # what is the current character we are working on

for char in s_t_s:
    for i in range(7, -1, -1):
        symbol_count += 1
        bit = (char >> i) & 1   #Check if bit is 1 or 0
        
        #Debug printing
        # print(f"pos{i}:{bit}")
        # points_array.append(90 + bit)

        #Check if bit is 1 and privious was 0
        if bit & 1 and (~prev_bit & 1):
            points_array.extend(rise_sequence)
            point_count += slew_rate_steps
        #Check if bit is 0 and privious was 1
        elif (~bit & 1) and prev_bit & 1:
            points_array.extend(fall_sequence)
            point_count += slew_rate_steps

        v_target = logic_high_level if bit & 1 else logic_low_level
        while((point_count * point_duration) < (symbol_duration * symbol_count)):
            points_array.append(v_target)
            point_count += 1
        prev_bit = bit

if dev_mode: display(points_array)


# In[47]:


# Fixing the last transition if needed (shopping a bit early)

#First bit of binary ASCII array is
if dev_mode: print(f"First bit: 0b{bin(s_t_s[0])[2:].zfill(8)}")
f_b = (s_t_s[0] >> 7) & 1
if dev_mode: print(f_b)

#Last bit of binary ASCII array is
if dev_mode: print(f"Last bit: 0b{bin(s_t_s[-1])[2:].zfill(8)}")
l_b = (s_t_s[-1] >> 0) & 1
if dev_mode: print(l_b)

if f_b != l_b:
    points_array = points_array[:-slew_rate_steps]
    if l_b == 1:        
        points_array.extend(fall_sequence)
    else:        
        points_array.extend(rise_sequence)


# ## Outputs
# The output is a waveform definition in .csv format, following standard set by EasyWaveX.

# In[48]:


import csv

output_array = [
    ["data length", f"{ data_length :.0f}"]
    , ["frequency", f"{ frequency :.6f}"]
    , ["amp", f"{ abs(logic_high_level - logic_low_level) / 2 :.6f}"]
    , ["offset", f"{ abs(logic_high_level - logic_low_level) / 2 :.6f}"]
    , ["phase", f"{0:.6f}"]
    # , [''] #1
    # , [''] #2
    # , [''] #3
    # , [''] #4
    # , [''] #5
    # , [''] #6
    # , [''] #7
    , ['xpos', 'value']
]

for index, point_val in enumerate(points_array):
    output_array.append([f"{index + 1:.0f}", f"{point_val:.6f}"])

if dev_mode: display(output_array[:100])

with open("output.csv", "w", newline="") as file:
    writer = csv.writer(file, quoting=csv.QUOTE_NONE, escapechar='\\')
    writer.writerows(output_array)

