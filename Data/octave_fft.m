pkg load signal; 
clear, clc, close all
[x, Fs] = audioread(file_in_loadpath("obj/out006.wav"));
x = x(:, 1);                        
xlen = length(x);                   
t = (0:length(x)-1)/Fs;
plot(t*1E+3, x);
step = fix(5*Fs/1000);     # one spectral slice every 5 ms
window = fix(40*Fs/1000);  # 40 ms data window
fftn = 2^nextpow2(window); # next highest power of 2
[S, f, t] = specgram(x, fftn, Fs, window, window-step);
S = abs(S(2:fftn*4000/Fs,:)); # magnitude in range 0<f<=4000 Hz.
S = S/max(S(:));           # normalize magnitude so that max is 0 dB.
S = max(S, 10^(-40/10));   # clip below -40 dB.
S = min(S, 10^(-3/10));    # clip above -3 dB.
imagesc (t, f, log(S));    # display in log scale
set (gca, "ydir", "normal"); # put the 'y' direction in the correct direction
