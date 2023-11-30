#!/usr/bin/octave -qf

% Plot test data
% Author: Jørgen Kragh Jakobsen

% Load data
data = load('test_log_29_11_2023.dat');
data = data';
subplot(2,1,1);
plot(log(data(1,:))./10);
subplot(2,1,2);
plot((data(2,:))); 
pause;
