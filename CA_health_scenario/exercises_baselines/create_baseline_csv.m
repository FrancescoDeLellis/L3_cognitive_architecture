%% GENERATION OF POSITION SIGNALS
clc
close all
clearvars
n = 30; % Number of points in the trajectory
phase = linspace(0,2*pi,n);

amp = 15*2;  %15
z_amp_ratio = 0.1;
initial_positions = [0 35 135 -47.768318 22.22 104.4664 -15.777701 13.938243 0.751431 22.339117 13.306132 2.390061];
%initial_positions = [7.180022 49.890251 151.600441 -47.768318 22.22 104.466431];
datapoints = zeros(n,length(initial_positions)+1);
datapoints(:,1) = phase';
datapoints(:,2) = initial_positions(1)*ones(n,1);
datapoints(:,3) = initial_positions(2) + amp*cos(phase)';
datapoints(:,4) = initial_positions(3) + amp*z_amp_ratio*sin(phase)';
datapoints(:,5:end) = repmat(initial_positions(4:end),n,1);

T = array2table(datapoints);
T.Properties.VariableNames = {'Phase','Hand_L_x', 'Hand_L_y', 'Hand_L_z', 'Hand_R_x', 'Hand_R_y', 'Hand_R_z', 'Foot_L_x', 'Foot_L_y', 'Foot_L_z', 'Foot_R_x', 'Foot_R_y', 'Foot_R_z'};
%T.Properties.VariableNames = {'Phase','Hand_L_x', 'Hand_L_y', 'Hand_L_z', 'Hand_R_x', 'Hand_R_y', 'Hand_R_z'};
writetable(T, 'kuramoto_baseline.csv');