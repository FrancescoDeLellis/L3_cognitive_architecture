%% GENERATION OF POSITION SIGNALS
clc
close all
clearvars
n = 500; % Number of points in the trajectory
phase = linspace(0,2*pi,n);

amp = 15;
z_amp_ratio = 0.1;
initial_positions = [10 35 150 10 -70 140 10 15 0 10 -15 0];  % Check real values of hand_r, foot_l and foot_r on UE
datapoints = zeros(n,length(initial_positions));
datapoints(:,1) = initial_positions(1)*ones(n,1);
datapoints(:,2) = initial_positions(2) + initial_positions(2)*amp*cos(phase)';
datapoints(:,3) = initial_positions(3) + initial_positions(3)*amp*z_amp_ratio*sin(phase)';
datapoints(:,4:end) = repmat(initial_positions(4:end),n,1);

T = array2table(datapoints);
T.Properties.VariableNames = {'Hand_L_x', 'Hand_L_y', 'Hand_L_z', 'Hand_R_x', 'Hand_R_y', 'Hand_R_z', 'Foot_L_x', 'Foot_L_y', 'Foot_L_z', 'Foot_R_x', 'Foot_R_y', 'Foot_R_z'};
writetable(T, 'kuramoto_baseline.csv');