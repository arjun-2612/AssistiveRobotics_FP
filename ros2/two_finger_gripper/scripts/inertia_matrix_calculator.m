% Arjun Viswanathan
% 3/14/25
% Inertia matrix calculator

clear 
clc

% Base link, Body, Hip, Upper Leg, Lower Leg, Ankle, Foot, Propeller
m = [0.24728; 3.34; 0.04828; 0.46; 0.0604; 0.05489; 0.0747; 0.218];
% m = [0.24728; 2.84; 0.04828; 0.46; 0.0604; 0.05489; 0.0747; 0.0];
w = [0.28; 0.4; 0.09; 0.09; 0.3; 0.03; 0.06; 0.05];
h = [0.15; 0.14; 0.09; 0.065; 0.025; 0.025; 0.04; 0.05];
d = [0.08; 0.17; 0.09; 0.28; 0.07; 0.13; 0.125; 0.05];

text = ["Base Link", "Body", "Hip", "Upper Leg", "Lower Leg", "Ankle", "Foot", "Propeller"];

for i = 1:8
    Ixx = (1/12)*m(i)*(h(i)^2 + d(i)^2);
    Iyy = (1/12)*m(i)*(w(i)^2 + d(i)^2);
    Izz = (1/12)*m(i)*(w(i)^2 + h(i)^2);
    I = [Ixx Iyy Izz];

    disp(text(i))
    disp(I)
end