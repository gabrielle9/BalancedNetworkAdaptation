% Simulation of spiking model autoencoder in Sophie's latest draft of paper
% Based on derivation from 26 June, 2017 in labnotebook
% Updated from derivation on 6 July 2017 - uses cost with ra instead of p
% re-scaled orientation axis due to onservations on 1 Aug 2017
% Updated on 23 July 2018 to include linear cost as agreed in April at Berkeley meeting

clearvars
savestr = 'OM_tilt_arrows_31July18';

%% Time Structure
dt = 0.01; %units of ms
time = 0:dt:3e3;
Tsteps = length(time);

%% Parameters

Nj = 2; %number of dimensions in input
Nn = 200; %number of neurons
tau = 5; %5; %10; %100*dt; %shorter tau gives estimate more variance, recruits more neurons
tau_a = 2000; %1000; %200; %1000*dt;
mu = 0.1;%0.002; %20; %0.2;
eta = 0;

asgain = 25;
sgain = 5; %5; %to get same results from transformation network with sgain=10 and tau=5
atime = 2:round(2e3/dt);
stime = round(2e3/dt):round(2.25e3/dt);

%% Connectivity Structure
Nw = Nn/2;
theta = 0:2*pi/Nw:2*pi-2*pi/Nw; % one period
theta = reshape(ones(2,1)*theta,1,[]); %make it Nn long again

bgain1 = 3;  %3;%1;%3;
bgain2 = 9; %15; %9;%6;%15;
bgain = reshape([bgain1;bgain2]*ones(1,Nw),1,[]); %interspersed gains

W = [cos(theta).*bgain; sin(theta).*bgain]; % these need to cover a full period

Gain = diag(diag(2./(W'*W + mu)));
lin_cost = eta.*diag(Gain);

thresh = 1;

%% Inputs and stuff (a 180 stim world but this doesn't do anything to scale the responses)
% also, all angles need to be shifted so that test angle is vertical

Np = 20;
% thetp = 0:2*pi/Np:2*pi-2*pi/Np;
thetp = 0:(pi/Np):pi-(pi/Np);
% adthet1 = thetp(5); %72 deg should be repulsive
% adthet2 = thetp(2); %18 deg should be attractive
% test_thet = thetp(6); %thetp(0); %pi/2; %thetp(6); %90 deg

% test_thet = thetp(11); %pi/2; %thetp(6); %90 deg
% adthet1 = thetp(11-4); %72 deg should be repulsive
% adthet2 = thetp(); %18 deg should be attractive

test_thet = pi; %which is thetp(20) or thetp(end)
adthet1 = thetp(16); %repulsive
adthet2 = thetp(7); %attractive

save(savestr)

%% Initialize with silence
itime = 0:dt:0.5e3;
iTsteps = length(itime);

s = zeros(Nj,iTsteps);
O = zeros(Nn,iTsteps);
V = zeros(Nn,iTsteps);
sest = zeros(Nj,iTsteps);
ra = zeros(Nn,iTsteps);

ds = [zeros(Nj,1) diff(s,1,2)]; %if s has extra dimensions
Input = Gain*W'*(s + tau.*ds); % weighted input
clear s

for t = 2:iTsteps
    dVdt = -V(:,t-1) + Input(:,t-1) - tau.*Gain*(W'*W)*O(:,t-1) - tau.*mu.*Gain*O(:,t-1) + ((tau/tau_a)-1)*mu.*Gain*ra(:,t-1); %this is dV/dt
    V(:,t) = V(:,t-1) + dt.*(dVdt./tau);
    
    adThr = thresh.*ones(Nn,1) + lin_cost;
    O(:,t) = (V(:,t)>=adThr)./dt; %spikes are placed on neurons that have crossed threshold
    
    if sum(O(:,t))>(1/dt)
        [~,vi] = max(V(:,t) - adThr);
        O(:,t) = 0;
        O(vi,t) = 1/dt; %to ensure only one spike per time step
    end
    
    dra = -(1/tau_a)*ra(:,t-1) + O(:,t-1);
    ra(:,t) = ra(:,t-1) + dt.*dra;
        
    dsest = -(1/tau)*sest(:,t-1) + W*O(:,t-1);
    sest(:,t) = sest(:,t-1) + dt.*dsest;
end

O00 = O(:,t);
V00 = V(:,t);
sest00 = sest(:,t);
ra00 = ra(:,t);

save(savestr,'O00','V00','sest00','ra00','-append')

%% Silence then test
O = zeros(Nn,Tsteps);
V = zeros(Nn,Tsteps);
sest = zeros(Nj,Tsteps);
ra = zeros(Nn,Tsteps);

O(:,1) = O00;
V(:,1) = V00;
sest(:,1) = sest00;
ra(:,1) = ra00;
    
s = zeros(Nj,length(time)); %command input.
% s(1,stime) = cos(2*test_thet)*sgain;
% s(2,stime) = sin(2*test_thet)*sgain;
s(1,stime) = cos(test_thet)*sgain;
s(2,stime) = sin(test_thet)*sgain;
    
ds = [zeros(Nj,1) diff(s,1,2)]; %if s has extra dimensions
Input = Gain*W'*(s + tau.*ds);

for t = 2:length(time)
    dVdt = -V(:,t-1) + Input(:,t-1) - tau.*Gain*W'*W*O(:,t-1) - tau.*mu.*Gain*O(:,t-1) + ((tau/tau_a)-1).*mu.*Gain*ra(:,t-1);
    V(:,t) = V(:,t-1) + dt.*(dVdt./tau);
    
    adThr = thresh.*ones(Nn,1) + lin_cost;
    O(:,t) = (V(:,t)>=adThr)./dt; %spikes are placed on neurons that have crossed threshold

    if sum(O(:,t))>(1/dt)
        [~,vi] = max(V(:,t) - adThr);
        O(:,t) = 0;
        O(vi,t) = 1/dt; %to ensure only one spike per time step
    end
        
    dra = -(1/tau_a)*ra(:,t-1) + O(:,t-1);
    ra(:,t) = ra(:,t-1) + dt.*dra;

    dsest = -(1/tau)*sest(:,t-1) + W*O(:,t-1);
    sest(:,t) = sest(:,t-1) + dt.*dsest;
    
end

mdecxe1 = DecodeTheta(nanmean(sest(:,stime),2));

save(savestr,'mdecxe1','-append')

%% Adapt near then test
adthet = adthet1;
O = zeros(Nn,Tsteps);
V = zeros(Nn,Tsteps);
sest = zeros(Nj,Tsteps);
ra = zeros(Nn,Tsteps);

O(:,1) = O00;
V(:,1) = V00;
sest(:,1) = sest00;
ra(:,1) = ra00;
    
s = zeros(Nj,length(time)); %command input.
% s(1,atime) = cos(2*adthet)*asgain;
% s(2,atime) = sin(2*adthet)*asgain;
% s(1,stime) = cos(2*test_thet)*sgain;
% s(2,stime) = sin(2*test_thet)*sgain;
s(1,atime) = cos(adthet)*asgain;
s(2,atime) = sin(adthet)*asgain;
s(1,stime) = cos(test_thet)*sgain;
s(2,stime) = sin(test_thet)*sgain;

ds = [zeros(Nj,1) diff(s,1,2)]; %if s has extra dimensions
Input = Gain*W'*(s + tau.*ds);

for t = 2:length(time)
    dVdt = -V(:,t-1) + Input(:,t-1) - tau.*Gain*(W'*W)*O(:,t-1) - tau.*mu.*Gain*O(:,t-1) + ((tau/tau_a)-1).*mu.*Gain*ra(:,t-1);
    V(:,t) = V(:,t-1) + dt.*(dVdt./tau);
    
    adThr = thresh.*ones(Nn,1) + lin_cost;
    O(:,t) = (V(:,t)>=adThr)./dt; %spikes are placed on neurons that have crossed threshold

    if sum(O(:,t))>(1/dt)
        [~,vi] = max(V(:,t) - adThr);
        O(:,t) = 0;
        O(vi,t) = 1/dt; %to ensure only one spike per time step
    end
    
    dra = -(1/tau_a)*ra(:,t-1) + O(:,t-1);
    ra(:,t) = ra(:,t-1) + dt.*dra;

    dsest = -(1/tau)*sest(:,t-1) + W*O(:,t-1);
    sest(:,t) = sest(:,t-1) + dt.*dsest;
    
end

mdecxe2 = DecodeTheta(nanmean(sest(:,stime),2));

save(savestr,'mdecxe2','-append')

%% Adapt oblique then test 
adthet = adthet2;
O = zeros(Nn,Tsteps);
V = zeros(Nn,Tsteps);
sest = zeros(Nj,Tsteps);
ra = zeros(Nn,Tsteps);

O(:,1) = O00;
V(:,1) = V00;
sest(:,1) = sest00;
ra(:,1) = ra00;
    
s = zeros(Nj,length(time)); %command input.
% s(1,atime) = cos(2*adthet)*asgain;
% s(2,atime) = sin(2*adthet)*asgain;
% s(1,stime) = cos(2*test_thet)*sgain;
% s(2,stime) = sin(2*test_thet)*sgain;
s(1,atime) = cos(adthet)*asgain;
s(2,atime) = sin(adthet)*asgain;
s(1,stime) = cos(test_thet)*sgain;
s(2,stime) = sin(test_thet)*sgain;

ds = [zeros(Nj,1) diff(s,1,2)]; %if s has extra dimensions
Input = Gain*W'*(s + tau.*ds);

for t = 2:length(time)
    dVdt = -V(:,t-1) + Input(:,t-1) - tau.*Gain*(W'*W)*O(:,t-1) - tau.*mu.*Gain*O(:,t-1) + ((tau/tau_a)-1).*mu.*Gain*ra(:,t-1);
    V(:,t) = V(:,t-1) + dt.*(dVdt./tau);
    
    adThr = thresh.*ones(Nn,1) + lin_cost;
    O(:,t) = (V(:,t)>=adThr)./dt; %spikes are placed on neurons that have crossed threshold

    if sum(O(:,t))>(1/dt)
        [~,vi] = max(V(:,t) - adThr);
        O(:,t) = 0;
        O(vi,t) = 1/dt; %to ensure only one spike per time step
    end
    
    dra = -(1/tau_a)*ra(:,t-1) + O(:,t-1);
    ra(:,t) = ra(:,t-1) + dt.*dra;

    dsest = -(1/tau)*sest(:,t-1) + W*O(:,t-1);
    sest(:,t) = sest(:,t-1) + dt.*dsest;
    
end

mdecxe3 = DecodeTheta(nanmean(sest(:,stime),2));

save(savestr,'mdecxe3','-append')

%% Scale and Plot
test_thet_sc = test_thet/2;
adthet1_sc = adthet1/2;
adthet2_sc = adthet2/2;

mdecxe_ch1 = mdecxe1./2; %scale decoded angle by 1/2 to correspond to tilt illusion axis
mxe_ch1 = [cos(mdecxe_ch1); sin(mdecxe_ch1)];
nmxe_ch1 = mxe_ch1./norm(mxe_ch1);

mdecxe_ch2 = mdecxe2./2;
mxe_ch2 = [cos(mdecxe_ch2); sin(mdecxe_ch2)];
nmxe_ch2 = mxe_ch2./norm(mxe_ch2);

mdecxe_ch3 = mdecxe3./2;
mxe_ch3 = [cos(mdecxe_ch3); sin(mdecxe_ch3)];
nmxe_ch3 = mxe_ch3./norm(mxe_ch3);

save(savestr,'test_thet_sc','adthet1_sc','adthet2_sc','mdecxe_ch1','mdecxe_ch2','mdecxe_ch3','nmxe_ch1','nmxe_ch2','nmxe_ch3','-append')


figure('WindowStyle','docked');
subplot(1,3,1); hold on
quiver(0,0,cos(test_thet_sc),sin(test_thet_sc),'Color',[1 0.1 0],'LineWidth',3)
quiver(0,0,nmxe_ch1(1,1),nmxe_ch1(2,1),'Color',[0.5 0.9 0],'LineWidth',2)
axis square
set(gca,'YTick',[0:1],'XTick',[0:1])
set(gca,'FontSize',14)
xlim([-0.2 1])
ylim([-0.2 1])
grid on
title('no adaptation')

subplot(1,3,2); hold on
quiver(0,0,cos(adthet1_sc),sin(adthet1_sc),'Color',[0.5 0.5 0.5],'LineWidth',2)
quiver(0,0,cos(test_thet_sc),sin(test_thet_sc),'Color',[1 0.1 0],'LineWidth',3)
quiver(0,0,nmxe_ch2(1,1),nmxe_ch2(2,1),'Color',[0.5 0.9 0],'LineWidth',2)
axis square
set(gca,'YTick',[0:1],'XTick',[0:1])
set(gca,'FontSize',14)
xlim([-0.2 1])
ylim([-0.2 1])
grid on
title('adaptation to a near orientation')

subplot(1,3,3); hold on
quiver(0,0,cos(adthet2_sc),sin(adthet2_sc),'Color',[0.5 0.5 0.5],'LineWidth',2)
quiver(0,0,cos(test_thet_sc),sin(test_thet_sc),'Color',[1 0.1 0],'LineWidth',3)
quiver(0,0,nmxe_ch3(1,1),nmxe_ch3(2,1),'Color',[0.5 0.9 0],'LineWidth',2)
axis square
set(gca,'YTick',[0:1],'XTick',[0:1])
set(gca,'FontSize',14)
title('adaptation to an oblique orientation')
grid on
xlim([-0.2 1])
ylim([-0.2 1])
legend('adaptor','test orientation','network estimate')

