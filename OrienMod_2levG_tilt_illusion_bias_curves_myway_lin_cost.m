% Figure 9: Decay of tilt illusion after a long adaptation
% Network is adapted to an orientation and then decoder error is measured
% after various delays following the adaptation.
% This is repeated for a series of adapting orientations.
% Updated with newest model version

clearvars

% savstr = 'OM_tilt_decay_2levG_28Jul17'; %what you want to name the file that will be saved
savstr = 'OM_tilt_decay_2levG_24Jul18'; %what you want to name the file that will be saved

%% Parameters
Nj = 2; % # of dynamical variables
Nn = 200;%50;%24;%100; %100; %6; %96; %25; %100; % # of neurons

tau = 5;
tau_a = 2000; %1000;
mu = 0.1; %0.2; %0.5; %0.05; %1200000; %2; %0.5;
eta = 0;

asgain = 25;%1.5; %0.75;%0.5;
sgain = 5;

dt = 0.01; % 1/10th ms
% Tstim = 100; %500; %250; %2000; %ms
% Tspace = 0; %50; %1000;
% Tstimd = round(Tstim/dt);
% Tspaced = round(Tspace/dt);
Nta = 40; %30;%12; %6;%12; %25; % # of adaptors to test. Increase for finer resolution curve
% Tsteps_init = Nta*(Tstimd+Tspaced);% + round(adpt/dt);
Tsteps_init = round(500/dt); %500 ms init

thet = 0:2*pi/Nta:2*pi-2*pi/Nta;

testi = 1; %index of test orientation
thep_test = thet(testi);
thet_shift = circshift(thet,[0,-testi+1]); %shifted so that thep_test is the 1st element

%% Weights
%neuron receptive fields are evenly distributed about the unit circle
Nw = Nn/2;
theta = 0:2*pi/Nw:2*pi-2*pi/Nw; % one period
theta = reshape(ones(2,1)*theta,1,[]); %make it Nn long again

bgain1 = 3;
bgain2 = 9;
bgain = reshape([bgain1;bgain2]*ones(1,Nw),1,[]); %interspersed gains

W = [cos(theta).*bgain; sin(theta).*bgain];
Gain = diag(diag(2./(W'*W + mu)));

thresh = 1; %(mu + diag(W'*W)')/2;
lin_cost = eta.*diag(Gain);

save(savstr)

%% initialize with silence
s = zeros(Nj,Tsteps_init);
O = zeros(Nn,Tsteps_init);
V = zeros(Nn,Tsteps_init);
sest = zeros(Nj,Tsteps_init);
ra = zeros(Nn,Tsteps_init);

% trp = randperm(length(thet));
% thetr = thet(trp);
% 
% for ts = 1:Nta
%     stimON = ((ts-1)*(Tstimd+Tspaced)+Tspaced)+1;
%     stimOFF = ts*(Tstimd+Tspaced);
%     thep = thetr(ts);
%     s(:,stimON:stimOFF) = [ones(1,Tstimd)*cos(thep)*sgain; ones(1,Tstimd)*sin(thep)*sgain];
% end
ds = [zeros(Nj,1) diff(s,1,2)]; %if s has extra dimensions
Input = Gain*W'*(s + tau.*ds); % weighted input
clear s

for t = 2:Tsteps_init
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

save('OM_init')

%% long adaptation and tests after various delays
delays = 1; %3;
mdecxe = zeros(delays,round(Nta/2));
% dec_err = zeros(delays,round(Nta/2));

clear s Input O V sest ra

adp_time = 0:dt:2e3; %6e3;
Tsteps_adpt = length(adp_time);

for q = 2:round(Nta/2)+1 % loop through half of the orientations as adaptors
    s = zeros(Nj,Tsteps_adpt);
    O = zeros(Nn,Tsteps_adpt);
    V = zeros(Nn,Tsteps_adpt);
    sest = zeros(Nj,Tsteps_adpt);
    ra = zeros(Nn,Tsteps_adpt);
    
    O(:,1) = O00;
    V(:,1) = V00;
    sest(:,1) = sest00;
    ra(:,1) = ra00;
    
    thea = thet_shift(q); %adapting orientation
    s(1,:) = cos(thea)*asgain;
    s(2,:) = sin(thea)*asgain;    
    ds = [zeros(Nj,1) diff(s,1,2)]; %if s has extra dimensions
    Input = Gain*W'*(s + tau.*ds); % weighted input
    clear s
    
    for t = 2:Tsteps_adpt
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
    
    O0 = O(:,t);
    V0 = V(:,t);
    sest0 = sest(:,t);
    ra0 = ra(:,t);
    
    % Test pulse after varied delays
    clear s Input O V sest ra
    for tr = 1:delays
        
        time = 0:dt:((tr-1)*2e3)+(0.25e3); %250 ms
        Tsteps_run = length(time);
        test_bl = round((tr-1)*2e3/dt)+1:length(time);
        
        s = zeros(Nj,Tsteps_run);
        O = zeros(Nn,Tsteps_run);
        V = zeros(Nn,Tsteps_run);
        sest = zeros(Nj,Tsteps_run);
        ra = zeros(Nn,Tsteps_run);
        
        O(:,1) = O0;
        V(:,1) = V0;
        sest(:,1) = sest0;
        ra(:,1) = ra0;
        
        s(:,test_bl) = [ones(1,length(test_bl))*cos(thep_test)*sgain; ones(1,length(test_bl))*sin(thep_test)*sgain];
        ds = [zeros(Nj,1) diff(s,1,2)]; %if s has extra dimensions
        Input = Gain*W'*(s + tau.*ds); % weighted input
        clear s
        
        for t = 2:Tsteps_run
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
        mdecxe(tr,q-1) = DecodeTheta(nanmean(sest(:,test_bl),2));
        clear s Input O V sest ra
    end
end

diff_thet = mdecxe-thep_test;
abdt = abs(diff_thet);
abdt_shift = abs(abdt-(2*pi));
dec_err = diff_thet-(2*pi);
dec_err(abdt<abdt_shift) = diff_thet(abdt<abdt_shift);

% save(savstr,'thetr','mdecxe','dec_err','delays','-append')
save(savstr,'mdecxe','dec_err','delays','-append')

% the effect of this axis manipulation is to collapse the space from 2*pi
% to pi and to convert it to degrees
thet_axis = thet_shift-thep_test;
thet_axis(thet_axis<0) = thet_axis(thet_axis<0)+(2*pi);
thet_ax_deg = thet_axis.*360./(4*pi);
dec_err_deg = dec_err.*360./(4*pi);

figure('WindowStyle','docked')
% plot(thet_axis(2:round(Nta/2)+1),dec_err,'o','LineWidth',2)
plot(thet_ax_deg(2:round(Nta/2)+1),dec_err_deg,'o','LineWidth',2)
grid on
xlabel('Angle between adaptor and test orientation (deg)')
ylabel('Angle between test adaptor and decoded orientation (deg)')
title({['Tilt 2levG, \mu=' num2str(mu) ', \tau=' num2str(tau) ', Nn=' num2str(Nn) ', adp gain=' num2str(asgain) ', stim gain=' num2str(sgain)],['2sec adaptation, eta=' num2str(eta)]})
set(gca,'FontSize',16)
% set(gcf,'Position',[300 320 525 300])

