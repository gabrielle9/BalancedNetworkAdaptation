% The letter figure, a.k.a. the pattern plots. Now as digital clock face.
% Nothing was changed since 15 Aug 2017. Just had to rerun to save new
% figure with balanced net + ad

% Train random network
% Obtain optimal decoder
% Run trained random net on stim set
% Run trained random net with adaptation on stim set
% Run balanced net with adaptation on stim set

clearvars
%% Time Structure
dt = 0.02; %0.01; %units of ms

%% Parameters
Nj = 7; %number of dimensions in input, digital clock bar positions
Nn = 400; %200; %400 %number of neurons - keep it even for simplicity

tau = 5;
tau_a = 2000; %1000;
mu = 0.02; %0.01; %0.005; %300; %1e-6; %1e-6/msdt; %quadratic cost, adaptation parameter

%noise parameters
% sigv = 0; %0.001; %5; %1e-3; %1e-3/msdt; 
% % sigs = 0; %0.01/msdt;
% sigth = 0; %0.0005; %0.5; %0.1;

%% Connectivity Structure
% W = rand(Nj,Nn)-0.5; %+/- weights
W = 2.*(rand(Nj,Nn)-0.5); %+/- weights between [-1,1]

W_rec = W'*W;
Gain = diag(diag(2./(W'*W + mu))); %diag matrix

Wd = diag(diag(W_rec)); % matrix of only diags
di = find(Wd); %indices of diagonal

Wline = reshape(W_rec,1,[]);
mW = mean(Wline);
stdevW = std(Wline);

% W_randrec = normrnd(mW,stdevW,[Nn Nn]);
W_randrec = normrnd(mW,stdevW,[Nn Nn])./Nn;
W_randrec(di) = diag(W_rec);

thresh = 1;

%% Stim set up
stim_time = 0:dt:3e2; % 300ms, time of each stim presentation with off time
Nstime = length(stim_time);
stimON = 1+round(1e2/dt):Nstime; %time block for stim preceded by silent space

Ntrainst = 100; %50; %20; %50; % # of training stim to present
rtrain = zeros(Nn,Ntrainst*Nstime);
strain = zeros(Nj,Ntrainst*Nstime);

%% Training of random unbalanced weights network
for rti = 1:Ntrainst
%     thNoise = randn(Nn,Nstime)*sigth;
%     vNoise = randn(Nn,Nstime)*sigv;
    
    rand_stim = rand(1,7);
    rand_gain = 4*rand(1,1);
    s = zeros(Nj,length(stim_time)); %this is the input. For now, it's just zeros.
%     s(:,stimON) = 2.*rand_stim'*ones(1,length(stimON)); %so that test stim corresponds to mean training stim
    s(:,stimON) = rand_gain.*rand_stim'*ones(1,length(stimON)); %so that test stim corresponds to mean training stim

    O = zeros(Nn,length(stim_time));
    r = zeros(Nn,length(stim_time));
    V = zeros(Nn,length(stim_time));

    for t = 2:length(stim_time)        
        dVdt = -V(:,t-1) + W'*s(:,t-1) - W_randrec*O(:,t-1);
        V(:,t) = V(:,t-1) + dt.*(dVdt./tau);

        adThr = thresh.*ones(Nn,1);
        O(:,t) = (V(:,t)>=adThr)./dt; %spikes are placed on neurons that have crossed threshold
        
        if sum(O(:,t))>(1/dt)
            [~,vi] = max(V(:,t) - adThr);
            O(:,t) = 0;
            O(vi,t) = 1/dt; %to ensure only one spike per time step
        end
        
        dr = -(1/tau)*r(:,t-1) + O(:,t-1);
        r(:,t) = r(:,t-1) + dt.*dr;        
    end
    
    rtrain(:,1+((rti-1)*Nstime):rti*Nstime) = r;
    strain(:,1+((rti-1)*Nstime):rti*Nstime) = s;
end

figure('WindowStyle','docked')
Otex = (((1:Nn)'*ones(1,length(stim_time))).*O.*dt)';
Otex(Otex==0)=nan;
plot(stim_time,Otex,'.')

%% Estimate best decoder using pseudoinverse
pi_rO = pinv(rtrain');
decW = pi_rO*strain';

clear pi_rO strain rtrain
save('SM_7J_randrec_trained_params')

%% Inputs (and noise) for test after training
% this is one long test block with several concatenated spaced stimuli
time = 0:dt:24e2;

% thNoise = randn(Nn,length(time))*sigth;
% vNoise = randn(Nn,length(time))*sigv;

% stim_nums = randperm(10)-1; %numbers 0 through 9
% stim_nums = stim_nums(1:8); %8 random test digits
% stim_nums = [3,6,7,2,0,3,3,3];
stim_nums = [5,7,8,3,0,8,6,8];
stimONb = 1+round(1e2/dt):round(3e2/dt); %200 ms chunks of stim preceded by 100ms of silence
s = zeros(Nj,length(time)); %this is the input. For now, it's just zeros.

% stimuli are digits that were not used for training
for si = 1:8
    s(:,stimONb+(si-1)*(Nstime-1)) = digi_clock_stim(stim_nums(si))'*ones(1,length(stimONb));
end
ds = [zeros(Nj,1) diff(s,1,2)]; %if s has extra dimensions

%% Test on random recurrent network
O = zeros(Nn,length(time));
r = zeros(Nn,length(time));
V = zeros(Nn,length(time));

for t = 2:length(time)
    dVdt = -V(:,t-1) + W'*s(:,t-1) - W_randrec*O(:,t-1);
    V(:,t) = V(:,t-1) + dt.*(dVdt./tau);
    
    adThr = thresh.*ones(Nn,1);
    O(:,t) = (V(:,t)>=adThr)./dt; %spikes are placed on neurons that have crossed threshold
    
    if sum(O(:,t))>(1/dt)
        [~,vi] = max(V(:,t) - adThr);
        O(:,t) = 0;
        O(vi,t) = 1/dt; %to ensure only one spike per time step
    end
    
    dr = -(1/tau)*r(:,t-1) + O(:,t-1);
    r(:,t) = r(:,t-1) + dt.*dr;
end

s_hat = decW'*r;

% figure('WindowStyle','docked')
% Otex = (((1:Nn)'*ones(1,length(time))).*O.*dt)';
% Otex(Otex==0)=nan;
% plot(time,Otex,'.')

%% Plot digi
% stimONbl = 1+round(2e2/dt):round(3e2/dt); %last 100 ms chunk of stim
stimONbl = stimONb;
frows = 4;

for ind = 1:8
    mx_all(:,ind) = mean(s(:,stimONbl+(ind-1)*(Nstime-1)),2);
    mxest_all(:,ind) = mean(s_hat(:,stimONbl+(ind-1)*(Nstime-1)),2);
end

maxest = max(max(mxest_all));
% cxmax = 2*ceil(maxest); %3; %maxest;
cxmax = 1.75*ceil(maxest); %3; %maxest;

digi_fig = figure;

for dg = 1:8
    subplot(frows,8,dg)
    colormap(hot)
    caxis([0 cxmax])
    set(gca,'Color','k')
    mx_dig = mx_all(:,dg);
    digi_display(mx_dig);
    axis off
end
subplot(frows,8,1)
title('stim','Color','w')

for test_dgi = 1:8    
    subplot(frows,8,8+test_dgi) %first digit
    colormap(hot)
    caxis([0 cxmax])
    set(gca,'Color','k')
    mxest_test_dig = mxest_all(:,test_dgi);
    digi_display(mxest_test_dig);
    axis off
end
subplot(frows,8,9)
title('trained rand rec','Color','w')

%% Test on random recurrent net with adaptation
O = zeros(Nn,length(time));
V = zeros(Nn,length(time));
r = zeros(Nn,length(time));
ra = zeros(Nn,length(time));

for t = 2:length(time)
    dVdt = -V(:,t-1) + W'*s(:,t-1) - W_randrec*O(:,t-1);% - mu*ra(:,t-1);
    V(:,t) = V(:,t-1) + dt.*(dVdt./tau);
    
    
    adThr = thresh.*ones(Nn,1) + mu*ra(:,t-1);
    O(:,t) = (V(:,t)>=adThr)./dt; %spikes are placed on neurons that have crossed threshold
    
    if sum(O(:,t))>(1/dt)
        [~,vi] = max(V(:,t) - adThr);
        O(:,t) = 0;
        O(vi,t) = 1/dt; %to ensure only one spike per time step
    end
    
    dr = -(1/tau)*r(:,t-1) + O(:,t-1);
    r(:,t) = r(:,t-1) + dt.*dr;
    
    dra = -(1/tau_a)*ra(:,t-1) + O(:,t-1);
    ra(:,t) = ra(:,t-1) + dt.*dra;
end

s_hat_a = decW'*r; %use decoder from non-adapting feedforward network

% figure('WindowStyle','docked')
% Otex = (((1:Nn)'*ones(1,length(time))).*O.*dt)';
% Otex(Otex==0)=nan;
% plot(time,Otex,'.')

%%
% mxest1a = mean(s_hat(:,10001:15000),2);
% mxest2a = mean(s_hat(:,25001:30000),2);
% mxest3a = mean(s_hat(:,40001:45000),2);
% mxest4a = mean(s_hat(:,55001:60000),2);
% mxest5a = mean(s_hat(:,70001:75000),2);
% mxest6a = mean(s_hat(:,85001:90000),2);
% mxest7a = mean(s_hat(:,100001:105000),2);
% mxest8a = mean(s_hat(:,115001:120000),2);
% mxest_all_a = [mxest1a mxest2a mxest3a mxest4a mxest5a mxest6a mxest7a mxest8a];
% maxest_a = max(max(mxest_all_a));

for ind = 1:8
    mxest_all_a(:,ind) = mean(s_hat_a(:,stimONbl+(ind-1)*(Nstime-1)),2);
end

figure(digi_fig)

for test_dgi = 1:8    
    subplot(frows,8,16+test_dgi) %first digit
    colormap(hot)
    caxis([0 cxmax])
    set(gca,'Color','k')
    mxesta_test_dig = mxest_all_a(:,test_dgi);
    digi_display(mxesta_test_dig);
    axis off
end
subplot(frows,8,17)
title('trained rand rec + ad','Color','w')

%% Balanced recurrent connections with adaptation
O = zeros(Nn,length(time));
r = zeros(Nn,length(time));
ra = zeros(Nn,length(time));
V = zeros(Nn,length(time));
sest = zeros(Nj,length(time));

Input = Gain*W'*(s + tau.*ds); % weighted input

for t = 2:length(time)
    dVdt = -V(:,t-1) + Input(:,t-1) - tau.*Gain*W_rec*O(:,t-1) - tau.*mu.*Gain*O(:,t-1) + ((tau/tau_a)-1)*mu.*Gain*ra(:,t-1); %this is dV/dt
    V(:,t) = V(:,t-1) + dt.*(dVdt./tau);
    
    adThr = thresh.*ones(Nn,1);
    O(:,t) = (V(:,t)>=adThr)./dt; %spikes are placed on neurons that have crossed threshold
    
    if sum(O(:,t))>(1/dt)
        [~,vi] = max(V(:,t) - adThr);
        O(:,t) = 0;
        O(vi,t) = 1/dt; %to ensure only one spike per time step
    end
    
    dr = -(1/tau)*r(:,t-1) + O(:,t-1);
    r(:,t) = r(:,t-1) + dt.*dr;
        
    dra = -(1/tau_a)*ra(:,t-1) + O(:,t-1);
    ra(:,t) = ra(:,t-1) + dt.*dra;
    
    dsest = -(1/tau)*sest(:,t-1) + W*O(:,t-1);
    sest(:,t) = sest(:,t-1) + dt.*dsest;
end

% figure('WindowStyle','docked')
% subplot(2,1,1)
% Otex = (((1:Nn)'*ones(1,length(time))).*O.*dt)';
% Otex(Otex==0)=nan;
% plot(time,Otex,'.')
% subplot(2,1,2)
% hold on
% plot(time,sest)
% plot(time,s,'--')

%%
for ind = 1:8
    mxest_all_ab(:,ind) = mean(sest(:,stimONbl+(ind-1)*(Nstime-1)),2);
end

figure(digi_fig)

for test_dgi = 1:8    
    subplot(frows,8,24+test_dgi) %first digit
    colormap(hot)
    caxis([0 cxmax])
    set(gca,'Color','k')
    mxestab_test_dig = mxest_all_ab(:,test_dgi);
    digi_display(mxestab_test_dig);
    axis off
end
subplot(frows,8,25)
title('balanced+ad','Color','w')

%%
figure(digi_fig)
linkaxes
ylim([-15 3])
xlim([-1 6])
set(gcf,'Color','k')

