% Simulation of spiking model autoencoder in Sophie's latest draft of paper
% Based on derivation from 26 June, 2017 in labnotebook
% Updated from derivation on 6 July 2017 - uses cost with ra instead of p
% Updated from 25 July 2017 to decode angle
% Updated to include linear cost to avoid ping pong which was activating
% opposite preference neurons. Also chose different parameters with Sophie
% so that we could more clearly see the change in accuracy variance.

clearvars
addpath(genpath('/Users/gabriellegutierrez/Documents/MATLAB/kakearney-boundedline-pkg-8179f9a'))
savestr = 'OM_pulse_resp_31July18';

%% Time Structure

dt = 0.01; %units of ms
time = 0:dt:3.5e3;

%% Parameters

Nj = 2; %number of dimensions in input
Nn = 200; %number of neurons
tau = 5; %5; %10; %100*dt; %shorter tau gives estimate more variance, recruits more neurons
tau_a = 2000; %1000; %200; %1000*dt;
mu = 0.1; %0.2;%0.002; %20; %0.2;

sgain = 50; %to get same results from transformation network with sgain=10 and tau=5
stime = round(0.5e3/dt):round(3.5e3/dt);

%% Connectivity Structure
Nw = Nn/2;
theta = 0:2*pi/Nw:2*pi-2*pi/Nw; % one period
theta = reshape(ones(2,1)*theta,1,[]); %make it Nn long again

bgain1 = 3;  %3;%1;%3;
bgain2 = 9; %15; %9;%6;%15;
bgain = reshape([bgain1;bgain2]*ones(1,Nw),1,[]); %interspersed gains

W = [cos(theta).*bgain; sin(theta).*bgain];

Gain = diag(diag(2./(W'*W + mu)));

thresh = 1;
eta = 10; %linear cost factor

%% Noise and Inputs
Np = 18; %increase this to increase the resolution of the transfer functions
thetp = 0:2*pi/Np:2*pi-2*pi/Np;

s = zeros(Nj,length(time)); %command input.
test_thet = thetp(9);
s(1,stime) = cos(test_thet)*sgain;
s(2,stime) = sin(test_thet)*sgain;
    
% ds = [0 diff(s)]./dt;
ds = [zeros(Nj,1) diff(s,1,2)]; %if s has extra dimensions

Input = Gain*W'*(s + tau.*ds);

save(savestr)

%% Initialization
O = zeros(Nn,length(time));
r = zeros(Nn,length(time));
ra = zeros(Nn,length(time));
V = zeros(Nn,length(time));
sest = zeros(Nj,length(time));

%% Integration
for t = 2:length(time)
%     dVdt = -V(:,t-1) + Input(:,t-1) - tau.*Gain*W'*W*O(:,t-1) - tau.*mu.*Gain*O(:,t-1) + ((tau/tau_a)-1).*mu.*Gain*ra(:,t-1);
    dVdt = -V(:,t-1) + Input(:,t-1) - tau.*Gain*W'*W*O(:,t-1) - tau.*mu.*Gain*O(:,t-1) + ((tau/tau_a)-1).*mu.*Gain*ra(:,t-1);
%     dVdt = -V(:,t-1) + Input(:,t-1) - tau.*Gain*diag(diag(W'*W))*O(:,t-1) - tau.*mu.*Gain*O(:,t-1) + ((tau/tau_a)-1).*mu.*Gain*ra(:,t-1); %no recurr
    V(:,t) = V(:,t-1) + dt.*(dVdt./tau);
    
    adThr = thresh.*ones(Nn,1) + eta*diag(Gain);
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

s_hat = W*r;

%% save stuff
save(savestr,'O','sest')

%% Plot raster and estimate with target
% cmap = cmapMaker(Nn);
bimap = cmapMaker(2);
cmap = repmat(bimap,(Nn/2),1);
time1 = time./1000;

% Tsteps = length(time);
Tsteps = length(time1);
Otex = (((1:Nn)'*ones(1,Tsteps)).*O.*dt)';
Otex(Otex==0)=nan;

% decx = DecodeTheta(nanmean(sest(:,stime),2));
% decx = DecodeTheta(sest(:,stime));
decx = DecodeTheta(sest);
decs = DecodeTheta(s);
goodtime = stime(100):stime(end);
% goodtime = stime(100):200001;
smdecx = smooth(decx(goodtime)',1e4);
% mvstdec = movstd((decx(goodtime)-decs(goodtime)),1e4);
mvstdec = movstd(decx(goodtime),1e4);

figure
subplot(2,1,1)
hold on
plot(time1,Otex(:,1:2:Nn),'.','Color',cmap(1,:),'MarkerSize',12)
hold on
plot(time1,Otex(:,2:2:Nn),'.','Color',cmap(2,:),'MarkerSize',12)
% for k = 1:Nn
%     ok = find(O(k,:));
%     for m = 1:length(ok)
% %         line([time(ok(m)),time(ok(m))],[Otex(ok(m),k),Otex(ok(m),k)+0.75]-0.4,'Color',cmap(k,:),'LineWidth',1.25)
%         line([time1(ok(m)),time1(ok(m))],[Otex(ok(m),k),Otex(ok(m),k)+0.75]-0.4,'Color',cmap(k,:),'LineWidth',1.25)
%     end
% end
set(gca,'FontSize',24)
% xlim([0 time(end)])
% xlim([0 time1(end)])
% xlim([time1(goodtime(1)) time1(goodtime(end))])
xlim([0.5 2])
ylim([0 Nn+1])
ylabel('Neuron #')
title(['mu=' num2str(mu) ', tau=' num2str(tau) ', tau_a=' num2str(tau_a) ', s=' num2str(sgain) ', W=[1:Nn]'],'FontSize',16)

%%
subplot(2,1,2)
hold on
plot(time1,decs,'Color',[0.35 0.35 0.5],'LineWidth',3')
% plot(time,s','Color',[0.35 0.35 0.5],'LineWidth',4)
% plot(time1,sest','Color',[1 0.25 0])
% plot(time1,decs,'Color',[0.35 0.35 0.5],'LineWidth',6)
% plot(time1,decx,'Color',[1 0.25 0],'LineWidth',1)
% [hl,hp] = boundedline(time1,decx,mvstdec);
% [hl,hp] = boundedline(time1(goodtime),decx(goodtime),mvstdec);
[hl,hp] = boundedline(time1(goodtime),smdecx,mvstdec);
% plot(time1,s','Color',[0.35 0.35 0.5],'LineWidth',4)
set(gca,'FontSize',24)
xlabel('time (s)')
% legend('x','x_{est}')
axis tight
ylim([2.5 3])
xlim([0.5 2])
% ylim([0 30])
title('smoothed decoded angle and moving std')
set(gcf,'Position',[360 110 560 590])


