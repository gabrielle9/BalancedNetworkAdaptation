% Simulation of spiking model autoencoder in Sophie's latest draft of paper
% Based on derivation from 26 June, 2017 in labnotebook
% Updated from derivation on 6 July 2017 - uses cost with ra instead of p
% Updated from 6 April 2018: using the std for the shaded error plot and
% eliminating hitch from error plot.

clearvars
addpath(genpath('/Users/gabriellegutierrez/Documents/MATLAB/kakearney-boundedline-pkg-8179f9a'))

%% Time Structure

dt = 0.01; %units of ms
time = 0:dt:3e3;

%% Parameters

Nj = 1; %number of dimensions in input
Nn = 10; %number of neurons
% lamd = 0.2; %decoder lambda
tau = 5; %5; %10; %100*dt; %shorter tau gives estimate more variance, recruits more neurons
mu = 0.2;%0.002; %20; %0.2;
tau_a = 1000; %1000; %200; %1000*dt;

sgain = 10; %7.5; %1.5; %50;
stime = round(0.25e3/dt):round(2.75e3/dt);

%% Connectivity Structure
% W = 4+0.1*(1:Nn);
W = 1:Nn;
% W = 5.*ones(1,Nn) + 0.01.*rand(1,Nn);
% W = 0.5:0.5:10;
% Nn = length(W);
% W = [1.01 1];
% W = 0.1.*(1:Nn);
% W = 0.1:0.1:(Nn*0.1);
Gain = diag(2./(W.*W + mu));
thresh = 1;

%% Noise and Inputs
% vnoise = sigv.*randn(Nn,length(time)); %voltage noise
% th_noise = sigth.*randn(Nn,length(time)); %threshold noise
% snoise = sigs.*randn(Nj,length(time)); %signal noise

s = zeros(Nj,length(time)); %command input.
s(:,stime) = sgain; %7.5; %1.5; %50;

ds = [0 diff(s)]./dt;
% ds = [zeros(Nj,1) diff(s,1,2)]; %if s has extra dimensions

Input = Gain*W'*(s + tau.*ds);

%% Initialization
O = zeros(Nn,length(time));
r = zeros(Nn,length(time));
ra = zeros(Nn,length(time));
V = zeros(Nn,length(time));
sest = zeros(Nj,length(time));

%% Integration
for t = 2:length(time)
    dVdt = -V(:,t-1) + Input(:,t-1) - tau.*Gain*W'*W*O(:,t-1) - tau.*mu.*Gain*O(:,t-1) + ((tau/tau_a)-1).*mu.*Gain*ra(:,t-1);
%     dVdt = -V(:,t-1) + Input(:,t-1) - tau.*Gain*diag(diag(W'*W))*O(:,t-1) - tau.*mu.*Gain*O(:,t-1) + ((tau/tau_a)-1).*mu.*Gain*ra(:,t-1); %no recurr
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

s_hat = W*r;

%% Plot raster and estimate with target
cmap = cmapMaker(Nn);
rws = 2;
cols = 2;

Tsteps = length(time);
Otex = (((1:Nn)'*ones(1,Tsteps)).*O.*dt)';
Otex(Otex==0)=nan;

smsest = smooth(sest(stime)',1e4);
% mvsest = movvar(sest(stime),1e4);
mvsest = movstd(sest(stime),1e4);

% err = (s-sest).*(s-sest);
err = (s(1:stime(end))-sest(1:stime(end))).*(s(1:stime(end))-sest(1:stime(end)));
smerr = smooth(err',1e4);
cost = mu*sum(ra.*ra);

time1 = time./1000;

%%
figure
% subplot(rws,cols,1)
hold on
for k = 1:Nn
    ok = find(O(k,:));
    for m = 1:length(ok)
%         line([time(ok(m)),time(ok(m))],[Otex(ok(m),k),Otex(ok(m),k)+0.75]-0.4,'Color',cmap(abs(W(k)),:),'LineWidth',1.25)
        line([time1(ok(m)),time1(ok(m))],[Otex(ok(m),k),Otex(ok(m),k)+0.75]-0.4,'Color',cmap(k,:),'LineWidth',1.25)
    end
end
set(gca,'FontSize',24)
xlim([0 time1(end)])
ylim([0 Nn+1])
ylabel('Neuron #')
title(['mu=' num2str(mu) ', tau=' num2str(tau) ', tau_a=' num2str(tau_a) ', s=' num2str(sgain) ', W=[1:Nn]'],'FontSize',16)
xlabel('time (s)')
set(gcf,'Position',[500 320 500 350])

figure
% subplot(rws,cols,2)
%plot error and cost
yyaxis left
% plot(time1(stime),smerr(stime),'LineWidth',3)
plot(time1(stime(1):stime(end-500)),smerr(stime(1):stime(end-500)),'LineWidth',3)
ylabel('error')
yyaxis right
plot(time1(stime),cost(stime),'LineWidth',3)
ylabel('cost')
set(gca,'FontSize',24)
xlabel('time (s)')
set(gcf,'Position',[500 320 500 350])

figure
% subplot(rws,cols,3); hold on
hold on
plot(time1,sest','Color',[1 0.25 0])
plot(time1,s','Color',[0.35 0.35 0.5],'LineWidth',4)
set(gca,'FontSize',24)
% legend('x','x_{est}')
axis tight
xlabel('time (s)')
set(gcf,'Position',[500 320 500 350])

figure
% subplot(rws,cols,4); hold on
hold on
[hl,hp] = boundedline(time1(stime),smsest,mvsest);
plot(time1,s','Color',[0.35 0.35 0.5],'LineWidth',4)
set(gca,'FontSize',24)
% legend('x','x_{est}')
axis tight
% ylim([0 30])
xlabel('time (s)')
set(gcf,'Position',[500 320 500 350])


% set(gcf,'Position',[80 135 960 570])
% set(gcf,'Position',[360 110 560 590])

% %% Homogeneous network
% W = 5.*ones(1,Nn) + 0.01.*rand(1,Nn);
% Gain = diag(2./(W.*W + mu));
% Input = Gain*W'*(s + tau.*ds);
% 
% %% Initialization
% O = zeros(Nn,length(time));
% r = zeros(Nn,length(time));
% ra = zeros(Nn,length(time));
% V = zeros(Nn,length(time));
% sest = zeros(Nj,length(time));
% 
% %% Integration
% for t = 2:length(time)
%     dVdt = -V(:,t-1) + Input(:,t-1) - tau.*Gain*W'*W*O(:,t-1) - tau.*mu.*Gain*O(:,t-1) + ((tau/tau_a)-1).*mu.*Gain*ra(:,t-1);
% %     dVdt = -V(:,t-1) + Input(:,t-1) - tau.*Gain*diag(diag(W'*W))*O(:,t-1) - tau.*mu.*Gain*O(:,t-1) + ((tau/tau_a)-1).*mu.*Gain*ra(:,t-1); %no recurr
%     V(:,t) = V(:,t-1) + dt.*(dVdt./tau);
%     
%     adThr = thresh.*ones(Nn,1);
%     O(:,t) = (V(:,t)>=adThr)./dt; %spikes are placed on neurons that have crossed threshold
% 
%     if sum(O(:,t))>(1/dt)
%         [~,vi] = max(V(:,t) - adThr);
%         O(:,t) = 0;
%         O(vi,t) = 1/dt; %to ensure only one spike per time step
%     end
%     
%     dr = -(1/tau)*r(:,t-1) + O(:,t-1);
%     r(:,t) = r(:,t-1) + dt.*dr;
%     
%     dra = -(1/tau_a)*ra(:,t-1) + O(:,t-1);
%     ra(:,t) = ra(:,t-1) + dt.*dra;
% 
%     dsest = -(1/tau)*sest(:,t-1) + W*O(:,t-1);
%     sest(:,t) = sest(:,t-1) + dt.*dsest;
%     
% end
% 
% s_hat = W*r;
% 
% %% Plot raster and estimate with target
% cmap = cmapMaker(Nn);
% 
% Tsteps = length(time);
% Otex = (((1:Nn)'*ones(1,Tsteps)).*O.*dt)';
% Otex(Otex==0)=nan;
% 
% figure
% subplot(2,1,1)
% hold on
% for k = 1:Nn
%     ok = find(O(k,:));
%     for m = 1:length(ok)
% %         line([time(ok(m)),time(ok(m))],[Otex(ok(m),k),Otex(ok(m),k)+0.75]-0.4,'Color',cmap(abs(W(k)),:),'LineWidth',1.25)
%         line([time(ok(m)),time(ok(m))],[Otex(ok(m),k),Otex(ok(m),k)+0.75]-0.4,'Color',cmap(5,:),'LineWidth',1.25)
%     end
% end
% 
% set(gca,'FontSize',24)
% xlim([0 time(end)])
% ylim([0 Nn+1])
% ylabel('Neuron #')
% title(['mu=' num2str(mu) ', tau=' num2str(tau) ', tau_a=' num2str(tau_a) ', s=' num2str(sgain) ', W=[5+jitter]'],'FontSize',16)
% 
% subplot(2,1,2)
% hold on
% time1 = time./1000;
% % plot(time,s','Color',[0.35 0.35 0.5],'LineWidth',4)
% plot(time1,sest','Color',[1 0.25 0])
% plot(time1,s','Color',[0.35 0.35 0.5],'LineWidth',4)
% set(gca,'FontSize',24)
% xlabel('time (s)')
% % legend('x','x_{est}')
% axis tight
% % ylim([0 30])
% set(gcf,'Position',[360 110 560 590])


% %% Plot raster and estimate with target
% 
% figure('WindowStyle','docked')
% subplot(1,2,1)
% plot(time,r); colormap(cmap)
% subplot(1,2,2)
% plot(time,ra)

% %% Plot ISI
% 
% hf1 = figure; hold on
% for j = 1:8:Nn;
%     o_ind6 = find(O(j,:));
%     spike_times6 = time(o_ind6);
%     isi6 = diff(spike_times6);
%     isi6inst = nan(size(time));
%     for i = 1:length(o_ind6)-1
%         isi6inst(o_ind6(i):o_ind6(i+1)-1) = isi6(i);
%     end
%     
%     figure(hf1)
%     plot(time,1./isi6inst,'Color',cmap(j,:),'LineWidth',2)
% end


% %% Plot 2 neuron figs: 3 panel
% figure;%('WindowStyle','docked')
% cost = mu*sum(ra.*ra);
% xl1 = 250;
% xl2 = 500;
% 
% subplot(3,1,1)
% hold on
% for k = 1:Nn
%     ok = find(O(k,:));
%     for m = 1:length(ok)
%         line([time(ok(m)),time(ok(m))],[Otex(ok(m),k),Otex(ok(m),k)+0.75]- 0.4,'LineWidth',2,'Color',cmap(k,:));%cmap(abs(W(k)),:))
%     end
% end
% xlim([xl1 xl2])
% ylim([0 3])
% set(gca,'FontSize',24)
% legend('neuron 1','neuron 2')
% 
% subplot(3,1,2)
% hold on
% for k = 1:Nn
%     plot(time,r(k,:),'LineWidth',3,'Color',cmap(k,:)); %cmap(abs(W(k)),:))
% end
% xlim([xl1 xl2])
% ylim([0 15])
% set(gca,'FontSize',24)
% ylabel('r(t)')
% legend('r_1','r_2')
% 
% subplot(3,1,3)
% hold on
% plot(time,cost,'Color',[0.95 0.75 0.1],'LineWidth',3)
% plot(time,s','Color',[0.35 0.35 0.5],'LineWidth',4)
% plot(time,sest','Color',[1 0.25 0],'LineWidth',2)
% xlim([xl1 xl2])
% set(gca,'FontSize',24)
% xlabel('time (ms)')
% ylim([0 20])
% legend('Cost','Stimulus','Estimate')
% 
% set(gcf,'Position',[60 175 474 540])


% figure
% subplot(2,1,1)
% hold on
% for k = 1:Nn
%     ok = find(O(k,:));
%     for m = 1:length(ok)
%         line([time(ok(m)),time(ok(m))],[Otex(ok(m),k),Otex(ok(m),k)+0.75]-0.4,'Color',cmap(abs(W(k)),:),'LineWidth',1.25)
%     end
% end
% 
% set(gca,'FontSize',24)
% xlim([0 time(end)])
% ylim([0 Nn+1])
% ylabel('Neuron #')
% 
% subplot(2,1,2)
% hold on
% plot(time,s','Color',[0.35 0.35 0.5],'LineWidth',4)
% plot(time,sest','Color',[1 0.25 0])
% plot(time,s','Color',[0.35 0.35 0.5],'LineWidth',2)
% set(gca,'FontSize',24)
% xlabel('t (ms)')
% % legend('x','x_{est}')
% axis tight
% ylim([0 30])
% set(gcf,'Position',[360 110 560 590])


