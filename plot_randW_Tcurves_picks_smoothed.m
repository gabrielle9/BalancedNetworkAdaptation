clearvars
% cd('/Users/gabriellegutierrez/Documents/MATLAB/DeneveAdaptation/2017/August/4')
% load('OM_Tcurves_randW_4Aug17.mat')
cd('/Users/gabriellegutierrez/Documents/MATLAB/DeneveAdaptation/2018/April/12')
load('OM_Tcurves_randW_lin_cost_12Apr18.mat')
cd('/Users/gabriellegutierrez/Documents/MATLAB/DeneveAdaptation/2019/July/30')

%%
picks = [142,21,70,64,119,182,87,55];
smp = 5;
lw = 3;

figure;
for ind = 1:length(picks)
    subplot(2,4,ind); hold on
    plot(thetp,smooth(sOp(picks(ind),:),smp),'LineWidth',lw)
    plot(thetp,smooth(sOpa1(picks(ind),:),smp),'LineWidth',lw) %weak adaptation
    plot(thetp,smooth(sOpa2(picks(ind),:),smp),'LineWidth',lw) %strong adaptation
    line([adthet adthet],[0 max([sOp(picks(ind),:),sOpa1(picks(ind),:),sOpa2(picks(ind),:)])+2],'LineStyle','--','Color','k','LineWidth',2)
%     title(['gain=' num2str(Gain(picks(ind),picks(ind))) ', Nn#=' num2str(picks(ind))])
    title(['neuron ' num2str(ind)])
    set(gca,'FontSize',14)
    axis tight
    xlim([0 thetp(end)])
    set(gca,'YTick',[0:10:10])
%     set(gca,'XTick',[])
end
xlabel('\theta')
set(gcf,'Position',[190   278   1040   420])
legend('before adaptation','weak adaptation','strong adaptation')
% linkaxes
% ylim([0 15])
