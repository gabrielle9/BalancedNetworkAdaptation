Ww = -10:0.1:10;
mu = 1; %0.2;

gain = 1./(Ww.*Ww + mu);
gainw = Ww./(Ww.*Ww + mu);

figure
plot(Ww,gain,'LineWidth',4)
hold on
plot(Ww,gainw,'LineWidth',4)
legend('Gain','Feedforward gain')
set(gca,'FontSize',24)
xlabel('Decoding weight')
grid on
