close all
clear all
clc

load('robust_acc_tradeoff.mat')

figure
boxplot([acc5, acc10, acc20], [5, 10, 20])


figure('Position',[100,100,300,300])
semilogx(mean(Lipnom),mean(accnom),'o','MarkerSize',8,'LineWidth',1.5)
hold on
semilogx(mean(LipL2_01),mean(accL2_01),'o','MarkerSize',8,'LineWidth',1.5)
semilogx(mean(LipL2_05),mean(accL2_05),'o','MarkerSize',8,'LineWidth',1.5)
semilogx(mean(LipL2_1),mean(accL2_1),'o','MarkerSize',8,'LineWidth',1.5)
semilogx(mean(Lip1),mean(acc1),'x','MarkerSize',8,'LineWidth',1.5)
semilogx(mean(Lip2),mean(acc2),'x','MarkerSize',8,'LineWidth',1.5)
semilogx(mean(Lip5),mean(acc5),'x','MarkerSize',8,'LineWidth',1.5)
semilogx(mean(Lip10),mean(acc10),'x','MarkerSize',8,'LineWidth',1.5)
semilogx(mean(Lip20),mean(acc20),'x','MarkerSize',8,'LineWidth',1.5)
semilogx(mean(Lip50),mean(acc50),'x','MarkerSize',8,'LineWidth',1.5)
semilogx(mean(Lip100),mean(acc100),'x','MarkerSize',8,'LineWidth',1.5)

line([1 1], [0.6 1],'LineStyle','--')
line([2 2], [0.6 1],'LineStyle','--')
line([5 5], [0.6 1],'LineStyle','--')
line([10 10], [0.6 1],'LineStyle','--')
line([20 20], [0.6 1],'LineStyle','--')
line([50 50], [0.6 1],'LineStyle','--')
line([100 100], [0.6 1],'LineStyle','--')
xlim([0.5,200])
ylim([0.6,1])
legend("vanilla",  "L2, $\gamma = 0.01$", "L2, $\gamma = 0.05$", "L2, $\gamma = 0.1$",  "Lip1", "Lip2", "Lip5", "Lip10","Lip20", "Lip50","Lip100","interpreter","latex", "location", "southeast")

matlab2tikz('robustness_accuracy_tradeoff.tex')

