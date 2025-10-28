N = 501;
minX = -5;
maxX = 20;
x = linspace(minX,maxX,N);
mf1 = gaussmf(x,[1 5]); trimf
mf2 = gaussmf(x,[1 7]);
% обчислення max та побудова графіку результата обчислення
mf = max(mf1,mf2);
figure(1)
plot(x,mf,'LineWidth',3); grid;
% обчислення min та побудова графіку результата обчислення
figure(2)
mf3 = min(mf1,mf2);
plot(x,mf3,'LineWidth',3); grid;
% обчислення об'еднання та побудова графіку результата обчислення
figure(3)
mf4 = probor([mf1;mf2])% об'еднання
plot(x,mf4,'LineWidth',3); grid;
% обчислення перетин та побудова графіку результата обчислення
figure(4)
mf4 = prod([mf1;mf2])
plot(x,mf4,'LineWidth',3); grid; % перетин

