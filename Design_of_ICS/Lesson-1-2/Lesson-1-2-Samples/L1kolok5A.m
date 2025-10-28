x = (0:0.2:10);
y1 = gbellmf(x, [1 1 5]);
y2 = gbellmf(x, [2 1 5]);
y3 = gbellmf(x, [3 1 5]);
plot(x,y1,x,y2,x,y3);
grid;
ylabel('y');
xlabel('x')