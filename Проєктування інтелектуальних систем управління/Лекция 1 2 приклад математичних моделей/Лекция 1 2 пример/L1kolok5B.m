x = (0:0.2:10);
y1 = gbellmf(x, [1 1 5]);
y2 = gbellmf(x, [1 2 5]);
y3 = gbellmf(x, [1 3 5]);
plot (x, y1, x, y2, x, y3);
grid
ylabel('y')
xlabel('x')
