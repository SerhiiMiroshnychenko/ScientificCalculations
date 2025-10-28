x = (0:0.2:10);
y1 = sigmf(x, [5 2]);
y2 = sigmf(x, [5 7]);
y3 = y1-y2;
plot(x,y3);
grid;
ylabel('y');
xlabel('x')
