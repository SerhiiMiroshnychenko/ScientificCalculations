x = (0:0.2:10);
y1 = sigmf(x, [3 5]);
y3 = sigmf(x, [1 5]);
y2 = sigmf(x, [2 5]);
plot(x,y1,x,y2,x,y3)
grid;
ylabel('y');
xlabel('x')