%x = (0:0.2:10);

%plot(x,A,x,B)
N = 501;
minX = -20;
maxX = 20;
x = linspace(minX,maxX,N);

A = gaussmf(x, [1 5]);
B = gaussmf(x, [1 7]);
