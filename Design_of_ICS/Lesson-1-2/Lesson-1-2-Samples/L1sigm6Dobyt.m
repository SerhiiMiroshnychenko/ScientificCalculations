x = (0:0.2:10);
params1 = [2 3];
params2 = [-5 8];
y = psigmf(x, [params1 params2]);
plot(x,y);
grid;
ylabel('y');
xlabel('x')