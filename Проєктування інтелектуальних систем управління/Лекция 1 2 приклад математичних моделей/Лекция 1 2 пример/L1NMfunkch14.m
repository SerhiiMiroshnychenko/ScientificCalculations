N = 501;
minX = -20;
maxX = 20;
x = linspace(minX,maxX,N);
A = trapmf(x,[-10 -2 1 3]);
B = gaussmf(x,[2 5]);
%Evaluate the sum, difference, product, and quotient of A and B.

Csum = fuzarith(x,A,B,'sum');
Csub = fuzarith(x,A,B,'sub');
Cprod = fuzarith(x,A,B,'prod');
Cdiv = fuzarith(x,A,B,'div');

%Plot the addition and subtraction results.

figure(5)
subplot(2,1,1)
plot(x,A,'--',x,B,':',x,Csum,'c')
title('Fuzzy Addition, A+B')
legend('A','B','A+B')
subplot(2,1,2)
plot(x,A,'--',x,B,':',x,Csub,'c')
title('Fuzzy Subtraction, A-B')
legend('A','B','A-B')
figure(6)
subplot(2,1,1)
plot(x,A,'--',x,B,':',x,Cprod,'c')
title('Fuzzy Multiplication, A*B')
legend('A','B','A*B')
subplot(2,1,2)
plot(x,A,'--',x,B,':',x,Cdiv,'c')
title('Fuzzy Division, A/B')
legend('A','B','A/B')