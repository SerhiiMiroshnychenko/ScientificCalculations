mf = [fismf(@gaussmf,[1.5 5]) fismf(@trapmf,[3 4 6 7])];
      x = (-2:0.1:12)';
      y = evalmmf(mf,x);
      plot(x,y); grid;
      xlabel('Universe of discourse (x)'),ylabel('Membership value (y)')
      legend('gaussmf, P=[1.5 5]','trapmf, P=[3 4 6 7]')