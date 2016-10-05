% just an example
vExample = [1*ones(1,5) 32*ones(1,12) 42*ones(1,8) 1*ones(1,8)];
MLattice = zeros(42,length(vExample));
for k=1:length(vExample)
  MLattice(vExample(k),k) = 1;
end
imagesc(MLattice)