% by Dave Touretzky (modified by Nikolay Nikolaev)
% https://www.cs.cmu.edu/afs/cs/academic/class/15782-f06/matlab/
load sunspot.dat
year=sunspot(:,1); relNums=sunspot(:,2); %plot(year,relNums)
ynrmv=mean(relNums(:)); sigy=std(relNums(:)); 
nrmY=relNums; %nrmY=(relNums(:)-ynrmv)./sigy; 
ymin=min(nrmY(:)); ymax=max(nrmY(:)); 
relNums=2.0*((nrmY-ymin)/(ymax-ymin)-0.5);
% create a matrix of lagged values for a time series vector
Ss=relNums';
idim=10; % input dimension
odim=length(Ss)-idim; % output dimension
for i=1:odim
   y(i)=Ss(i+idim);
   for j=1:idim
       x(i,j) = Ss(i-j+idim); %x(i,idim-j+1) = Ss(i-j+idim);
   end
end
Patterns = x'; Desired = y; NHIDDENS = 5; prnout=Desired;
[NINPUTS,NPATS] = size(Patterns); [NOUTPUTS,NP] = size(Desired);
%apply the backprop here...
LearnRate = 0.001; Momentum = 0; DerivIncr = 0; deltaW1 = 0; deltaW2 = 0;
Inputs1= [Patterns;ones(1,NPATS)]; %Inputs1 = [ones(1,NPATS); Patterns];
Weights1 = 0.5*(rand(NHIDDENS,1+NINPUTS)-0.5);
Weights2 = 0.5*(rand(1,1+NHIDDENS)-0.5); 
TSS_Limit = 0.02
for epoch = 1:200
  % Forward propagation
  NetIn1 = Weights1 * Inputs1;
  %Hidden = 1.0 ./( 1.0 + exp( -NetIn1 ));
  Hidden=1-2./(exp(2*NetIn1)+1); %Hidden = tanh( NetIn1 );
  Inputs2 = [Hidden; ones(1,NPATS)];
  NetIn2 = Weights2 * Inputs2;
  %Out = 1.0 ./ ( 1.0 + exp( -NetIn2 ));  
  %Out=1-2./(exp(2*NetIn2)+1); %Out = tanh(NetIn2);
  Out = NetIn2;  prnout=Out;
  % Backward propagation of errors
  Error = Desired - Out;
  TSS = sum(sum( Error.^2 )); % sum(sum(E.*E)); 
  %Beta = Out .* (1.0 - Out) .* Error;
  %Beta = ( 1.0 - Out.^ 2) .* Error;
  Beta = Error;
  bperr = ( Weights2' * Beta );
  %HiddenBeta = Hidden .* (1.0 - Hidden) .* bperr(1:end-1,:);
  HiddenBeta = (1.0 - Hidden .^2 ) .* bperr(1:end-1,:);
  % Calculate the weight updates:
  dW2 = Beta * Inputs2';
  dW1 = HiddenBeta * Inputs1';
  deltaW2 = LearnRate * dW2 + Momentum * deltaW2;
  deltaW1 = LearnRate * dW1 + Momentum * deltaW1;
  % Update the weights:
  Weights2 = Weights2 + deltaW2;
  Weights1 = Weights1 + deltaW1;
  fprintf('Epoch %3d:  Error = %f\n',epoch,TSS);
  if TSS < TSS_Limit, break, end
end
plot(year(11:288),Desired,year(11:288),prnout)
title('Sunspot Data')