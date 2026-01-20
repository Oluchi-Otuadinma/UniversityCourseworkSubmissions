clear all
close all
load sunspot.dat
year = sunspot(:,1); spots = sunspot(:,2);
mean_spots = mean(spots); std_spots = std(spots);
warning('off', 'MATLAB:nearlySingularMatrix')

normalised = spots; 
norm_Min = min(normalised(:));
norm_Max = max(normalised(:));

spots =  2 * ((normalised - norm_Min) / (norm_Max - norm_Min) - 0.5);

% create a matrix of lagged values for a time series vector
Ss = spots';
NINPUTS = 10;
idim = NINPUTS; % input dimension
odim = length(Ss) - idim; % output dimension
for i = 1:odim
   y(i) = Ss(i+idim);
   for j = 1:idim
       x(i,j) = Ss(i-j+idim); %x(i,idim-j+1) = Ss(i-j+idim);
   end
end

%Patterns 10 278
%Data_in 10 278
%Data_out 1 278
%NINPUTS & NHIDDENS = nodes

max_epochs = 50;
Data_points = x'; Desired = y; NHIDDENS = 5; [NOUTPUTS, Data_out] = size(Desired);
Data_in = size(x,1);
Lam = 0.1; max_iter=20;
LearnRate = 0.001; deltaFirst = 0; deltaSecond = 0;
Inputs1 = [Data_points;ones(1,Data_in)];
First_layB = 0.5*(rand(NHIDDENS,1+NINPUTS)-0.5);
Second_layB = 0.5*(rand(1,1+NHIDDENS)-0.5); 
In_outB = 0.5*(rand(NOUTPUTS, NINPUTS)-0.5);
TSS_Limit = 0.02;
Search_max = 10;
First_lay = First_layB;
Second_lay = Second_layB;
In_out = In_outB;
%MSE = TSS / Data_in;
    H =  zeros(size(First_lay,1) * size(First_lay,2) + size(Second_lay,1) * size(Second_lay,2) + size(In_out,1) * size(In_out,2));

for epoch = 1:max_epochs

  % Forward propagation
  NetIn1 = First_lay * Inputs1;

  Hidden = 1-2./(exp(2*NetIn1)+1); 
  Inputs2 = [Hidden; ones(1, Data_in)];
  Out = Second_lay  * Inputs2;
 
  Out = Out + In_out * Inputs1(1:end-1,:);

  
  % Backward propagation of errors
  Error = Desired - Out;
  TSS = sum(sum(Error.^2 ));

  Error = ones(size(Error));

  bperr = ( Second_lay' * Error );

  HiddenBeta = (1.0 - Hidden .^2 ) .* bperr(1:end-1,:);

  H = zeros(size(H));

  for example=1:size(Inputs1, 2)
      temp_input1 = Inputs1(:, example);
      temp_input2 = Inputs2(:, example);
      temp_error = Error(:, example);
      temp_hiddenbeta = HiddenBeta(:, example);

      dfirst = temp_hiddenbeta * temp_input1';
      dsecond = temp_error * temp_input2;
      dthird = temp_error * temp_input1(1:end-1,:);

      dfirst = dfirst(:);
      dsecond = dsecond(:);
      dthird = dthird(:);
      H = H + ([dfirst;dsecond;dthird]*[dfirst;dsecond;dthird]');
    
  end
  H = H./example;

  for s=1:max_iter
    H_inverse = inv(H + eye(size(H))*Lam);
    H_lr = diag(H_inverse);
    Error = Desired - Out;
    bperr = ( Second_lay' * Error );
    HiddenBeta = (1.0 - Hidden .^2 ) .* bperr(1:end-1,:);
     % Calculate the weight updates:
     %Gradients
      DFirst = HiddenBeta * Inputs1';
      DSecond = Error * Inputs2';
      DThird = Error * Inputs1(1:end-1,:)';
      
      copy1 = First_lay(:);
      copy2 = Second_lay(:);
      copy3 = In_out(:);
      
      copy1 = copy1 + DFirst(:).*H_lr(1:length(copy1));
      copy2 = copy2 + DSecond(:).*H_lr(length(copy1)+1:length(copy1)+length(copy2));
      len2 = length(copy1)+length(copy2)+1;
      copy3 = copy3 + DThird(:).*H_lr(len2:len2+length(copy3)-1);
      
      copy1 = reshape(copy1, size(First_lay));
      copy2 = reshape(copy2, size(Second_lay));
      copy3 = reshape(copy3, size(In_out));
    
      %Forward Prop
      NetIn1 = copy1 * Inputs1;

      Hidden = 1-2./(exp(2*NetIn1)+1); 
      Inputs2 = [Hidden; ones(1, Data_in)];
      Out = copy2  * Inputs2;
      Out = Out + copy3 * Inputs1(1:end-1,:); 

      Error = Desired - Out;
      MSE_comp = sum(sum(Error.^2 ));
        
if MSE_comp < TSS
    
    Lam = Lam/10;
    First_lay = copy1;
    Second_lay = copy2;
    In_out = copy3;

else
    Lam = Lam*10;
    
end
         

  end
  
  % Update the weights:
  First_lay = First_lay + deltaFirst;
  Second_lay = Second_lay + deltaSecond;

  
  MSE = TSS / Data_in;
  MSEPlot(epoch)=MSE;

  fprintf('Epoch %3d:  Error = %f\n',epoch,TSS);
  if TSS < TSS_Limit, break, end
end

First_lay = First_layB;
Second_lay = Second_layB;
In_out = In_outB;

for epoch = 1:max_epochs

  % Forward propagation
  NetIn1 = First_lay * Inputs1;

  Hidden = 1-2./(exp(2*NetIn1)+1); 
  Inputs2 = [Hidden; ones(1, Data_in)];
  Out1 = Second_lay  * Inputs2;
 
  Out1 = Out1 + In_out * Inputs1(1:end-1,:);

  
  % Backward propagation of errors
  Error = Desired - Out1;
  TSS = sum(sum(Error.^2 ));


  bperr = ( Second_lay' * Error );

  HiddenBeta = (1.0 - Hidden .^2 ) .* bperr(1:end-1,:);
  DFirst = HiddenBeta * Inputs1';
  DSecond = Error * Inputs2';
  DThird = Error * Inputs1(1:end-1,:)';

  First_lay = First_lay + DFirst * LearnRate;
  Second_lay = Second_lay + DSecond * LearnRate;
  In_out = In_out + DThird * LearnRate;


  MSE = TSS / Data_in;
  MSEPlot2(epoch)=MSE;
  fprintf('MSE = %f\n', MSE);
end


figure(1);
plot(year(11:288), Desired, 'k', 'LineWidth', 1.5); % Plot Desired in black
hold on;
plot(year(11:288), Out, 'b', 'LineWidth', 1.1);     % Plot Out in red

% Add legend
legend('Actual Data', 'Predicted (Out)');
xlabel('Year');
ylabel('Sunspot Activity');
title('Sunspot Data');
hold off;

figure(2);
step_size = 10;
indices = 11:step_size:278;
plot(year(indices), Desired(indices), 'k', 'LineWidth', 1.5); % Plot Desired in black
hold on;
plot(year(indices), Out(indices), 'b', 'LineWidth', 1.1);     % Plot Out in red

% Add legend
legend('Actual Data', 'Predicted (Out)');
xlabel('Year');
ylabel('Sunspot Activity');
title('Sunspot Data Reduced');
hold off;

figure(3);
plot(1:length(MSEPlot), MSEPlot, "r", 'LineWidth',1.5);
hold on;
plot(1:length(MSEPlot2), MSEPlot2, "b", 'LineWidth', 1.5);

legend('Newton', 'Regular Back Prop', 'Location', 'northeast');
xlabel('Epochs');
ylabel('Mean Squared Error (MSE)');
title('Training and Validation MSE Progression');
hold off;