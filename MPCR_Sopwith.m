%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------------------------------------%
%
% Machine Perception and Cognitive Robotics Laboratory
%
%     Center for Complex Systems and Brain Sciences
%
%              Florida Atlantic University
%
%------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------------------------------------%
% Sopwith 1984
% https://en.wikipedia.org/wiki/Sopwith_(video_game)
%
% Boxer/DosBox 
% http://boxerapp.com/ 
% http://www.dosbox.com/
%------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------------------------------------%

function MPCR_Sopwith
clear all
close all
clc
beep off

% rng(123) % set seed for random number generator

format long



import java.awt.Robot;

import java.awt.event.*;

robot=Robot();



x=175;
y=175;
w=600;
h=500;


image1 = robot.createScreenCapture(java.awt.Rectangle(x,y,w,h));

data=image1.getData();
pix=data.getPixels(0,0,w,h,[]);
tmp=reshape(pix(:),3,w,h);

for i=1:3
    outputImage(:,:,i)=squeeze(tmp(i,:,:))';
end

pattern=outputImage/255;

figure(2)
imagesc(pattern)

% return



pattern=im2double(pattern(:))';




bias=ones(size(pattern,1),1);
pattern = [pattern bias];

n1 = size(pattern,2);   %Set the Number of Input Nodes Equal to Number of Pixels in the Input image
n2 = 26;   %n2-1        %Number of Hidden Nodes (Free Parameter)
n3 = 3;%size(category,2);  %Set the Number of Output Nodes Equal to the Number of Distinct Categories {left,forward,right}


w1 = 0.5*(1-2*rand(n1,n2-1)); %Randomly Initialize Hidden Weights
w2 = 0.5*(1-2*rand(n2,n3));   %Randomly Initialize Output Weights

dw1 = zeros(size(w1));          %Set Initial Hidden Weight Changes to Zero
dw2 = zeros(size(w2));          %Set Initial Output Changes to Zero

L = 0.25;             % Learning    %Avoid Overshooting Minima
M = 0.8;           % Momentum    %Smooths out the learning landscape



c=100;


choice={'e_up', 'roll', 'e_down', 'auto'};






while true
    
    robot.mouseMove(200,1350)
    robot.mousePress(InputEvent.BUTTON1_MASK);
    robot.mouseRelease(InputEvent.BUTTON1_MASK);
    clc
    
    
    image1 = robot.createScreenCapture(java.awt.Rectangle(x,y,w,h));
    
    data=image1.getData();
    pix=data.getPixels(0,0,w,h,[]);
    tmp=reshape(pix(:),3,w,h);
    
    for i=1:3
        outputImage(:,:,i)=squeeze(tmp(i,:,:))';
    end
    
    pattern=outputImage/255;
    
    figure(2)
    imagesc(pattern)
    
    
    pattern=im2double(pattern(:))';
    
    
    
    pattern = pattern(ones(1,c),:);
    
    
    pattern = pattern + 0.5*randn(size(pattern));
    
    
    
    
    
    %     for i=1:size(pattern,1)
    %
    %     figure(4)
    %     imagesc(reshape(pattern(i,:),251,960))
    %     pause
    %
    %     end
    
    
    
    
    
    
    bias=ones(size(pattern,1),1);
    
    
    pattern = [pattern bias];
    
    
    
    
    act1 = [af(0.1*pattern * w1) bias];
    
    
    
    act2 = af(act1 * w2);
    
    
    
    act22=HahnWTA(act2);
    
    
    [uA,~,uIdx] = unique(act22,'rows');
    act2mode = uA(mode(uIdx),:);
    
    
    
   
    motor = input(['Enter 1 for elevator up, 2 for roll, 3 for elevator down, 4 for network(' choice{find(act2mode)} ')-->']);
    
    
    
    
    
    
    switch motor
        
        case 1
            
            category=[1 0 0];
            category=category(ones(1,c),:);
            
            motor1=1;
            
            
            
        case 2
            
            category=[0 1 0];
            category=category(ones(1,c),:);
            motor1=2;
            
        case 3
            
            category=[0 0 1];
            category=category(ones(1,c),:);
            motor1=3;
            
        case 4
            
            category=act2;
            
            motor1=find(act2mode);
            
            
            
            
    end
    
    
    
    
    
    
    switch motor1
        
        case 1
            
            robot.mouseMove(200,40)
            robot.mousePress(InputEvent.BUTTON1_MASK);
            robot.mouseRelease(InputEvent.BUTTON1_MASK);
            
            
            
            
            robot.keyPress(KeyEvent.VK_COMMA);
            robot.keyRelease(KeyEvent.VK_COMMA);
            
            
            
      
            
            
        case 2
            
            robot.mouseMove(200,40)
            robot.mousePress(InputEvent.BUTTON1_MASK);
            robot.mouseRelease(InputEvent.BUTTON1_MASK);
            
            
            
            robot.keyPress(KeyEvent.VK_PERIOD);
            robot.keyRelease(KeyEvent.VK_PERIOD);
            
       
            
            
            
        case 3
            
            
            robot.mouseMove(200,40)
            robot.mousePress(InputEvent.BUTTON1_MASK);
            robot.mouseRelease(InputEvent.BUTTON1_MASK);
            
            
            robot.keyPress(KeyEvent.VK_SLASH);
            robot.keyRelease(KeyEvent.VK_SLASH);
            
            
          
            
            
            
    end
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    error = category - act2;  %Calculate Error
    
    
    
    delta_w2 = error .* act2 .* (1-act2); %Backpropagate Errors
    delta_w1 = delta_w2*w2' .* act1 .* (1-act1);
    
    
    delta_w1(:,size(delta_w1,2)) = []; %Remove Bias
    
    
    dw1 = L * pattern' * delta_w1 + M * dw1; %Calculate Hidden Weight Changes
    dw2 = L * act1' * delta_w2 + M * dw2;    %Calculate Output Weight Changes
    
    
    
    w1 = w1 + dw1; %Adjust Hidden Weights
    w2 = w2 + dw2; %Adjust Output Weights
    
    
    
%     w1 = w1 + 0.001*rand(size(w1)); %Adjust Hidden Weights
%     w2 = w2 + 0.001*rand(size(w2)); %Adjust Output Weights
    
    
    
    
    
    
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Visualize Input Weights as Receptive Fields
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    figure(1)
    
    Wp=w1;
    Wp = ((Wp - min(min(Wp)))/(max(max(Wp)) - min(min(Wp))));
    
    
    for i =1:n2-1
        subplot(sqrt(n2-1),sqrt(n2-1),i)
        imagesc(reshape(Wp(1:n1-1,i),500,600,3))
        axis off
    end
    drawnow()
    
   
    
    drawnow()
    
    
    
    
    
    
    
    
end %end while loop






end %end function













%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------------------------------------%
%--------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function action = af (weighted_sum)


action = 1./(1+exp(-weighted_sum));  		% Logistic / Sigmoid Function


end




function x = HahnWTA(x)



for i=1:size(x,1)
    
    [a,b]=max(x(i,:));
    
    x(i,:)=1:size(x,2)==b;
    
    
end



end




