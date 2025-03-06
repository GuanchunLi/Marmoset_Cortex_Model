load spine
%load ('C:\Work-NAS\Working_Project\Large-Scale-Model\Marmoset_exp_data\Go20150826S1\ElectrodesGo.mat')
load('ElectrodesJi.mat')
image(LINE);
colormap(map);
hold on;
for i=1:length(X)
  plot(X,Y,'O');
  text(X(i),Y(i),num2str(i))
  hold on
end
% plot(XI,YI,'r*');

% load('C:\Work-NAS\Working_Project\Large-Scale-Model\Marmoset_exp_data\Hn20160120S1\Event.mat')
