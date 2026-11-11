function ax = plot_topoplot(values, positions, plot_title, ax)
%PLOT_TOPOPLOT Optional MATLAB plot of a channel vector and N-by-2 positions.
% Coordinates use a head radius of 0.5; x points right and y anterior.
% No FieldTrip, montage files, participant IDs, or study inputs are needed.
if nargin<3,plot_title='';end
if nargin<4,figure('Color','w');ax=axes;end
values=double(values(:));positions=double(positions);
assert(isequal(size(positions),[numel(values),2]) && numel(values)>=3);
assert(all(isfinite(values)) && all(isfinite(positions),'all'));
assert(size(unique(positions,'rows'),1)==numel(values));
assert(rank(positions-mean(positions,1))==2);
[x,y]=meshgrid(linspace(-.5,.5,150));
interpolant=scatteredInterpolant(positions(:,1),positions(:,2),values,'natural','nearest');
z=interpolant(x,y);mask=x.^2+y.^2<=.5^2;
image=imagesc(ax,[-.5 .5],[-.5 .5],z);image.AlphaData=mask;
set(ax,'YDir','normal');hold(ax,'on');
theta=linspace(0,2*pi,200);
plot(ax,.5*cos(theta),.5*sin(theta),'k','LineWidth',1);
plot(ax,[-.06 0 .06],[.495 .56 .495],'k','LineWidth',1);
scatter(ax,positions(:,1),positions(:,2),12,'k','filled');
axis(ax,'equal');axis(ax,'off');ylim(ax,[-.52 .58]);
title(ax,plot_title,'Interpreter','none');colorbar(ax);
end
