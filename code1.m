%% First section

clear all;
close all;

% load the data
irisdata = load('irisdata.dat');
irisdata_ori = irisdata(:,2:5);

% attributes
sepal_length = irisdata(:, 2);
sepal_width = irisdata(:, 3);
petal_length = irisdata(:, 4);
petal_width = irisdata(:, 5);

% attributes in 50x3 matrix
sepal_length_all = [sepal_length(1:50) sepal_length(51:100) sepal_length(101:150)];
sepal_width_all = [sepal_width(1:50) sepal_width(51:100) sepal_width(101:150)];
petal_length_all = [petal_length(1:50) petal_length(51:100) petal_length(101:150)];
petal_width_all = [petal_width(1:50) petal_width(51:100) petal_width(101:150)];

% calculate the average of each attributes for each classes
setosa_sepal_l = sepal_length(1:50);
setosa_sepal_w = sepal_width(1:50);
setosa_petal_l = petal_length(1:50);
setosa_petal_w = petal_width(1:50);
avg_setosa_sepal_l = sum(setosa_sepal_l) / 50;
avg_setosa_sepal_w = sum(setosa_sepal_w) / 50;
avg_setosa_petal_l = sum(setosa_petal_l) / 50;
avg_setosa_petal_w = sum(setosa_petal_w) / 50;

versicolor_sepal_l = sepal_length(51:100);
versicolor_sepal_w = sepal_width(51:100);
versicolor_petal_l = petal_length(51:100);
versicolor_petal_w = petal_width(51:100);
avg_versicolor_sepal_l = sum(versicolor_sepal_l) / 50;
avg_versicolor_sepal_w = sum(versicolor_sepal_w) / 50;
avg_versicolor_petal_l = sum(versicolor_petal_l) / 50;
avg_versicolor_petal_w = sum(versicolor_petal_w) / 50;

virginica_sepal_l = sepal_length(101:150);
virginica_sepal_w = sepal_width(101:150);
virginica_petal_l = petal_length(101:150);
virginica_petal_w = petal_width(101:150);
avg_virginica_sepal_l = sum(virginica_sepal_l) / 50;
avg_virginica_sepal_w = sum(virginica_sepal_w) / 50;
avg_virginica_petal_l = sum(virginica_petal_l) / 50;
avg_virginica_petal_w = sum(virginica_petal_w) / 50;

%% Horizontal line plot
close all;

% sepal length
s_l_p = figure;
title('Sepal Lengths (cm)'); hold on
plot(sepal_length(1:50),0,'rs','MarkerSize',10); hold on 
plot(sepal_length(51:100),0,'go','MarkerSize',10); hold on
plot(sepal_length(101:150),0,'bx','MarkerSize',10); hold on
set(gca,'box','off', 'xtick', [], 'ytick', [], 'XColor', 'none', 'YColor', 'none');
set(gcf, 'Position', [100 100 700 100]);
hAxes = axes('NextPlot','add','DataAspectRatio',[1 1 1],'XLim',[0 max(sepal_length)],...
    'YLim', [0 eps]); hold off
saveas(s_l_p, 'sepal_length_line_plot.jpg');

% sepal width
s_w_p = figure;
title('Sepal Widths (cm)'); hold on
plot(sepal_width(1:50),0,'rs','MarkerSize',10); hold on 
plot(sepal_width(51:100),0,'go','MarkerSize',10); hold on
plot(sepal_width(101:150),0,'bx','MarkerSize',10); hold on
set(gca,'box','off', 'xtick', [], 'ytick', [], 'XColor', 'none', 'YColor', 'none');
set(gcf, 'Position', [100 100 700 100]);
hAxes = axes('NextPlot','add','DataAspectRatio',[1 1 1],'XLim',[0 max(sepal_width)],...
    'YLim', [0 eps]); hold off
saveas(s_w_p, 'sepal_width_line_plot.jpg');

% petal length
p_l_p = figure;
title('Petal Lengths (cm)'); hold on
plot(petal_length(1:50),0,'rs','MarkerSize',10); hold on 
plot(petal_length(51:100),0,'go','MarkerSize',10); hold on
plot(petal_length(101:150),0,'bx','MarkerSize',10); hold on
set(gca,'box','off', 'xtick', [], 'ytick', [], 'XColor', 'none', 'YColor', 'none');
set(gcf, 'Position', [100 100 700 100]);
hAxes = axes('NextPlot','add','DataAspectRatio',[1 1 1],'XLim',[0 max(petal_length)],...
    'YLim', [0 eps]); hold off
saveas(p_l_p, 'petal_length_line_plot.jpg');

% petal width
p_w_p = figure;
title('Petal Widths (cm)'); hold on
plot(petal_width(1:50),0,'rs','MarkerSize',10); hold on 
plot(petal_width(51:100),0,'go','MarkerSize',10); hold on
plot(petal_width(101:150),0,'bx','MarkerSize',10); hold on
set(gca,'box','off', 'xtick', [], 'ytick', [], 'XColor', 'none', 'YColor', 'none');
set(gcf, 'Position', [100 100 700 100]);
hAxes = axes('NextPlot','add','DataAspectRatio',[1 1 1],'XLim',[0 max(petal_width)],...
    'YLim', [0 eps]); hold off
saveas(p_w_p, 'petal_width_line_plot.jpg');

%% Histogram plot 
close all;

% sepal length
figure;
h_s_l = histogram(sepal_length, 20);
title('Iris data - Histogram of sepal lengths');
xlabel('Sepal Lengths (cm)');
saveas(h_s_l, 'sepal_length_histogram.jpg');

% sepal width
figure;
h_s_w = histogram(sepal_width, 20);
title('Iris data - Histogram of sepal widths');
xlabel('Sepal Widths (cm)');
saveas(h_s_w, 'sepal_width_histogram.jpg');

% petal length
figure;
h_p_l = histogram(petal_length, 20);
title('Iris data - Histogram of petal lengths');
xlabel('Petal Lengths (cm)');
saveas(h_p_l, 'petal_length_histogram.jpg');

% petal length
figure;
h_p_w = histogram(petal_width, 20);
title('Iris data - Histogram of petal widths');
xlabel('Petal Widths (cm)');
saveas(h_p_w, 'petal_width_histogram.jpg');


%% Box plot
close all;

% sepal length
labels = {'Iris Setosa', 'Iris Versicolor', 'Iris Virginica'};
data = [sepal_length(1:50), sepal_length(51:100), sepal_length(101:150)];
figure;
b_s_l = boxplot(data, labels);
title('Distribution of sepal length');
xlabel('Classes');
ylabel('Sepal Length (cm)');
saveas(gcf, 'sepal_length_box.jpg');

% sepal width
labels = {'Iris Setosa', 'Iris Versicolor', 'Iris Virginica'};
data = [sepal_width(1:50), sepal_width(51:100), sepal_width(101:150)];
figure;
b_s_w = boxplot(data, labels);
title('Distribution of sepal width');
xlabel('Classes');
ylabel('Sepal Width (cm)');
saveas(gcf, 'sepal_width_box.jpg');

% petal length
labels = {'Iris Setosa', 'Iris Versicolor', 'Iris Virginica'};
data = [petal_length(1:50), petal_length(51:100), petal_length(101:150)];
figure;
b_p_l = boxplot(data, labels);
title('Distribution of petal length');
xlabel('Classes');
ylabel('Petal Length (cm)');
saveas(gcf, 'petal_length_box.jpg');

% petal width
labels = {'Iris Setosa', 'Iris Versicolor', 'Iris Virginica'};
data = [petal_width(1:50), petal_width(51:100), petal_width(101:150)];
figure;
b_p_w = boxplot(data, labels);
title('Distribution of petal width');
xlabel('Classes');
ylabel('Petal Width (cm)');
saveas(gcf, 'petal_width_box.jpg');


%% 2-D plot
close all;

% sepal length vs sepal width
figure;
x1 = sepal_length_all(:, 1)';
y1 = sepal_width_all(:, 1)';
scatter(x1, y1, 'ro', 'filled'); hold on

x2 = sepal_length_all(:, 2)';
y2 = sepal_width_all(:, 2)';
scatter(x2, y2, 'go', 'filled'); hold on

x3 = sepal_length_all(:, 3)';
y3 = sepal_width_all(:, 3)';
scatter(x3, y3, 'bo', 'filled'); hold off
title('Iris Data');
xlabel('Sepal Width (cm)');
ylabel('Sepal Length (cm)');
saveas(gcf, 'sepal_length_vs_sepal_width.jpg');


figure;
x1 = sepal_length_all(:, 1)';
y1 = sepal_width_all(:, 1)';
scatter(y1, x1, 'ro', 'filled'); hold on

x2 = sepal_length_all(:, 2)';
y2 = sepal_width_all(:, 2)';
scatter(y2, x2, 'go', 'filled'); hold on

x3 = sepal_length_all(:, 3)';
y3 = sepal_width_all(:, 3)';
scatter(y3, x3, 'bo', 'filled'); hold off
title('Iris Data');
ylabel('Sepal Width (cm)');
xlabel('Sepal Length (cm)');
saveas(gcf, 'sepal_width_vs_sepal_length.jpg');

% sepal length vs petal length
figure;
x1 = sepal_length_all(:, 1)';
y1 = petal_length_all(:, 1)';
scatter(x1, y1, 'ro', 'filled'); hold on

x2 = sepal_length_all(:, 2)';
y2 = petal_length_all(:, 2)';
scatter(x2, y2, 'go', 'filled'); hold on

x3 = sepal_length_all(:, 3)';
y3 = petal_length_all(:, 3)';
scatter(x3, y3, 'bo', 'filled'); hold off
title('Iris Data');
xlabel('Sepal Length (cm)');
ylabel('Petal Length (cm)');
saveas(gcf, 'sepal_length_vs_petal_length.jpg');


figure;
x1 = sepal_length_all(:, 1)';
y1 = petal_length_all(:, 1)';
scatter(y1, x1, 'ro', 'filled'); hold on

x2 = sepal_length_all(:, 2)';
y2 = petal_length_all(:, 2)';
scatter(y2, x2, 'go', 'filled'); hold on

x3 = sepal_length_all(:, 3)';
y3 = petal_length_all(:, 3)';
scatter(y3, x3, 'bo', 'filled'); hold off
title('Iris Data');
ylabel('Sepal Length (cm)');
xlabel('Petal Length (cm)');
saveas(gcf, 'petal_length_vs_sepal_length.jpg');

% sepal length vs petal width
figure;
x1 = sepal_length_all(:, 1)';
y1 = petal_width_all(:, 1)';
scatter(x1, y1, 'ro', 'filled'); hold on

x2 = sepal_length_all(:, 2)';
y2 = petal_width_all(:, 2)';
scatter(x2, y2, 'go', 'filled'); hold on

x3 = sepal_length_all(:, 3)';
y3 = petal_width_all(:, 3)';
scatter(x3, y3, 'bo', 'filled'); hold off
title('Iris Data');
xlabel('Sepal Length (cm)');
ylabel('Petal Width (cm)');
saveas(gcf, 'sepal_length_vs_petal_width.jpg');


figure;
x1 = sepal_length_all(:, 1)';
y1 = petal_width_all(:, 1)';
scatter(y1, x1, 'ro', 'filled'); hold on

x2 = sepal_length_all(:, 2)';
y2 = petal_width_all(:, 2)';
scatter(y2, x2, 'go', 'filled'); hold on

x3 = sepal_length_all(:, 3)';
y3 = petal_width_all(:, 3)';
scatter(y3, x3, 'bo', 'filled'); hold off
title('Iris Data');
ylabel('Sepal Length (cm)');
xlabel('Petal Width (cm)');
saveas(gcf, 'petal_width_vs_sepal_length.jpg');

% sepal width vs petal length
figure;
x1 = sepal_width_all(:, 1)';
y1 = petal_length_all(:, 1)';
scatter(x1, y1, 'ro', 'filled'); hold on

x2 = sepal_width_all(:, 2)';
y2 = petal_length_all(:, 2)';
scatter(x2, y2, 'go', 'filled'); hold on

x3 = sepal_width_all(:, 3)';
y3 = petal_length_all(:, 3)';
scatter(x3, y3, 'bo', 'filled'); hold off
title('Iris Data');
xlabel('Sepal Width (cm)');
ylabel('Petal Length (cm)');
saveas(gcf, 'sepal_width_vs_petal_length.jpg');


figure;
x1 = sepal_width_all(:, 1)';
y1 = petal_length_all(:, 1)';
scatter(y1, x1, 'ro', 'filled'); hold on

x2 = sepal_width_all(:, 2)';
y2 = petal_length_all(:, 2)';
scatter(y2, x2, 'go', 'filled'); hold on

x3 = sepal_width_all(:, 3)';
y3 = petal_length_all(:, 3)';
scatter(y3, x3, 'bo', 'filled'); hold off
title('Iris Data');
ylabel('Sepal Width (cm)');
xlabel('Petal Length (cm)');
saveas(gcf, 'petal_length_vs_sepal_width.jpg');

% sepal width vs petal width
figure;
x1 = sepal_width_all(:, 1)';
y1 = petal_width_all(:, 1)';
scatter(x1, y1, 'ro', 'filled'); hold on

x2 = sepal_width_all(:, 2)';
y2 = petal_width_all(:, 2)';
scatter(x2, y2, 'go', 'filled'); hold on

x3 = sepal_width_all(:, 3)';
y3 = petal_width_all(:, 3)';
scatter(x3, y3, 'bo', 'filled'); hold off
title('Iris Data');
xlabel('Sepal Width (cm)');
ylabel('Petal Width (cm)');
saveas(gcf, 'sepal_width_vs_petal_width.jpg');


figure;
x1 = sepal_width_all(:, 1)';
y1 = petal_width_all(:, 1)';
scatter(y1, x1, 'ro', 'filled'); hold on

x2 = sepal_width_all(:, 2)';
y2 = petal_width_all(:, 2)';
scatter(y2, x2, 'go', 'filled'); hold on

x3 = sepal_width_all(:, 3)';
y3 = petal_width_all(:, 3)';
scatter(y3, x3, 'bo', 'filled'); hold off
title('Iris Data');
ylabel('Sepal Width (cm)');
xlabel('Petal Width (cm)');
saveas(gcf, 'petal_width_vs_sepal_width.jpg');

% petal length vs petal width
figure;
x1 = petal_length_all(:, 1)';
y1 = petal_width_all(:, 1)';
scatter(x1, y1, 'ro', 'filled'); hold on

x2 = petal_length_all(:, 2)';
y2 = petal_width_all(:, 2)';
scatter(x2, y2, 'go', 'filled'); hold on

x3 = petal_length_all(:, 3)';
y3 = petal_width_all(:, 3)';
scatter(x3, y3, 'bo', 'filled'); hold off
title('Iris Data');
xlabel('Petal Length (cm)');
ylabel('Petal Width (cm)');
saveas(gcf, 'petal_length_vs_petal_width.jpg');


figure;
x1 = petal_length_all(:, 1)';
y1 = petal_width_all(:, 1)';
scatter(y1, x1, 'ro', 'filled'); hold on

x2 = petal_length_all(:, 2)';
y2 = petal_width_all(:, 2)';
scatter(y2, x2, 'go', 'filled'); hold on

x3 = petal_length_all(:, 3)';
y3 = petal_width_all(:, 3)';
scatter(y3, x3, 'bo', 'filled'); hold off
title('Iris Data');
ylabel('Petal Length (cm)');
xlabel('Petal Width (cm)');
saveas(gcf, 'petal_width_vs_petal_length.jpg');


%% Plot for sizes
close all;

% size = width * length
sepal_size_all = sepal_length.*sepal_width;
petal_size_all = petal_length.*petal_width;

min_sepal_size = min(sepal_size_all);
max_sepal_size = max(sepal_size_all);
avg_sepal_size = sum(sepal_size_all) / 50;

min_petal_size = min(petal_size_all);
max_petal_size = max(petal_size_all);
avg_petal_size = sum(petal_size_all) / 50;

% setosa
setosa_sepal_size = sepal_size_all(1:50);
setosa_petal_size = petal_size_all(1:50);
setosa_min_sepal_size = min(setosa_sepal_size);
setosa_max_sepal_size = max(setosa_sepal_size);
setosa_min_petal_size = min(setosa_petal_size);
setosa_max_petal_size = max(setosa_petal_size);
avg_setosa_sepal_s = sum(setosa_sepal_size) / 50;
avg_setosa_petal_s = sum(setosa_petal_size) / 50;

%versicolor
versicolor_sepal_size = sepal_size_all(51:100);
versicolor_petal_size = petal_size_all(51:100);
versicolor_min_sepal_size = min(versicolor_sepal_size);
versicolor_max_sepal_size = max(versicolor_sepal_size);
versicolor_min_petal_size = min(versicolor_petal_size);
versicolor_max_petal_size = max(versicolor_petal_size);
avg_versicolor_sepal_s = sum(versicolor_sepal_size) / 50;
avg_versicolor_petal_s = sum(versicolor_petal_size) / 50;

%virginica
virginica_sepal_size = sepal_size_all(101:150);
virginica_petal_size = petal_size_all(101:150);
virginica_min_sepal_size = min(virginica_sepal_size);
virginica_max_sepal_size = max(virginica_sepal_size);
virginica_min_petal_size = min(virginica_petal_size);
virginica_max_petal_size = max(virginica_petal_size);
avg_virginica_sepal_s = sum(virginica_sepal_size) / 50;
avg_virginica_petal_s = sum(virginica_petal_size) / 50;

% plot for sepal size
labels = {'Iris Setosa', 'Iris Versicolor', 'Iris Virginica'};
data = [sepal_size_all(1:50), sepal_size_all(51:100), sepal_size_all(101:150)];
figure;
b_s_l = boxplot(data, labels);
title('Distribution of sepal sizes');
xlabel('Classes');
ylabel('Sepal Size (cm^2)');
saveas(gcf, 'sepal_size_box.jpg');

% plot for petal size
labels = {'Iris Setosa', 'Iris Versicolor', 'Iris Virginica'};
data = [petal_size_all(1:50), petal_size_all(51:100), petal_size_all(101:150)];
figure;
b_s_l = boxplot(data, labels);
title('Distribution of petal sizes');
xlabel('Classes');
ylabel('Petal Size (cm^2)');
saveas(gcf, 'petal_size_box.jpg');

% 2D plot
figure;
x1 = petal_size_all(1:50)';
y1 = sepal_size_all(1:50)';
scatter(y1, x1, 'ro', 'filled'); hold on

x2 = petal_size_all(51:100)';
y2 = sepal_size_all(51:100)';
scatter(y2, x2, 'go', 'filled'); hold on

x3 = petal_size_all(101:150)';
y3 = sepal_size_all(101:150)';
scatter(y3, x3, 'bo', 'filled'); hold off
title('Iris Data');
ylabel('Sepal Size (cm^2)');
xlabel('Petal Size (cm^2)');
saveas(gcf, 'petal_size_vs_sepal_size.jpg');


%% PCA
close all;

%de-mean
X = bsxfun(@minus, irisdata_ori, mean(irisdata_ori));

% PCA
[coeff, score, latent] = pca(X);

% Calcualate the covariance matrix
covMatrix = cov(X);

% Eigenvalues and eigenvectors of the covariance matrix
[V,D] = eig(covMatrix);

% "coeff" are the principal component vectors.
% These are the eigenvectors of the covariance matrix
% compare coeff and V

% Multiply the original data by the principal component vectors to get
%  the projections of the original data on the principal component
%  vector space. This is also the output of score
dataInPrincipalCompSpace = X * coeff;
% Compare dataInPrincipalCompSpace and score

% The columns of X*coeff are orthogonal to each other
% this is shown with
corrcoef(dataInPrincipalCompSpace);

% The variances of these vectors are the eigenvalues of the covariance
%  matrix. These three outputs:
%var(dataInPrincipalCompSpace)'
%latent
%sort(diag(D),'descend')


%% PCA 2
close all;

[W, pc] = pca(irisdata_ori);
pc = pc';
W = W';

% Draw stuffs
figure;
pca2_plot = plot(pc(1,:), pc(2,:), '.');
title('{\bf PCA}');
xlabel('PC 1');
ylabel('PC 2');
saveas(pca2_plot, 'pca2_plot.jpg');

% remove the mean variable-wise (row-wise)
data = irisdata_ori';
data = data - repmat(mean(data,2), 1, size(data, 2));

% calculate eigenvectors (loadings) W, and eigenvalues of the covariance
%  matrix. 
[W, E_value_matrix] = eig(cov(data'));
E_values = diag(E_value_matrix);
E_values_unsorted = E_values;

plot(real(E_values), imag(E_values), 'r*');

% order by largest eigenvalue
E_values = E_values(end:-1:1);
W = W(:,end:-1:1);
W = W';

% generate PCA component space (PCA scores)
pc = W * data;

% plot PCA space of the first two PCs: PC1 and PC2
figure;
scatter(pc(1,1:50), pc(2,1:50), 'ro', 'filled'); hold on
scatter(pc(1,51:100), pc(2,51:100), 'go', 'filled'); hold on
scatter(pc(1,101:150), pc(2,101:150), 'bo', 'filled'); hold off
title('Principal Component Analysis');
ylabel('Principal Component 2');
xlabel('Principal Component 1');
saveas(gcf, 'pca2_plot2.jpg');

% biplot
figure;
h = biplot(coeff(:,1:2), 'scores', score(:,1:2), 'varlabels',...
    {'sepal length', 'sepal width', 'petal length', 'petal width'});
saveas(h, 'biplot.jpg');

%% PCA 3 - Eigenvalues
close all;

x_axis = [1 2 3 4];

figure;
eigenvalue_spec = plot(x_axis, E_values);
title('Eigenvalue spectrum');
set(gca, 'XLim', [1 5], 'XTick', [0:1:5], 'YTick', [0:1:5]);
saveas(eigenvalue_spec, 'eigenvalue_spectrum.jpg');

cumulative_ev = cumsum(E_values);
cumulative_ev = cumulative_ev / max(cumulative_ev);
figure;
cumulative_eigenvalue = plot(x_axis, cumulative_ev);
title('Cumulative sum of eigenvalues');
set(gca, 'XLim', [0 5], 'XTick', [0:1:5]);
saveas(cumulative_eigenvalue, 'cumulative_eigenvalues.jpg');


%% New Label - Add size as another attribute

% add sepal sizes
irisdata(:,6) = sepal_size_all;

% add petal sizes
irisdata(:,7) = petal_size_all;

% do another PCA (with new dimensions)

data = irisdata(:,2:7)';
data = data - repmat(mean(data,2), 1, size(data, 2));

% calculate eigenvectors (loadings) W, and eigenvalues of the covariance
%  matrix. 
[W_s, E_value_matrix_s] = eig(cov(data'));
E_values_s = diag(E_value_matrix_s);
E_values_s_unsorted = E_values_s;

% order by largest eigenvalue
E_values_s = E_values_s(end:-1:1);
W_s = W_s(:,end:-1:1);
W_s = W_s';

% generate PCA component space (PCA scores)
pc_s = W_s * data;

% eigenvalues
axis = [1 2 3 4 5 6];

figure;
eigenvalue_spec_s = plot(axis, E_values_s);
title('Eigenvalue spectrum (new dimension)');
set(gca, 'XLim', [1 6], 'XTick', [0:1:6], 'YTick', [0:5:35]);
saveas(eigenvalue_spec_s, 'eigenvalue_spectrum_new_dimension.jpg');

% cumulative eigenvalues
cumulative_ev_s = cumsum(E_values_s);
cumulative_ev_s = cumulative_ev_s / max(cumulative_ev_s);
figure;
cumulative_eigenvalue_s = plot(axis, cumulative_ev_s);
title('Cumulative sum of eigenvalues (new dimension)');
set(gca, 'XLim', [0 6], 'XTick', [0:1:6]);
saveas(cumulative_eigenvalue_s, 'cumulative_eigenvalues_new_dimension.jpg');

% pca plot
figure;
scatter(pc_s(1,1:50), pc_s(2,1:50), 'ro', 'filled'); hold on
scatter(pc_s(1,51:100), pc_s(2,51:100), 'go', 'filled'); hold on
scatter(pc_s(1,101:150), pc_s(2,101:150), 'bo', 'filled'); hold off
title('Principal Component Analysis (new dimension)');
ylabel('Principal Component 2');
xlabel('Principal Component 1');
saveas(gcf, 'pca2_plot2_new_dimension.jpg');

% biplot

%de-mean
X = bsxfun(@minus, irisdata(:,2:7), mean(irisdata(:,2:7)));

% PCA
[coeff, score, latent] = pca(X);

figure;
h_s = biplot(coeff(:,1:2), 'scores', score(:,1:2), 'varlabels',...
    {'sepal length', 'sepal width', 'petal length', 'petal width',...
    'sepal size', 'petal size'});
saveas(h_s, 'biplot.jpg');