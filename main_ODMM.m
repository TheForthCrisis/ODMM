clc, clear
rng(1)
% Load dataset
load("AnimalFace.mat")

% Parameter setting
lambda = 1;
tau = 16;
mu = 0.4;
theta = 0.4;

rho = 1;
iteration = 100;

% Normalize dataset
mm = Normalizer(3, X_train);
X_train = mm.transform(X_train);
X_test = mm.transform(X_test);

% Train and predict
model = Train_ODMM(X_train, y_train, lambda, theta, mu, tau, rho, 'iter', iteration, 'InfoOutput', 2);
y_pred = Predict_ODMM(model, X_test);

% Accuracy
acc = sum(y_test==y_pred)/length(y_test);

disp(acc)