function [y_pred, gamma] = Predict_ODMM(model, Xlist)
m = size(Xlist, 3);
gamma = zeros(1, m);
for k=1:m
    gamma(k) = sum(model.W .* Xlist(:, :, k), 'all') + model.b;
end
y_pred = sign(gamma);