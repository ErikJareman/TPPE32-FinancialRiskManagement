weeklyData = readmatrix('lab1.xlsx', 'Sheet', 'Weekly', 'Range', 'A2:C1255');
dailyData = readmatrix('lab1.xlsx', 'Sheet', 'Daily', 'Range', 'A2:C6032');
garchOmxVariance = readmatrix('lab1.xlsx', 'Sheet', 'Weekly', 'Range', 'U4:U1255');
garchFxVariance = readmatrix('lab1.xlsx', 'Sheet', 'Weekly', 'Range', 'AP4:AP1255');

% Variabler som används ofta:
nWeekly = size(weeklyData, 1);
rWeeklyOMX = returnArray(weeklyData, 2, nWeekly);
rWeeklyFX = returnArray(weeklyData, 3, nWeekly);
nDaily = size(dailyData, 1);
rDailyOMX = returnArray(dailyData, 2, nDaily);
rDailyFX = returnArray(dailyData, 3, nDaily);

% Uppgift 1
% a)
% Tidsserier plottade i samma graf
figure(1);
yyaxis left
plot(weeklyData(:,1), weeklyData(:,2))
yyaxis right
plot(weeklyData(:,1), weeklyData(:,3))

% Log. avkastningar för veckodata plottat i en graf
figure(2);
yyaxis left
plot(rWeeklyOMX);
yyaxis right
plot(rWeeklyFX);

% Genomsnittlig avkastning för veckodata (samt daglig för senare)
myDailyOMX = calcMy(nDaily, rDailyOMX);
myDailyFX = calcMy(nDaily, rDailyFX);
myWeeklyOMX = calcMy(nWeekly, rWeeklyOMX);
myWeeklyFX = calcMy(nWeekly, rWeeklyFX);
myYearlyOMX = myWeeklyOMX * 52;
myYearlyFX = myWeeklyFX * 52;

% Volatilitet för veckodata
volWeeklyOMX = calcVol(nWeekly, rWeeklyOMX, myWeeklyOMX);
volWeeklyFX = calcVol(nWeekly, rWeeklyFX, myWeeklyFX);
volYearlyOMX = volWeeklyOMX * sqrt(52);
volYearlyFX = volWeeklyFX * sqrt(52);

% CI95 förväntade logaritmiska avkastningar
sqrtN = sqrt((nWeekly - 1)/52);
hiOMX = myYearlyOMX + (1.96 * volYearlyOMX / sqrtN);
loOMX = myYearlyOMX - (1.96 * volYearlyOMX / sqrtN);
hiFX = myYearlyFX + (1.96 * volYearlyFX / sqrtN);
loFX = myYearlyFX - (1.96 * volYearlyFX / sqrtN);


% b)
% i) skevhet- samt kurtosis för daglig- och veckodata
skevhetWeeklyOMX = skevhet(nWeekly, rWeeklyOMX, myWeeklyOMX);
skevhetWeeklyFX = skevhet(nWeekly, rWeeklyFX, myWeeklyFX);
skevhetDailyOMX = skevhet(nDaily, rDailyOMX, myDailyOMX);
skevhetDailyFX = skevhet(nDaily, rDailyFX, myDailyFX);

kurtosisWeeklyOMX = kurtosis(nWeekly, rWeeklyOMX, myWeeklyOMX);
kurtosisWeeklyFX = kurtosis(nWeekly, rWeeklyFX, myWeeklyFX);
kurtosisDailyOMX = kurtosis(nDaily, rDailyOMX, myDailyOMX);
kurtosisDailyFX = kurtosis(nDaily, rDailyFX, myDailyFX);

% ii)Histogram för historisk avkastning samt 1,5,95,99 percentiler.
figure(3);
subplot(2,2,1)
histDailyOMX = histfit(rDailyOMX(), [], 'Normal');
title("Daily OMX");
subplot(2,2,2)
histDailyFX = histfit(rDailyFX(), [], 'Normal');
title("Daily USD/SEK");
subplot(2,2,3)
histWeeklyOMX = histfit(rWeeklyOMX(), [], 'Normal');
title("Weekly OMX");
subplot(2,2,4)
histWeeklyFX = histfit(rWeeklyFX(), [], 'Normal');
title("Weekly USD/SEK");

pw1DailyOMX = prctile(rDailyOMX(), 1);
pw5DailyOMX = prctile(rDailyOMX(), 5);
pw95DailyOMX = prctile(rDailyOMX(), 95);
pw99DailyOMX = prctile(rDailyOMX(), 99);
pw1WeeklyOMX = prctile(rWeeklyOMX(), 1);
pw5WeeklyOMX = prctile(rWeeklyOMX(), 5);
pw95WeeklyOMX = prctile(rWeeklyOMX(), 95);
pw99WeeklyOMX = prctile(rWeeklyOMX(), 99);
pw1DailyFX = prctile(rDailyFX(), 1);
pw5DailyFX = prctile(rDailyFX(), 5);
pw95DailyFX = prctile(rDailyFX(), 95);
pw99DailyFX = prctile(rDailyFX(), 99);
pw1WeeklyFX = prctile(rWeeklyFX(), 1);
pw5WeeklyFX = prctile(rWeeklyFX(), 5);
pw95WeeklyFX = prctile(rWeeklyFX(), 95);
pw99WeeklyFX = prctile(rWeeklyFX(), 99);

% iii) QQ-plot för samtliga fyra tidsserier (figure(4))
xWeekly = zeros(1, nWeekly-1);
rWeeklyOmxSort = sort(rWeeklyOMX);
rWeeklyFXSort = sort(rWeeklyFX);
for i = 1:nWeekly-1
    xWeekly(i) = (norminv((i-0.5)/nWeekly));
end

xDaily = zeros(1, nDaily-1);
rDailyOMXSort = sort(rDailyOMX);
rDailyFXSort = sort(rDailyFX);
for i = 1:nDaily-1
    xDaily(i) = (norminv((i-0.5)/nDaily));
end;

figure(4);
subplot(2,2,1);
scatter(xWeekly,rWeeklyOmxSort, '+');
xlim([-4 4])
ylim([-0.2 0.2])
title("Weekly OMX");

subplot(2,2,2);
scatter(xDaily,rDailyOMXSort, '+');
title("Daily OMX");

subplot(2,2,3);
scatter(xWeekly,rWeeklyFXSort, '+');
xlim([-4 4])
ylim([-0.05 0.05])
title("Weekly USD/SEK");

subplot(2,2,4);
scatter(xDaily,rDailyFXSort, '+');
title("Daily USD/SEK");

% Matlabs qqplot() som jämförelse (figure(5))
figure(5);
subplot(2,2,1)
qqplot(rWeeklyOMX)
title("Weekly OMX");

subplot(2,2,2)
qqplot(rDailyOMX)
title("Daily OMX");

subplot(2,2,3)
qqplot(rWeeklyFX)
title("Weekly USD/SEK");

subplot(2,2,4)
qqplot(rDailyFX)
title("Daily USD/SEK");

% Uppgift 2
% a) EqWMA för veckodata. 30 respektive 90 veckors fönster.
EqWMAOMX30 = EqWMA_arr(rWeeklyOMX, (nWeekly-1), 30);
EqWMAOMX90 = EqWMA_arr(rWeeklyOMX, (nWeekly-1), 90);
EqWMAFX30 = EqWMA_arr(rWeeklyFX, (nWeekly-1), 30);
EqWMAFX90 = EqWMA_arr(rWeeklyFX, (nWeekly-1), 90);
figure(6);
subplot(2,2,1)
plot(EqWMAOMX30);
title("EqWMA - OMXweekly - 30");
subplot(2,2,2)
plot(EqWMAOMX90);
title("EqWMA - OMXweekly - 90");
subplot(2,2,3)
plot(EqWMAFX30);
title("EqWMA - FXweekly - 30");
subplot(2,2,4)
plot(EqWMAFX90);
title("EqWMA - FXweekly - 90");

% b) EWMA för veckodata. RiskMetrics lambda (0.94).
EWMAWeeklyOMXarr = EWMA_arr(rWeeklyOMX, 0.94);
EWMAWeeklyFXarr = EWMA_arr(rWeeklyFX, 0.94);
figure(7)
subplot(2,1,1)
plot(EWMAWeeklyOMXarr * sqrt(52));
title("EWMA - OMX - RiskMetrics");
subplot(2,1,2)
plot(EWMAWeeklyFXarr * sqrt(52));
title("EWMA - FX - RiskMetrics");

% c) Solver / excel
% MLE lambda
EWMAlambdaOMX = 0.913351131;
EWMAlambdaFX = 0.902396586;
EWMAlambdaOMXarr = EWMA_arr(rWeeklyOMX, EWMAlambdaOMX);
EWMAlambdaFXarr = EWMA_arr(rWeeklyFX, EWMAlambdaFX);
figure(8)
subplot(2,1,1)
plot(EWMAlambdaOMXarr * sqrt(52));
title("EWMA - OMX - calcLambda");
subplot(2,1,2)
plot(EWMAlambdaFXarr * sqrt(52));
title("EWMA - FX - calcLambda");
% GARCH
figure(9)
subplot(2,1,1)
plot(sqrt(garchOmxVariance) * sqrt(52));
title("Garch - OMX");
subplot(2,1,2)
plot(sqrt(garchFxVariance) * sqrt(52));
title("Garch - FX");

% d) QQ-plot av std. tidsserier utifrån estimerade GARCH(1,1)-volatiliteter.
weeklyOmxGarchStand = rWeeklyOMX;
weeklyFxGarchStand = rWeeklyFX;
for i = 1:nWeekly-3
    weeklyOmxGarchStand(i) = (weeklyOmxGarchStand(i+1))/sqrt(garchOmxVariance(i));
    weeklyFxGarchStand(i) = (weeklyFxGarchStand(i+1))/sqrt(garchFxVariance(i));
end
figure(10);
subplot(2,1,1)
qqplot(weeklyOmxGarchStand);
title("weekly OMX (garch(1,1)) std.");
xlim([-4 4])
ylim([-4 4])

subplot(2,1,2)
qqplot(weeklyFxGarchStand);
title("weekly FX (garch(1,1)) std.");

% Uppgift 3
% a) Korrelation mellan tidsserierna (veckodata)
corrOMXFX = corr(rWeeklyOMX.', rWeeklyFX.');

% b) Autokorrelation i tidsserierna med log. avk. (1-5 veckors lag)
aCorrOMX1 = acorr(1, rWeeklyOMX);
aCorrOMX2 = acorr(2, rWeeklyOMX);
aCorrOMX3 = acorr(3, rWeeklyOMX);
aCorrOMX4 = acorr(4, rWeeklyOMX);
aCorrOMX5 = acorr(5, rWeeklyOMX);
aCorrFX1 = acorr(1, rWeeklyFX);
aCorrFX2 = acorr(2, rWeeklyFX);
aCorrFX3 = acorr(3, rWeeklyFX);
aCorrFX4 = acorr(4, rWeeklyFX);
aCorrFX5 = acorr(5, rWeeklyFX);

% c) 
% Anpassa data till olika copula och beräkna log-likelihood värden. 
x = weeklyOmxGarchStand;
y = weeklyFxGarchStand;
U = normcdf([x.', y.'], 0, 1);

rho_normal = copulafit('gaussian', U);
rho_clayton = copulafit('clayton', U);
rho_frank = copulafit('frank', U);
rho_gumbel = copulafit('gumbel', U);
[rho_t, frihet] = copulafit('t', U);

y_normal = copulapdf('gaussian', U, rho_normal);
y_clayton = copulapdf('clayton', U, rho_clayton);
y_frank = copulapdf('frank', U, rho_frank);
y_gumbel = copulapdf('gumbel', U, rho_gumbel);
y_t = copulapdf('t', U, rho_t, frihet);

sum_normal = 0;
sum_clayton = 0;
sum_frank = 0;
sum_gumbel = 0;
sum_t = 0;
for i = 1:nWeekly - 2
    sum_normal = sum_normal + log(y_normal(i));
    sum_clayton = sum_clayton + log(y_clayton(i));
    sum_frank = sum_frank + log(y_frank(i));
    sum_gumbel = sum_gumbel + log(y_gumbel(i));
    sum_t = sum_t + log(y_t(i));
end
%disp("Log-likelihood-värden copulas:");
%disp("   normal: " + sum_normal);
%disp("  clayton: " + sum_clayton);
%disp("    frank: " + sum_frank);
%disp("   gumbel: " + sum_gumbel);
%disp("        t: " + sum_t);


% Välj ut högsta log likelihood (t) och generera slumptal för denna.
% Jämför slumptal och tidsserier transformerade till likformig fördelning.
random = copularnd('t', rho_t, frihet, 1253);
x1 = random(:,1);
y1 = random(:,2);
figure(11);
subplot(2,1,1);
scatter(x1, y1);
title('Slump t');
subplot(2,1,2);
scatter(U(:,1), U(:,2));
title('Data t');


% --- output variabel använd i printResults() ---
output.RIC = {'.OMXS30', 'ERICb:ST'};
output.stat.mu = [myYearlyOMX myYearlyFX];
output.stat.sigma = [volYearlyOMX volYearlyFX];     
output.stat.CI = [loOMX hiOMX; loFX hiFX];
output.stat.skew = [skevhetWeeklyOMX skevhetWeeklyFX skevhetDailyOMX skevhetWeeklyFX];
output.stat.kurt = [kurtosisWeeklyOMX kurtosisWeeklyFX kurtosisDailyOMX kurtosisDailyFX];
output.stat.perc = [pw1DailyOMX pw5DailyOMX pw95DailyOMX pw99DailyOMX; pw1WeeklyOMX pw5WeeklyOMX pw95WeeklyOMX pw99WeeklyOMX;
                    pw1DailyFX pw5DailyFX pw95DailyFX pw99DailyFX; pw1WeeklyFX pw5WeeklyFX pw95WeeklyFX pw99WeeklyFX];
output.stat.corr = [corrOMXFX];
output.stat.acorr = [aCorrOMX1 aCorrFX1;
                     aCorrOMX2 aCorrFX2;
                     aCorrOMX3 aCorrFX3;
                     aCorrOMX4 aCorrFX4;
                     aCorrOMX5 aCorrFX5];
output.EWMA.obj = [7720.798778 9086.07641]; %[log-L (RIC1), log-L (RIC2)]
output.EWMA.param = [0.913351131 0.902396586]; %[lambda (RIC1), lambda (RIC2)]
output.GARCH.obj = [7774.939918 9225.712674];
output.GARCH.param = [sqrt(0.001072649 * 52), 0.146721664, 0.818243435, sqrt(0.00024445 * 52), 0.090904829, 0.81050282]; % [sigma, alpha, beta (RIC1), sigma, alpha, beta (RIC2) (unconstrained MLE)] %sigma is the yearly volatility, i.e. sqrt(VL*52), from MLE
output.GARCH.objVT = [7774.417373 9225.66413];
output.GARCH.paramVT = [sqrt(0.00090824*52), 0.137805564, 0.819829809, sqrt(0.00024899*52), 0.092413291, 0.810896465];%[sigma, alpha, beta (RIC1), sigma, alpha, beta (RIC2) (variance targeting)]
output.copulaLogL = [sum_normal sum_t sum_gumbel sum_clayton sum_frank];

printResults(output, true);

%------------------------------------------------------------------
function result = acorr(t, r)
    n = size(r, 2);
    rLower = zeros(1, n - t);
    rUpper = zeros(1, n - t);
    for i = 1:n
        if (i < n - t)
            rLower(i) = r(i);
        end
        if (i > t)
            rUpper(i - t) = r(i);
        end
    end
    result = corr(rLower.', rUpper.');
end

function result = skevhet(N, r, my)
    sumUpper = 0;
    sumLower = 0;
    for i = 1:(N-1)
        sumUpper = sumUpper + ((r(i) - my).^3);
        sumLower = sumLower + ((r(i) - my).^2);
    end
    numerator = (1/(N - 1)) * sumUpper;
    denominator = (sqrt((1/(N-1)) * sumLower)).^3;
    result = numerator / denominator;
end

function result = kurtosis(N, r, my)
    sumUpper = 0;
    sumLower = 0;
    for i = 1:(N-1)
        sumUpper = sumUpper + ((r(i) - my).^4);
        sumLower = sumLower + ((r(i) - my).^2);
    end
    numerator = (1/(N - 1)) * sumUpper;
    denominator = (sqrt((1/(N-1)) * sumLower)).^4;
    result = numerator / denominator;
end

function result = returnArray(data, c, N)
    r = zeros(1, N - 1);
    for i = 1:N - 1
        r(i) = log((data((i + 1), c)) / (data((i), c)));
    end
    result = r;
end

function result = calcMy(N, r)
    rSum = 0;
    for i = 1:N -1
        rSum = rSum + r(i);
    end
    result = rSum / (N-1);
end

function result = calcVol(N, r, my)
    sum = 0;
    for i = 1:N - 1
       sum = sum + (r(i) - my).^2;
    end
    result = sqrt(sum/(N - 2));
end

function res = EqWMA_arr(r, t, window)
    arr = zeros(1, t - window);
    for k = window:t
        sum = 0;
        for i = (k-(window-1)):k
            sum = sum + r(i).^2;
        end
        arr(k - (window - 1)) = sqrt ( sum / window ) * sqrt(52) * 100;
    end
    res = arr;
end

function res = EWMA_arr(r, lambda)
    n = size(r, 2);
    arr = zeros(1, n);
    arr(1) = sqrt(r(1).^2);
    for i = 2:(n)
        arr(i) = sqrt( lambda * arr(i-1).^2 + (1 - lambda) * r(i-1).^2 );
    end
    res = arr;
end
