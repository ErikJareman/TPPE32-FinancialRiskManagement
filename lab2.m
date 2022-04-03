data = readmatrix('timeSeries.xlsx', 'Sheet', 'Problem 1 and 2', 'Range', 'C3:Q1649');
nStock = size(data, 2);
nDate = size(data, 1);
R = zeros(nDate-1, nStock);
for i = 1:nStock
    for j = 1:(nDate-1)
        R(j, i) = (data(j + 1, i) - data(j, i)) / data(j, i);
    end
end
Rp = sum(R, 2) ./ nStock;


% Uppgift 1a
% VaR (varians-kovarians)
C = cov(R(1:502, :));
VaR95 = VaR(C, 10000000, 0.95);
VaR975 = VaR(C, 10000000, 0.975);
VaR99 = VaR(C, 10000000, 0.99);
% disp("Uppgift 1a:");
% VaR95
% VaR975
% VaR99


% Uppgift 1b
% Relativt VaR (logaritmiska avkastningar, EWMA)
Rlogp = flipud(log(1 + Rp));
volLogEWMA = zeros(nDate-501, 1);
for i = 1:(nDate-501)
    initLogVol = sqrt(Rlogp(i).^2);
    tmp = EWMA(Rlogp(i:i+499,:), 0.94, initLogVol);
    volLogEWMA(i) = tmp(end);
end
logRelVaR95 = (1 - exp(- norminv(0.95) * volLogEWMA));
logRelVaR99 = (1 - exp(- norminv(0.99) * volLogEWMA));
% disp("Uppgift 1b:");
% logRelVaR95
% logRelVaR99
% figure(1);
% subplot(2,1,1)
% plot(logRelVaR95)
% title("logRelVar95");
% 
% subplot(2,1,2)
% plot(logRelVaR99)
% title("logRelVar99");


% Uppgift 1c
% Relativt VaR (historisk simulering)
histSimRelVaR95 = zeros(nDate-501, 1);
histSimRelVaR99 = zeros(nDate-501, 1);
for j = 1:(nDate-501)
    deltaVp = zeros(500, 1);
    for i = 1:500
        scenAvk = data(j + i - 1, :)./data(j + i, :) - 1;
        scenAvk(isinf(scenAvk) | isnan(scenAvk)) = 0;
        deltaVp(i) = (10000000/nStock) .* (data(j+i-1, :)./data(1, :)) * scenAvk';
    end
    deltaVp = sort(deltaVp, 'ascend');
    histSimRelVaR95(j) = abs(deltaVp(25) / 10000000);
    histSimRelVaR99(j) = abs(deltaVp(5) / 10000000);
end
histSimRelVaR95 = flipud(histSimRelVaR95); % relVar på samma håll...
histSimRelVaR99 = flipud(histSimRelVaR99); % ...för jämförelse
% Expected shortfall för T+1
Rp500 = sum(Rp(1:500, 1), 2);
Rp500 = sort(Rp500, 'ascend');
ES95 = sum(Rp500(1:25, 1)) * 10000000 / 25;
% disp("Uppgift 1c:"); 
% histSimRelVaR95
% histSimRelVaR99
% ES95
% figure(2);
% subplot(2,1,1)
% plot(histSimRelVaR95)
% title("histSimRelVaR95");
% ylim([0 0.04])
% 
% subplot(2,1,2)
% plot(histSimRelVaR99)
% title("histSimRelVaR99");


% Uppgift 1d
RpFlip = flipud(Rp);
avgR20 = sum(RpFlip(1:20, 1), 1) / 20;
initVol = sqrt( (sum((RpFlip(1:20, 1) - avgR20).^2)) / 19);
volEWMA = EWMA(RpFlip(:, 1), 0.94, initVol);
HullWhiteVp = zeros(nDate-2, 1);
HullWhiteVp(:, 1) = 10000000 * RpFlip(1:end-1,1) .* volEWMA(1:end-1,1) ./ volEWMA(2:end,1);

HullWhiteRelVaR95 = zeros(nDate-501, 1);
HullWhiteRelVaR99 = zeros(nDate-501, 1);
for i = 1:(nDate-501)
    VpWindow = HullWhiteVp(i:i+499, 1);
    VpWindow = sort(VpWindow, 'ascend');
    HullWhiteRelVaR95(i) = abs(VpWindow(25) / 10000000);
    HullWhiteRelVaR99(i) = abs(VpWindow(5) / 10000000);
end
% disp("Uppgift 1d:");
% HullWhiteRelVaR95
% HullWhiteRelVaR99
% figure(3);
% subplot(2,1,1)
%plot(HullWhiteRelVaR95)
% title("HullWhiteRelVaR95");
% 
% subplot(2,1,2)
% plot(HullWhiteRelVaR99)
% title("HullWhiteRelVaR99");


% Uppgift 1e
logHyp95FRT = FailureRateTest(Rp, flipud(logRelVaR95), 0.05, 0.95);
logHyp99FRT = FailureRateTest(Rp, flipud(logRelVaR99), 0.01, 0.99);
histSimHyp95FRT = FailureRateTest(Rp, flipud(histSimRelVaR95), 0.05, 0.95);
histSimHyp99FRT = FailureRateTest(Rp, flipud(histSimRelVaR99), 0.01, 0.99);
HullWhiteHyp95FRT = FailureRateTest(Rp, flipud(HullWhiteRelVaR95), 0.05, 0.95);
HullWhiteHyp99FRT = FailureRateTest(Rp, flipud(HullWhiteRelVaR99), 0.01, 0.99);
% disp("true=reject")
% disp("logHyp95FRT " + logHyp95FRT)
% disp("logHyp99FRT " + logHyp99FRT)
% disp("histSimHyp95FRT " + histSimHyp95FRT)
% disp("histSimHyp99FRT " + histSimHyp99FRT)
% disp("HullWhiteHyp95FRT " + HullWhiteHyp95FRT)
% disp("HullWhiteHyp99FRT " + HullWhiteHyp99FRT)


% Uppgift 1f
logHyp95Ch = SerBer(Rp, flipud(logRelVaR95), 0.05);
logHyp99Ch = SerBer(Rp, flipud(logRelVaR99), 0.01);
histSimHyp95Ch = SerBer(Rp, flipud(histSimRelVaR95), 0.05);
histSimHyp99Ch = SerBer(Rp, flipud(histSimRelVaR99), 0.01);
HullWhiteHyp95Ch = SerBer(Rp, flipud(HullWhiteRelVaR95), 0.05);
HullWhiteHyp99Ch = SerBer(Rp, flipud(HullWhiteRelVaR99), 0.01);
% disp("true=reject");
% disp("logHyp95FCh " + logHyp95Ch)
% disp("logHyp99Ch " + logHyp99Ch)
% disp("histSimHyp95Ch " + histSimHyp95Ch)
% disp("histSimHyp99Ch " + histSimHyp99Ch)
% disp("HullWhiteHyp95Ch " + HullWhiteHyp95Ch)
% disp("HullWhiteHyp99Ch " + HullWhiteHyp99Ch)

% Uppgift 2a
RpSort = sort(Rp, 'ascend');
u = RpSort(82, 1);
y = RpSort(1:81) - u;
f = @(xy) -sum(log(1/xy(1))*(1+xy(2)*(y)/xy(1)).^(-1/xy(2)-1));
%     [beta; xi]
xy0 = [0.1; 0.7];
ansMLE = fmincon(f, xy0);
testVaR99 = u+(ansMLE(1)/ansMLE(2))*(((1646/81)*0.01)^(-ansMLE(2))-1);


% Uppgift 2b
% - Hitta den 5-års period med störst genomsnittlig volatilitet.
% - EWMA halvbra ty viktat -> använder stickprovsskattning.
% - Vi beräknar variansen utifrån 260v(5år) fönster och väljer
%   den största stickprovsskattningen som slutdatum för den mest
%   volatila perioden.
% Logaritmerade utdelningar:
U = zeros(nDate-1, nStock);
dataFlip = flipud(data);
for i = 1:nStock
    for j = 1:(nDate-1)
        U(j, i) = log(dataFlip(j + 1, i) / dataFlip(j, i));
    end
end
Up = flipud(sum(U, 2) ./ nStock);
UpAvg = mean(Up);
% Väntevärdesriktig stickprovsskattning av varians, 260v fönster
volStickprov260 = zeros(nDate-261, 1);
for i = 1:(nDate-261)
    tmp = 0;
    for j = 1:260
        tmp = tmp + ((Up(i+j-1, 1) - UpAvg)^2);
    end
    volStickprov260(i) = sqrt(tmp / 259);
end
[volMax, index] = max(volStickprov260);
highVolIndexInRp = [1646-index-259, 1646-index];
% Intervall (5 år): 2000-mars-03 --> 2005-feb-25

% Parameterestimering enl. uppgift 2a, nytt Rp:
RpHighVol = Rp(highVolIndexInRp(1):highVolIndexInRp(2), 1);
RpHighVolSort = sort(RpHighVol, 'ascend');
uHighVol = RpHighVolSort(13, 1); %0.05*260=13
yHighVol = RpHighVolSort(1:12, 1) - uHighVol;
fHighVol = @(xy) -sum(log(1/xy(1))*(1+xy(2)*(yHighVol)/xy(1)).^(-1/xy(2)-1));
ansHighVolMLE = fmincon(fHighVol, xy0); % samma startlösning
figure(100);
subplot(2,1,1);
plot(1:length(y), (1/ansMLE(1))*(1+ansMLE(2)*(y(1:end)/ansMLE(1)).^(-1/ansMLE(2)-1)));
subplot(2,1,2);
plot(1:length(yHighVol), (1/ansHighVolMLE(1))*(1+ansHighVolMLE(2)*(yHighVol)/ansHighVolMLE(1)).^(-1/(ansHighVolMLE(2)-1)));



% Uppgift 3a
SPX = readmatrix('timeSeries.xlsx', 'Sheet', 'Problem 3', 'Range', 'C4:C3429');
VIX = readmatrix('timeSeries.xlsx', 'Sheet', 'Problem 3', 'Range', 'D4:D3429');
USD = readmatrix('timeSeries.xlsx', 'Sheet', 'Problem 3', 'Range', 'E4:E3429');
Tmar = days252bus('2022-01-10', '2022-03-22')/252;
Tapr = days252bus('2022-01-10', '2022-04-22')/252;
marCall = BSM(SPX(1), 4700, 15.77, Tmar, 0, convertLibor3m(USD(1)/100), 1);
marPut = BSM(SPX(1), 4600, 18.28, Tmar, 0, convertLibor3m(USD(1)/100), 0);
aprCall = BSM(SPX(1), 4750, 16.245, Tapr, 0, convertLibor3m(USD(1)/100), 1);
% greker
gMarCall = [delta(SPX(1), 4700, 15.77/100, Tmar, 0, convertLibor3m(USD(1)/100), 0.05, 1)*SPX(1)
            vega(SPX(1), 4700, 15.77/100, Tmar, 0, convertLibor3m(USD(1)/100), 0.05)
            rho(SPX(1), 4700, 15.77/100, Tmar, 0, convertLibor3m(USD(1)/100), 1)];
gMarPut = [delta(SPX(1), 4600, 18.28/100, Tmar, 0, convertLibor3m(USD(1)/100), 0.05, 0)*SPX(1)
            vega(SPX(1), 4600, 18.28/100, Tmar, 0, convertLibor3m(USD(1)/100), 0.05)
            rho(SPX(1), 4600, 18.28/100, Tmar, 0, convertLibor3m(USD(1)/100), 0)];
gAprCall = [delta(SPX(1), 4750, 16.245/100, Tapr, 0, convertLibor3m(USD(1)/100), 0.05, 1)*SPX(1)
            vega(SPX(1), 4750, 16.245/100, Tapr, 0, convertLibor3m(USD(1)/100), 0.05)
            rho(SPX(1), 4750, 16.245/100, Tapr, 0, convertLibor3m(USD(1)/100), 1)];
rSP = zeros(size(SPX, 1), 1);
deltaVol = zeros(size(SPX, 1), 1);
deltaR = zeros(size(SPX, 1), 1);
for i = 1:(size(SPX, 1)-1)
    rSP(i) = log(SPX(i) / SPX(i+1));
    deltaVol(i) = VIX(i)/100 - VIX(i+1)/100;
    deltaR(i) = convertLibor3m(USD(i)/100) - convertLibor3m(USD(i+1)/100);
end
lambda = [rSP deltaVol deltaR];
C = cov(lambda);
h = [10000 10000 20000]';
G = [gMarCall'; gMarPut'; gAprCall'];
var = h' * G' * C * G * h;
VaRrfm = norminv(0.99) * sqrt(var);


% Uppgift 3b
% Marginellt bidrag till VaR99, resp. option:
optMB = norminv(0.99)*sqrt(1)*( (G'*C*G*h) / sqrt(var) );

% Marginellt bidrag till VaR99, resp. riskfaktor:
hf = G*h; % H = 0?
volRF = (C*hf) / sqrt(var);
VaR99RF = norminv(0.99)*volRF;


% Funktioner
function rc = convertLibor3m(rs)
    T = 3/12;
    d = 1 / (1 + rs*T);
    rc = -log(d) / T;
end

function res = delta(S, K, vol, T, t, r, q, call)
    d = calcD(S, K, vol, T, t, r);
    if (call == 1) 
        res = exp(-q*(T-t))*normcdf(d(1));
    else
        res = -exp(-q*(T-t))*normcdf(-d(1));
    end
end

function res = vega(S, K, vol, T, t, r, q)
    d = calcD(S, K, vol, T, t, r);
    res = S*exp(-q*(T-t))*normcdf(d(1))*sqrt(T-t);
end

function res = rho(S, K, vol, T, t, r, call)
    d = calcD(S, K, vol, T, t, r);
    if (call == 1)
        res = K*(T-t)*exp(-r*(T-t))*normcdf(d(2));
    else
        res = -K*(T-t)*exp(-r*(T-t))*normcdf(-d(2));
    end
end

function res = BSM(S, K, vol, T, t, r, call)
    d = calcD(S, K, vol, T, t, r);
    if (call == 1)
        res = S*normcdf(d(1)) - K*exp(-r*(T-t))*normcdf(d(2));
    else
        res = K*exp(-r*(T-t))*normcdf(-d(2)) - S*normcdf(-d(1));
    end
end

function res = calcD(S, K, vol, T, t, r)
    d1 = (log(S/K)+(r+((vol)^2)/2)*(T-t)) / (vol*sqrt(T-t));
    d2 = d1 - (vol*sqrt(T-t));
    res = [d1, d2];
end

function res = FailureRateTest(Rp, relVaR, alpha, c)
    n = size(Rp, 1);
    ml = norminv(alpha/2);
    mu = norminv(1-(alpha/2));
    count = 0;
    for j = 1:(n-501)
        count = count + (Rp(j) < - relVaR(j));
    end
    Z = (count-(n-501)*(1-c))/(sqrt((n-501)*(1-c)*(1-(1-c))));
    res = (Z < ml || Z > mu);
end

function res = SerBer(r, VaR, alpha)
    n = size(r, 1);
    z = chi2inv((1-alpha), 1);
    nXY = zeros(2, 2);
    prev = 0;
    for j = 1:(n-501)
       curr = (r(j) < - VaR(j));
       nXY(prev+1, curr+1) = nXY(prev+1, curr+1) + 1;
       prev = curr;
    end
    res = VL(nXY(1,1), nXY(1,2), nXY(2,1), nXY(2,2)) > z;
end

function res = VL(n00, n01, n10, n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11);
    pi01 = n01 / (n00 + n01);
    pi11 = n11 / (n10 + n11);
    term1=-2*log(((1-pi)^(n00+n10)) * (pi^(n01+n11)));
    term2=2*log((1-pi01)^(n00)*pi01^(n01)*(1-pi11)^(n10)*pi11^(n11));
    res = term1 + term2;
end

function res = EWMA(r, lambda, initVol)
    n = size(r, 1);
    arr = zeros(n, 1);
    arr(1) = initVol;
    for i = 2:(n)
        arr(i) = sqrt( lambda * arr(i-1).^2 + (1 - lambda) * r(i-1).^2 );
    end
    res = arr;
end

function res = VaR(C, Vp, confidence)
    nStock = size(C, 2);
    w = (1 / nStock) .* ones(nStock, 1);
    sigma = sqrt(w' * C * w);
    
    res = norminv(confidence) * sigma * Vp;
end