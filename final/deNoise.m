function tranlated = deNoise(series)
  series = detrend(series); %y = detrend(x) removes the best straight-line fit from vector x and returns it in y.
  m = mean(series);
  sigma = sqrt(mean(series.^2) - m.^2); %standard deviation
  tranlated = (series - m)./sigma;
end