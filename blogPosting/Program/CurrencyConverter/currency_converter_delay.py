from currency_converter import CurrencyConverter

cc = CurrencyConverter('https://www.ecb.europa.eu/stats/eurofxref/eurofxref.zip')
fcc = round(cc.convert(1, 'USD', 'KRW'), 2)
print(fcc)
