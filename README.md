# Laboratorní cvičení BIN4 : Neuronové sítě

Cílem čtvrtého cvičení v předmětu BIN je seznámit se s neuronovými sítěmi a jejich výpočetní náročností.

## Info o NN
Projděte si teorii představenou na přednášce BIN. Dále se seznamte s prací s neuronovými sítěmi ve frameworku Tensorflow, která je popsaná v Jupyter noteboocích popisujících 

* MLP sítě: https://github.com/mrazekv/bin-lab-nn/blob/master/mlp.ipynb
* Konvoluční sítě: https://github.com/mrazekv/bin-lab-nn/blob/master/conv.ipynb

## Vlastní spuštění a testování 
Pro spuštění máte několik možností:

* __DOPORUČENO__: stáhnout notebook na [Google CoLab](https://colab.research.google.com/notebook), můžete otevřít projekt přímo z Githubu (nutné zadat `mrazekv` do cesty, pak vybrat `bin-lab-nn` a příslušný notebook). Potom kliknete do boxu s kódem (kde můžete dělat změny) a pomocí Shift+Enter spustit daný blok. Pozor, je nutné postupovat postupně a nepřeskakovat kernely.

* z repozitáře https://github.com/mrazekv/bin-lab-nn si stáhnout Python soubory. Tyto soubory můžete pustit u sebe (nutnost Python3 + Tensorflow + Keras a nejlépe aspoň základní GPU) či na serveru merlin (pozor, je potřeba spouštět příkazem `python3.8 <nazev_skriptu>`). 



## Úkoly 
Implementujte trénování a testování následujících sítí pro dataset MNIST a trénování pro __10 epoch__.

Implementujte následující sítě:
* MLP (fully-connected): 784-300-10
* MLP (fully-connected): 784-100-10
* MLP (fully-connected): 784-100-100-10
* MLP (fully-connected): 784-300-300-10
* Konvoluční 2 konvoluce + 120-84-10 fully connected
* Konvoluční 2 konvoluce + 120-10 fully connected
* Konvoluční 1 konvoluce + 120-84-10 fully connected (odstraňte druhou konvoluci a následnou pooling layer)
* Konvoluční 1 konvoluce + 120-10 fully connected (odstraňte druhou konvoluci a následnou pooling layer)

Vlastnosti sítí shrňte v __tabulce__, kde bude uveden:
* Typ sítě
* Dosažená validační přesnost (po __10 epochách__)
* Počet násobení v plně propojených vrstvách
* Počet násobení v konvolučních vrstvách
* Počet trénovacích parametrů

Vytvořte X-Y (scatter) __graf__, kde na ose X bude celkový počet násobení (~energie) a na ose Y bude výsledná přesnost. Diskutujte výsledky. Pro vykreslení můžete použít Excel, nebo rovnou můžete využít Python v Jupyter notebooku - doporučuji využít Matplotlib. Je nutné dodržet všechny vlastnosti grafů (barevné oddělení konvolučních / MLP sítí, popisky os a podobně).

Do výpočtu počtu násobení v plně propojených vrstvách je nutné zahrnout: počet vstupních neuronů a počet neuronů ve vrstvě. Pro výpočet počtu násobení v konvolučních vrstvách je nutné zahrnout: velikost vstupního obrázku, počet kanálů ve vstupním obrázku, velikost filtru, počet výstupních kanálů. 


## Co si připravit pro hodnocení
* Tabulka přesností + vzorec výpočtů)
* Graf (včetně dodržení všech náležitostí grafu)
* Shrnutí výsledků a závěr (ústně)
