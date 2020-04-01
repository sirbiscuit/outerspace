outerspace
==========

An interactive widget for Jupyter notebooks to explore the parameters of t-SNE.

<img src="https://github.com/sirbiscuit/outerspace/raw/master/demo.gif">


Installation
------------

```bash
pip install outerspace
```

Additionally, if you use `jupyter notebook`:
```bash
jupyter nbextension enable --py widgetsnbextension
```

... or for `jupyter lab`:
```bash
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install @bokeh/jupyter_bokeh
```

Usage
-----

Run t-SNE on the digits data set (see result in the image above):

```python
from outerspace import tsne_playground
from sklearn.datasets import load_digits
digits = load_digits()
X, y = digits.data, digits.target
tsne_playground(X, y)
```

Show the actual digit images in a tooltip:

<img align="right" width="300px" src="https://github.com/sirbiscuit/outerspace/raw/master/tooltip_image.png">

```python
from outerspace import tsne_playground, array2d_to_html_img
from sklearn.datasets import load_digits

digits = load_digits()
X, y, images = digits.data, digits.target, digits.images

images = 256 - images * 16      # convert range and invert
images = images.clip(0, 255)    # clip values at 255
images = images.astype('uint8') # convert to uint8
images = [array2d_to_html_img(image, resize=(32,32))
          for image in images]  # convert to HTML images

tsne_playground(X, y,
                additional_columns=dict(images=images),
                tooltips='@images{safe}') # safe = do not escape HTML
```

Further examples
----------------

Evaluating the chemical space of a set of molecules (with molecule images as tooltip):

<img align="right" width="300px" src="https://github.com/sirbiscuit/outerspace/raw/master/tooltip_molecules.png">

```python
from outerspace import tsne_playground, pil_to_html_img
from rdkit.Chem import SDMolSupplier, Draw, AllChem
import requests
import numpy as np

url = 'https://raw.githubusercontent.com/rdkit/rdkit/Release_2020_03/Docs/Book/data/solubility.test.sdf'
response = requests.get(url)

supplier = SDMolSupplier()
supplier.SetData(response.text)
ms = [m for m in supplier]

X = np.array([list(AllChem.GetMACCSKeysFingerprint(m)) for m in ms])
y = [m.GetProp('SOL_classification') for m in ms]
images = [pil_to_html_img(Draw.MolToImage(m, size=(150, 150))) for m in ms]

tsne_playground(X, y, 
                additional_columns=dict(images=images),
                tooltips='@images{safe}')
```