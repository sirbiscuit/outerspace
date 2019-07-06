outerspace
==========

An interactive widget for Jupyter notebooks to explore the parameters of t-SNE.

<img src="https://github.com/sirbiscuit/outerspace/raw/master/demo.gif" width="716" height="428"/>

Installation
------------

```bash
pip install outerspace
jupyter nbextension enable --py widgetsnbextension
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
