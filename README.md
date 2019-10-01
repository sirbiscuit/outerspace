outerspace
==========

An interactive widget for Jupyter notebooks to explore the parameters of t-SNE
or UMAP.

![outerspace demo](demo.gif)

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
jupyter labextension install jupyterlab_bokeh
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
