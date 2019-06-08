outerspace
==========

Work in progress

Installation
------------

```bash
pip install outerspace
```

Usage
-----

```python
from outerspace import tsne_playground
import mnist

# load data
X = mnist.train_images()
X = X.reshape(len(X), -1) # flatten each training sample
y = mnist.train_labels()

tsne_playground(X, y)
```

```python
from outerspace import tsne_playground, nparr_to_html_tag
additional_columns = dict(image=[nparr_to_html_tag(sample) for sample in X_sample])
tooltips = tooltips = [
            ('index', '$index'),
            ('(x, y)', '($x, $y)'),
            ('label', '@label'),
            ('image', '@image{safe}')
        ]
tsne_playground(X, y, additional_columns=additional_columns, tooltips=tooltips)
```
