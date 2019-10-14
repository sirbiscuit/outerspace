from openTSNE import TSNE, nearest_neighbors
from multiprocessing import cpu_count
import warnings
from numba import NumbaWarning
import math
import numpy as np
from .PlaygroundWidget import PlaygroundWidget
from .TransformationMethod import TransformationMethod
from .ui import create_widgets


class TSNETransformationMethod(TransformationMethod):
    def get_widgets(self):
        widgets = create_widgets([
            dict(name='_basic_params', type='heading',
                 text='Basic parameters'),
            dict(name='perplexity', type='int_slider',
                 description='Perplexity:', min=2, max=100, step=1, value=30),
            dict(name='learning_rate', type='int_slider',
                 description='Learning rate:', min=1, max=1000, step=1,
                 value=200),
            dict(name='initialization', type='toggle_buttons',
                 description='Initialization:', options=['PCA', 'random'],
                 value='random'),  # use "random" for better interactivity
            dict(name='negative_gradient_method', type='toggle_buttons',
                 description='Gradient method:',
                 options=['interpolation', 'barnes-hut'],
                 value='interpolation'),
            dict(name='final_momentum', type='float_slider',
                 description='Momentum:', min=0, max=1, step=0.05, value=0.8),

            dict(name='_nearest_neighbors', type='heading',
                 text='Nearest neighbors'),
            dict(name='neighbors_method', type='toggle_buttons',
                 description='Method:', options=['exact', 'approx'],
                 value='approx'),
            dict(name='n_jobs', type='int_slider', description='Num jobs:',
                 min=1, max=cpu_count(), step=1,
                 value=math.ceil(cpu_count()/4)),
            dict(name='exact_method_metric', type='dropdown',
                 description='Metric:',
                 options=nearest_neighbors.BallTree.VALID_METRICS,
                 value='euclidean'),
            dict(name='approx_method_metric', type='dropdown',
                 description='Metric:',
                 options=nearest_neighbors.NNDescent.VALID_METRICS,
                 value='euclidean'),

            dict(name='_early_exaggeration_phase', type='heading',
                 text='Early exaggeration phase'),
            dict(name='early_exaggeration_iter', type='int_slider',
                 description='Number of steps:', min=0, max=1000, step=10,
                 value=250),
            dict(name='early_exaggeration', type='int_slider',
                 description='Exagg factor:', min=0, max=100, step=1,
                 value=12),
            dict(name='initial_momentum', type='float_slider',
                 description='Momentum:', min=0, max=1, step=0.05, value=0.5),

            dict(name='_other_settings', type='heading', text='Other'),
            dict(name='random_state', type='int_slider',
                 description='Random state', min=0, max=65000, step=1,
                 value=2506),
        ])

        #
        # tag widgets as advanced
        #
        basic = ['_basic_params', 'perplexity', 'learning_rate',
                 '_early_exaggeration_phase', 'early_exaggeration_iter']
        for name, widget in widgets.items():
            widget.advanced = (name not in basic)

        #
        # additional behaviour
        #
        def on_change_neighbors(change):
            if change.name == 'value':
                if change.new == 'exact':
                    widgets.exact_method_metric.layout.display = 'flex'
                    widgets.approx_method_metric.layout.display = 'none'
                else:
                    widgets.exact_method_metric.layout.display = 'none'
                    widgets.approx_method_metric.layout.display = 'flex'
        widgets.neighbors_method.observe(on_change_neighbors, names='value')
        # change the value once to trigger change handler
        widgets.neighbors_method.value = 'exact'
        widgets.neighbors_method.value = 'approx'

        return widgets

    def get_current_params(self, widgets):
        metric_per_neighbor_method = {
            'exact': widgets.exact_method_metric.value,
            'approx': widgets.approx_method_metric.value
        }

        return dict(
            initialization=widgets.initialization.value.lower(),
            perplexity=widgets.perplexity.value,
            learning_rate=widgets.learning_rate.value,
            negative_gradient_method=widgets.negative_gradient_method.value.lower(),
            final_momentum=widgets.final_momentum.value,

            neighbors=widgets.neighbors_method.value.lower(),
            n_jobs=widgets.n_jobs.value,
            metric=metric_per_neighbor_method[widgets.neighbors_method.value.lower()],

            early_exaggeration=widgets.early_exaggeration.value,
            early_exaggeration_iter=widgets.early_exaggeration_iter.value,
            initial_momentum=widgets.initial_momentum.value,

            n_components=2,
            n_iter=10000000,  # TODO
            random_state=widgets.random_state.value)

    def run_transformation(self, X, y, transformation_params, callback):
        class CallbackAdapter:
            def __init__(self, callback, early_exaggeration_iter):
                self.callback = callback
                self.exaggeration_phase = early_exaggeration_iter > 0
                self.early_exaggeration_iter = early_exaggeration_iter

            def __call__(self, iteration, error, embedding):
                if not self.exaggeration_phase:
                    iteration += self.early_exaggeration_iter
                if self.exaggeration_phase and iteration == self.early_exaggeration_iter:
                    self.exaggeration_phase = False

                self.callback('embedding', iteration, dict(
                    embedding=embedding.view(np.ndarray),
                    error_metrics=dict(
                        kl_divergence=error
                    )
                ))

        callback_adapter = CallbackAdapter(
            callback, transformation_params['early_exaggeration_iter'])

        tsne = TSNE(**transformation_params,
                    min_grad_norm=0,  # never stop
                    callbacks=callback_adapter,
                    callbacks_every_iters=1)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=NumbaWarning)
            callback('start', 0, dict(error_metrics=[
                dict(name='kl_divergence', label='KL divergence:')
                ])
            )
            callback('status', 0, dict(message='Initializing TSNE'))
            tsne.fit(X)

        # umap = UMAP(callback=self.callback)
        # with warnings.catch_warnings():
        #     warnings.filterwarnings('ignore', category=NumbaWarning)
        #     umap.fit(self.X)


DEFAULT_TOOLTIPS = [
    ('index', '$index'),
    ('(x, y)', '($x, $y)'),
    ('label', '@label')
]


def tsne_playground(X, y, advanced_mode=False, autostart=True,
                    plot_every_iters=None, additional_columns=dict(),
                    tooltips=DEFAULT_TOOLTIPS, colors=None):
    """ Displays an interactive widget for manipulating parameters of TSNE.

    Parameters
    ----------
    X : np.ndarray of shape (n_examples, n_features)
        The feature variables.
    y : array_like of shape (n_examples,)
        The target variable used for coloring the data points.
    advanced_mode: bool
        If true, show more parameters to tune. The default value is False.
    autostart: bool
        If true, the optimization process is started immediately after
        executing this function. The default value is False.
    plot_every_iters : int or None
        The number of gradient descent steps between plotting the current
        embedding. Increase this number if the widget seems unresponsive, i.e.
        clicking a button or changing a slider does not work immediately. By
        specifying None, the number of steps depends on the size of X:
            max(len(X) // 2000, 1)
        The default value is None.
    additional_columns : dict, each value is an array of shape (n_examples,)
        Additional data to display in the tooltip. A column named key can be
        referenced in the tooltips parameter using @key.
    tooltips : array, str or None
        Describes the data that is shown when hovering over a data point. For
        more information, see `here <https://bokeh.pydata.org/en/latest/docs/
        user_guide/tools.html#basic-tooltips>`_. Besides of the default
        variables $index, $x, $sx and so on, the parameter @label (not $label)
        can be used to refer to the label column that was passed as y to this
        method. Disable tooltips by setting this parameter to None. The default
        value is:
            [
                ('index', '$index'),
                ('(x, y)', '($x, $y)'),
                ('label', '@label')
            ]
    colors : array, str or None
        Specifies the fill color of the circles in the scatter plot. If array,
        the labels are randomly assigned to the specified colors. If str, the
        colors are fetched from a column provided in additional_columns. If
        None, the colors are chosen from palettes Category10, Category20 or
        Viridis(n) depending on the number of unique labels. The default value
        is None.

    Returns
    -------
    PlaygroundWidget
        The actual interaction widget that can be rendered in a Jupyter
        notebook.

    Examples
    --------
    Run t-SNE on the digits data set:

        from outerspace import tsne_playground
        from sklearn.datasets import load_digits
        digits = load_digits()
        X, y = digits.data, digits.target
        tsne_playground(X, y)

    Show the actual digit images in a tooltip:

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
    """
    transformer_widget = PlaygroundWidget(
        TSNETransformationMethod(),
        X, y, advanced_mode, autostart,
        plot_every_iters,
        additional_columns, tooltips,
        colors)

    return transformer_widget
