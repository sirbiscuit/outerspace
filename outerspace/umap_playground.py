from umap import UMAP
from umap.umap_ import find_ab_params
from umap.distances import named_distances
import warnings
from numba import NumbaWarning
from .PlaygroundWidget import PlaygroundWidget
from .TransformationMethod import TransformationMethod
from .ui import create_widgets


class UMAPTransformationMethod(TransformationMethod):
    def get_widgets(self):
        default_spread = 1
        default_min_dist = 0.1

        default_a, default_b = find_ab_params(default_spread, default_min_dist)

        metrics = sorted(named_distances.keys())
        target_metrics = sorted(metrics + ['categorical'])

        widgets = create_widgets([
            dict(name='_basic_params', type='heading',
                 text='Basic parameters'),
            dict(name='n_neighbors', type='int_slider',
                 description='# Neighbors:', min=1, max=100, step=1, value=15),
            dict(name='parameter_control', type='toggle_buttons',
                 description='Parameter control:',
                 options=['basic', 'advanced'],
                 value='basic'),
            dict(name='min_dist', type='float_slider',
                 description='Minimum distance:', min=0.01, max=100, step=0.1,
                 value=default_min_dist),
            dict(name='spread', type='float_slider',
                 description='Spread:', min=0.1, max=100, step=0.1,
                 value=default_spread),
            dict(name='a', type='float_slider',
                 description='Parameter a:', min=0.1, max=100, step=0.1,
                 value=default_a),
            dict(name='b', type='float_slider',
                 description='Parameter b:', min=0.1, max=100, step=0.1,
                 value=default_b),
            dict(name='init', type='toggle_buttons',
                 description='Initialization:', options=['spectral', 'random'],
                 value='spectral'),
            dict(name='learning_rate', type='float_slider',
                 description='Learning rate:', min=1, max=1000, step=1,
                 value=1),
            dict(name='repulsion_strength', type='float_slider',
                 description='Repulsion_strength:', min=0.1, max=100, step=0.1,
                 value=1),
            dict(name='negative_sample_rate', type='int_slider',
                 description='Negative sample rate:', min=1, max=100, step=1,
                 value=5),

            dict(name='_simplical_set_construction', type='heading',
                 text='Simplical set construction'),
            dict(name='metric', type='dropdown',
                 description='Metric:', options=metrics,
                 value='euclidean'),
            dict(name='angular_rp_forest', type='toggle_buttons',
                 description='Angular random projection forest:',
                 options=['yes', 'no'],
                 value='no'),
            dict(name='set_op_mix_ratio', type='float_slider',
                 description='Set operation mix ratio:', min=0, max=1,
                 step=0.05,
                 value=1),
            dict(name='local_connectivity', type='int_slider',
                 description='Local connectivity:', min=1, max=100, step=1,
                 value=1),
            dict(name='target_n_neighbors', type='int_slider',
                 description='# Neighbors (target):', min=-1, max=100, step=1,
                 value=-1),
            dict(name='target_metric', type='dropdown',
                 description='Target metric:', options=target_metrics,
                 value='categorical'),
            dict(name='target_weight', type='float_slider',
                 description='Target weight', min=0, max=1, step=0.05,
                 value=1),

            dict(name='_other_settings', type='heading',
                 text='Other settings'),
            dict(name='random_state', type='int_slider',
                 description='Random state:', min=0, max=65000, step=1,
                 value=2506),

            # # only for transform / fit_transform:
            # dict(name='transform_seed', type='int_slider',
            #      description='Transform seed:', min=0, max=65000, step=1,
            #      value=42),
            # dict(name='transform_queue_size', type='float_slider',
            #      description='Repulsion_strength:', min=0.1, max=100, step=0.1,
            #      value=4),
        ])

        #
        # tag widgets as advanced
        #
        basic = ['_basic_params', 'n_neighbors', 'min_dist']
        for name, widget in widgets.items():
            widget.advanced = (name not in basic)

        #
        # additional behaviour
        #
        def on_change_parameter_control(change):
            if change.name == 'value':
                if change.new == 'basic':
                    widgets.min_dist.layout.display = 'flex'
                    widgets.spread.layout.display = 'flex'
                    widgets.a.layout.display = 'none'
                    widgets.b.layout.display = 'none'
                else:
                    widgets.min_dist.layout.display = 'none'
                    widgets.spread.layout.display = 'none'
                    widgets.a.layout.display = 'flex'
                    widgets.b.layout.display = 'flex'
        widgets.parameter_control.observe(on_change_parameter_control,
                                          names='value')
        # change the value once to trigger change handler
        widgets.parameter_control.value = 'advanced'
        widgets.parameter_control.value = 'basic'

        return widgets

    def get_current_params(self, widgets):
        a = None
        b = None
        if widgets.parameter_control.value.lower() == 'advanced':
            a = widgets.a.value
            b = widgets.b.value

        return dict(
            n_neighbors=widgets.n_neighbors.value,
            min_dist=widgets.min_dist.value,
            spread=widgets.spread.value,
            a=a,
            b=b,
            init=widgets.init.value,
            learning_rate=widgets.learning_rate.value,
            repulsion_strength=widgets.repulsion_strength.value,
            negative_sample_rate=widgets.negative_sample_rate.value,

            metric=widgets.metric.value.lower(),
            angular_rp_forest=widgets.angular_rp_forest.value.lower() == 'yes',
            set_op_mix_ratio=widgets.set_op_mix_ratio.value,
            local_connectivity=widgets.local_connectivity.value,
            target_n_neighbors=widgets.target_n_neighbors.value,
            target_metric=widgets.target_metric.value.lower(),
            target_weight=widgets.target_weight.value,

            n_components=2,
            random_state=widgets.random_state.value,

            n_epochs=10000000,  # TODO
            verbose=False
            )

    def run_transformation(self, X, y, transformation_params, callback):
        class CallbackAdapter:
            def __init__(self, callback):
                self.callback = callback

            def __call__(self, iteration, embedding):
                self.callback('embedding', iteration, dict(
                    embedding=embedding
                ))

        callback_adapter = CallbackAdapter(callback)

        umap = UMAP(callback=callback_adapter, **transformation_params)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=NumbaWarning)
            try:
                callback('start', 0, None)
                callback('status', 0, dict(message='Initializing UMAP'))
                umap.fit(X, y)
            except Exception as e:
                callback('error', 0, dict(message=str(e)))


DEFAULT_TOOLTIPS = [
    ('index', '$index'),
    ('(x, y)', '($x, $y)'),
    ('label', '@label')
]


def umap_playground(X, y, advanced_mode=False, autostart=True,
                    plot_every_iters=None, additional_columns=dict(),
                    tooltips=DEFAULT_TOOLTIPS, colors=None):
    """ Displays an interactive widget for manipulating parameters of UMAP.

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
        UMAPTransformationMethod(),
        X, y, advanced_mode, autostart,
        plot_every_iters,
        additional_columns, tooltips,
        colors)

    return transformer_widget
