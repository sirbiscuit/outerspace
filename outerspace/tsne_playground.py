from openTSNE import nearest_neighbors
from IPython import display
import math
import ipywidgets as widgets
from multiprocessing import cpu_count
from bokeh.transform import factor_cmap
from bokeh.palettes import Category10_10, Category20_20, viridis
from bokeh.models import ColumnDataSource, Label
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
from .EmbeddingTask import EmbeddingTask

output_notebook(hide_banner=True)

DEFAULT_TOOLTIPS = [
    ('index', '$index'),
    ('(x, y)', '($x, $y)'),
    ('label', '@label')
]


def tsne_playground(X, y, advanced_mode=False, autostart=True,
                    steps_between_plotting=None, additional_columns=dict(),
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
    steps_between_plotting : int or None
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

    if steps_between_plotting is None:
        # TODO: measure unresponsiveness caused by plotting
        steps_between_plotting = max(len(X) // 2000, 1)

    unique_values = set(y)
    factors = [str(u) for u in unique_values]

    if colors is None:
        if len(unique_values) <= 10:
            color = factor_cmap('label', Category10_10, factors)
        elif len(unique_values) <= 20:
            color = factor_cmap('label', Category20_20, factors)
        else:
            color = factor_cmap('label', viridis(len(unique_values)), factors)
    elif type(colors) == list:
        if len(colors) < len(unique_values):
            raise ValueError('not enough colors specified')
        color = factor_cmap('label', colors, factors)
    elif type(colors) == str:
        if colors not in additional_columns.keys():
            raise ValueError(f'the key {colors} does not exist in additional_columns')
        color = colors

    source = ColumnDataSource(data=dict(
        x=X[:, 0],
        y=X[:, 1],
        label=[str(e) for e in y],
        **additional_columns))

    # create plot
    tools = ['pan', 'box_zoom', 'wheel_zoom', 'save', 'reset']
    if tooltips:
        tools.append('hover')
    p = figure(
        output_backend="webgl",
        plot_height=300,
        plot_width=300,
        sizing_mode='scale_width',
        tools=tools,
        active_drag='pan',
        active_scroll='wheel_zoom',
        toolbar_location="below",
        tooltips=tooltips
      )
    p.toolbar.logo = None  # remove bokeh logo

    scatter = p.scatter(
        source=source,
        x='x',
        y='y',
        color=color,
        legend=False,
        line_width=0,
        alpha=0.8,
        size=5)
    scatter.visible = False

    # status message on top of graph
    status = Label(x=20, y=20, x_units='screen', y_units='screen',
                   text='⏳', render_mode='css', text_font_size='40px')
    status.visible = False
    p.add_layout(status)

    # render all widgets
    selection_style = {'button_width': '100px'}
    label_style = {'description_width': '100px'}
    advanced_layout = {'display': 'flex' if advanced_mode else 'none'}

    out = widgets.Output(
        layout={'flex': '1', 'max_width': '90vh'})

    heading1 = widgets.HTML(
        value="<h3>Basic parameters</h3>")
    perplexity_slider = widgets.IntSlider(
        value=30,
        min=2,
        max=100,
        step=1,
        description='Perplexity:',
        style=label_style,
        continuous_update=False)
    learning_rate_slider = widgets.IntSlider(
        value=200,
        min=1,
        max=1000,
        step=1,
        description='Learning rate:',
        style=label_style,
        continuous_update=False)  # epsilon
    initialization_select = widgets.ToggleButtons(
        value='PCA',
        description='Initialization:',
        style=dict(**selection_style, **label_style),
        layout=advanced_layout,
        options=['PCA', 'random'])
    negative_gradient_method_select = widgets.ToggleButtons(
        value='interpolation',
        options=['interpolation', 'barnes-hut'],
        style=dict(**selection_style, **label_style),
        layout=advanced_layout,
        description='Gradient method:')
    final_momentum_slider = widgets.FloatSlider(
        value=0.8,
        min=0,
        max=1,
        step=0.05,
        description='Momentum:',
        style=label_style,
        layout=advanced_layout,
        continuous_update=False)

    heading2 = widgets.HTML(
        value="<h3>Nearest neighbors</h3>",
        layout=advanced_layout)
    neighbors_select = widgets.ToggleButtons(
        value='approx',  # exakt = sklearn ball_tree, approx = pynndescent
        options=['exact', 'approx'],
        style=dict(**selection_style, **label_style),
        layout=advanced_layout,
        description='Method:')
    n_jobs_slider = widgets.IntSlider(
        value=math.ceil(cpu_count()/4),
        min=1,
        max=cpu_count(),
        step=1,
        description='Num jobs:',
        style=label_style,
        layout=advanced_layout,
        continuous_update=False)
    exact_method_metric = widgets.Dropdown(
        value='euclidean',
        options=nearest_neighbors.BallTree.VALID_METRICS,
        style=label_style,
        layout=dict(list(advanced_layout.items())+list(dict(display='none').items())),
        description='Metric:'
    )
    approx_method_metric = widgets.Dropdown(
        value='euclidean',
        options=nearest_neighbors.NNDescent.VALID_METRICS.keys(),
        style=label_style,
        layout=dict(list(advanced_layout.items())+list(dict(display='flex').items())),
        description='Metric:'
    )

    heading3 = widgets.HTML(
        value="<h3>Early exaggeration phase</h3>")
    early_exaggeration_iter_slider = widgets.IntSlider(
        value=250,
        min=0,
        max=1000,
        step=10,
        description='Number of steps:',
        style=label_style,
        continuous_update=False)
    early_exaggeration_slider = widgets.IntSlider(
        value=12,
        min=0,
        max=100,
        step=1,
        description='Exagg factor:',
        style=label_style,
        layout=advanced_layout,
        continuous_update=False)
    initial_momentum_slider = widgets.FloatSlider(
        value=0.5,
        min=0,
        max=1,
        step=0.05,
        style=label_style,
        layout=advanced_layout,
        description='Momentum:',
        continuous_update=False)

    heading4 = widgets.HTML(
        value="<h3>Other</h3>",
        layout=advanced_layout)
    random_state_slider = widgets.IntSlider(
        value=2506,
        min=0,
        max=65000,
        step=1,
        description='Random state:',
        style=label_style,
        layout=advanced_layout,
        continuous_update=False)

    heading5 = widgets.HTML(value="<h3>Stats</h3>")
    speed_text = widgets.HTML(description='Speed:', style=label_style)
    iteration_text = widgets.HTML(description='Iteration:', style=label_style)
    play_pause_button = widgets.Button(icon='play', layout=widgets.Layout(width='40px', height='40px'))
    stop_button = widgets.Button(icon='stop', layout=widgets.Layout(width='40px', height='40px'))
    player_controls = widgets.HBox([play_pause_button, stop_button])

    log = widgets.Output()

    control_collection = [
        heading1,
        perplexity_slider,
        learning_rate_slider,
        initialization_select,
        negative_gradient_method_select,
        final_momentum_slider,

        heading2,
        neighbors_select,
        n_jobs_slider,
        exact_method_metric,
        approx_method_metric,

        heading3,
        early_exaggeration_iter_slider,
        early_exaggeration_slider,
        initial_momentum_slider,

        heading4,
        random_state_slider,

        heading5,
        speed_text,
        iteration_text,
        player_controls,

        log]
    controls = widgets.VBox(control_collection, layout={'flex': '0 0 350px'})
    hbox = widgets.HBox([out, controls], layout={'width': '100%'})
    display.display(hbox)

    with out:
        handle = show(p, notebook_handle=True)

    #
    # HELPER METHODS
    #
    def get_current_params():
        metric_per_neighbor_method = {
            'exact': exact_method_metric.value,
            'approx': approx_method_metric.value
        }

        return dict(
            initialization=initialization_select.value.lower(),
            perplexity=perplexity_slider.value,
            learning_rate=learning_rate_slider.value,
            negative_gradient_method=negative_gradient_method_select.value.lower(),
            final_momentum=final_momentum_slider.value,

            neighbors=neighbors_select.value.lower(),
            n_jobs=n_jobs_slider.value,
            metric=metric_per_neighbor_method[neighbors_select.value.lower()],

            early_exaggeration=early_exaggeration_slider.value,
            early_exaggeration_iter=early_exaggeration_iter_slider.value,
            initial_momentum=initial_momentum_slider.value,

            n_components=2,
            n_iter=10000,
            random_state=random_state_slider.value)

    def update_plot(iteration, current_embedding, iteration_duration):
        speed_text.value = f'{iteration_duration:.3f} seconds per iteration'
        iteration_text.value = f'{iteration}'

        # update chart
        if iteration == 1 or iteration % steps_between_plotting == 0:
            status.visible = False
            scatter.visible = True

            scatter.data_source.data['x'] = current_embedding[:, 0]
            scatter.data_source.data['y'] = current_embedding[:, 1]
            push_notebook(handle)

    def on_change(_):
        process.stop()
        start_process(is_paused=(play_pause_button.icon == 'play'))

    def show_status(msg):
        status.text = f'⏳ {msg}'
        status.visible = True
        push_notebook(handle)

    def start_process(is_paused=False):
        scatter.visible = False
        push_notebook(handle)
        show_status('Initializing TSNE')
        process.start(is_paused=is_paused, **get_current_params())

    def on_change_neighbors(change):
        if change.name == 'value' and advanced_mode:
            if change.new == 'exact':
                exact_method_metric.layout.display = 'flex'
                approx_method_metric.layout.display = 'none'
            else:
                exact_method_metric.layout.display = 'none'
                approx_method_metric.layout.display = 'flex'

    def play_pause_click(w):
        if w.icon == 'play':
            if process.is_running and process.is_paused:
                process.resume()
            else:
                start_process(is_paused=False)
            w.icon = 'pause'
        elif w.icon == 'pause':
            process.pause()
            w.icon = 'play'

    def stop_button_click(w):
        process.stop()
        play_pause_button.icon = 'play'

    # subscribe to changes of all controls
    for control in control_collection:
        if type(control) not in [widgets.HTML]:
            control.observe(on_change, names='value')

    # additional
    neighbors_select.observe(on_change_neighbors, names='value')

    process = EmbeddingTask(X)
    process.add_handler(update_plot)

    play_pause_button.on_click(play_pause_click)
    stop_button.on_click(stop_button_click)

    if autostart:
        play_pause_button.click()
