from openTSNE import TSNE, nearest_neighbors
from IPython import display
import time
import math
import ipywidgets as widgets
import numpy as np
import sys

import warnings

from bokeh.transform import factor_cmap
from bokeh.palettes import Category10_10
from bokeh.models import ColumnDataSource, Label
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
output_notebook(hide_banner=True)

from multiprocessing import Process, Semaphore, Value, Lock, cpu_count, Pool

from PIL import Image
from io import BytesIO
import base64

def nparr_to_html_tag(nparr, image_mode='L', style='vertical-align:text-top'):
    img = Image.fromarray(nparr, image_mode)
    bytesio = BytesIO()
    img.save(bytesio, format='PNG')
    data = base64.b64encode(bytesio.getvalue()).decode()
    html = f'<img src="data:image/png;base64,{data}" style="{style}"/>'
    return html

def tsne_playground(X, y, steps_between_plotting=30, tooltips = None, additional_columns = dict()):
    # timestamp of last update
    last = time.time()
    last_id = Value('i', 0)
    
    if tooltips is None:
        tooltips = [
            ('index', "$index"),
            ("(x, y)", "($x, $y)"),
            ("label", "@label")
        ]
            
    unique_values = set(y)
    factors = [str(e) for e in unique_values]
    source = ColumnDataSource(data=dict(
        x=X[:,0], 
        y=X[:,1], 
        label=[str(e) for e in y], 
        **additional_columns))
    p = figure(
        output_backend="webgl",
        plot_height=300, 
        plot_width=300, 
        sizing_mode='scale_width', 
        tools='pan,hover,box_zoom,wheel_zoom,save,reset',
        active_drag='pan',
        active_scroll='wheel_zoom',
        toolbar_location="below", 
        tooltips=tooltips
      )
    p.toolbar.logo = None
    scatter = p.scatter(
        source=source, 
        x='x', 
        y='y', 
        color=factor_cmap('label', Category10_10, factors), 
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
    selection_style = {'button_width':'100px'}
    label_style = {'description_width': '100px'}
    
    out = widgets.Output(layout={ 'flex': '1', 'max_width': '90vh'})
    
    heading1 = widgets.HTML(value="<h3>Basic parameters</h3>")
    perplexity_slider = widgets.IntSlider(
        value=30, 
        min=2, 
        max=100, 
        step=1, 
        description='Perplexity', 
        style=label_style, 
        continuous_update=False)
    learning_rate_slider = widgets.IntSlider(
        value=200, 
        min=1, 
        max=1000, 
        step=1, 
        description='Learning rate', 
        style=label_style, 
        continuous_update=False) # epsilon
    initialization_select = widgets.ToggleButtons(
        value='PCA', 
        description='Initialization', 
        style= dict(**selection_style, **label_style), 
        options=['PCA', 'random'])
    negative_gradient_method_select = widgets.ToggleButtons(
        value='interpolation', 
        options=['interpolation', 'barnes-hut'], 
        style= dict(**selection_style, **label_style), 
        description='Gradient method')
    final_momentum_slider = widgets.FloatSlider(
        value=0.8,
        min=0,
        max=1,
        step=0.05,
        description='Momentum', 
        style=label_style,
        continuous_update=False)
    
    heading2 = widgets.HTML(value="<h3>Nearest neighbors</h3>")
    neighbors_select = widgets.ToggleButtons(
        value='approx', 
        options=['exact', 'approx'], 
        style= dict(**selection_style, **label_style), 
        description='Method') # exakt = sklearn ball_tree, approx = pynndescent.NNDescent
    n_jobs_slider = widgets.IntSlider(
        value=math.ceil(cpu_count()/4), 
        min=1, 
        max=cpu_count(), 
        step=1, 
        description='Num jobs', 
        style=label_style, 
        continuous_update=False)
    exact_method_metric = widgets.Dropdown(
        value='euclidean',
        options=nearest_neighbors.BallTree.VALID_METRICS,
        style=label_style,
        description='Metric'
    )
    approx_method_metric = widgets.Dropdown(
        value='euclidean',
        options=nearest_neighbors.NNDescent.VALID_METRICS.keys(),
        style=label_style,
        description='Metric'
    )
    
    heading3 = widgets.HTML(value="<h3>Early exaggeration phase</h3>")
    early_exaggeration_iter_slider = widgets.IntSlider(
        value=250, 
        min=0, 
        max=1000, 
        step=10, 
        description='Number of steps', 
        style=label_style, 
        continuous_update=False)
    early_exaggeration_slider = widgets.IntSlider(
        value=12, 
        min=0, 
        max=100, 
        step=1, 
        description='Exagg factor', 
        style=label_style, 
        continuous_update=False)
    initial_momentum_slider = widgets.FloatSlider(
        value=0.5,
        min=0,
        max=1,
        step=0.05,
        style=label_style,
        description='Momentum', 
        continuous_update=False)
    
    heading4 = widgets.HTML(value="<h3>Other</h3>")
    random_state_slider = widgets.IntSlider(
        value=2506, 
        min=0, 
        max=65000, 
        step=1, 
        description='Random state', 
        style=label_style, 
        continuous_update=False)
    
    heading5 = widgets.HTML(value="<h3>Stats</h3>")
    timer = widgets.HTML(description='Speed', style=label_style)
    iteration = widgets.HTML(description='Iteration', style=label_style)
    plotting_time = widgets.HTML(description='Plotting time', style=label_style)
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
        timer, 
        iteration, 
        plotting_time, 
        player_controls,
    
        log]
    controls = widgets.VBox(control_collection, layout={ 'flex': '0 0 400px' })
    hbox = widgets.HBox([out, controls], layout={ 'width': '100%' })
    display.display(hbox)
    
    with out:
        handle = show(p, notebook_handle=True)

    class EmbeddingTask(Process):
        def __init__(self, is_paused=False):
            super().__init__()
            self.early_exaggeration_iter = 0
            self.exaggeration_phase = True
            self._is_paused = is_paused
            self.pause_lock = Lock()
            if is_paused:
                self.pause_lock.acquire()
        
        def callback(self, id, i, error, embedding):
            # if user paused then this lock can not be aquired (until user clicks play again)
            with self.pause_lock:
                if not self.exaggeration_phase:
                    i = i + self.early_exaggeration_iter

                # stop this tsne run if a new run was started
                if id < last_id.value:
                    return True

                # update chart
                if i == 1 or i % steps_between_plotting == 0:
                    status.visible = False
                    scatter.visible = True
                    
                    before = time.time()
                    scatter.data_source.data['x'] = embedding[:,0]
                    scatter.data_source.data['y'] = embedding[:,1]
                    push_notebook(handle)
                    after = time.time()
                    plotting_time.value = f'{(after - before):.3f}s'

                # print stats
                nonlocal last
                now = time.time()
                timer.value = f'{(now - last):.3f}s per iteration'
                iteration.value = str(i)
                last = now
                
                if self.exaggeration_phase and i == self.early_exaggeration_iter:
                    self.exaggeration_phase = False
        
        def run(self):
            # TODO: atomic
            last_id.value += 1
            current_id = last_id.value
            
            metric_per_neighbor_method = {
                'exact': exact_method_metric.value,
                'approx': approx_method_metric.value
            }

            tsne = TSNE(initialization=initialization_select.value.lower(),
                        perplexity=perplexity_slider.value, 
                        learning_rate=learning_rate_slider.value,
                        negative_gradient_method=negative_gradient_method_select.value.lower(),
                        final_momentum=final_momentum_slider.value,

                        neighbors=neighbors_select.value.lower(),
                        n_jobs = n_jobs_slider.value, 
                        metric = metric_per_neighbor_method[neighbors_select.value.lower()],

                        early_exaggeration=early_exaggeration_slider.value,
                        early_exaggeration_iter=early_exaggeration_iter_slider.value, 
                        initial_momentum=initial_momentum_slider.value,

                        n_components=2, 
                        n_iter=10000, 
                        random_state=random_state_slider.value,

                        callbacks=lambda i, error, embedding: self.callback(current_id, i, error, embedding), 
                        callbacks_every_iters=1)
            
            self.early_exaggeration_iter = tsne.early_exaggeration_iter
            self.exaggeration_phase = tsne.early_exaggeration_iter > 0

#             params = tsne.get_params()
#             for key in ['callbacks', 'callbacks_every_iters', 'n_iter', 'random_state']:
#                 del params[key]
#             param_list = '\n'.join([f'{key} = {value}' for key, value in params.items()])
            
            # the following empty print statement is necessary, because processes
            # behave ridiculously in jupyter notebooks
            print2(f' ')
            scatter.visible = False
            show_status('Initializing TSNE')

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', r"\nThe keyword argument 'parallel=True' was specified but no transformation for parallel execution was possible.")
                embedding = tsne.fit(X)
        #     # for interactivity:
        #     while last_id == id:
        #         embedding.optimize(n_iter=10, inplace=True)
        
        def pause(self):
            self.pause_lock.acquire()
            self._is_paused = True
            
        def resume(self):
            self.pause_lock.release()
            self._is_paused = False
            
        def is_paused(self):
            return self._is_paused
        
    process = Process()

    def on_change(_):
        stop_process()
        start_process(is_paused=(play_pause_button.icon == 'play'))
            
    def print2(msg):
        with log:
            display.clear_output()
            print(msg)
            sys.stdout.flush()
            
    def show_status(msg):
        status.text = f'⏳ {msg}'
        status.visible = True
        push_notebook(handle)
            
    def start_process(is_paused = False):
        nonlocal process
        if not process.is_alive():
            process = EmbeddingTask(is_paused=is_paused)
            process.start()
        
    def stop_process():
        nonlocal process
        if process.is_alive():
            process.terminate()
            process.join()
            
    for control in control_collection:
        if type(control) not in [widgets.HTML]:
            control.observe(on_change, names='value')
    
    def on_change_neighbors(change):
        if change.name == 'value':
            if change.new == 'exact':
                exact_method_metric.layout.display = 'flex'
                approx_method_metric.layout.display = 'none'
            else:
                exact_method_metric.layout.display = 'none'
                approx_method_metric.layout.display = 'flex'
    
    neighbors_select.observe(on_change_neighbors, names='value')
    neighbors_select.value = 'exact'
    neighbors_select.value = 'approx'
    
    def play_pause_click(w):
        if w.icon == 'play':
            if not process.is_alive():
                start_process()
            else:
                process.resume()
            w.icon = 'pause'
        elif w.icon == 'pause':
            process.pause()
            w.icon = 'play'

    def stop_button_click(w):
        stop_process()
        play_pause_button.icon = 'play'

    play_pause_button.on_click(play_pause_click)
    stop_button.on_click(stop_button_click)

    play_pause_button.click()