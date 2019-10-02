from bokeh.io import output_notebook, push_notebook, show
from bokeh.transform import factor_cmap
from bokeh.palettes import Category10_10, Category20_20, viridis
from bokeh.models import ColumnDataSource, Label
from bokeh.plotting import figure
from .ui import create_widgets
from ipywidgets import HBox, VBox, Output, Layout
from .EmbeddingTask import EmbeddingTask
from IPython.display import display
from .util import objdict

output_notebook(hide_banner=True)


class PlaygroundWidget:
    def __init__(self, transformation_method, X, y, advanced_mode, autostart,
                 plot_every_iters, additional_columns, tooltips, colors):
        if plot_every_iters is None:
            # TODO: measure unresponsiveness caused by plotting
            plot_every_iters = max(len(X) // 2000, 1)

        self.transformation_method = transformation_method
        self.advanced_mode = advanced_mode
        self.autostart = autostart

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
            x=X[:, 0],  # arbitrary
            y=X[:, 1],  # arbitrary
            label=[str(e) for e in y],
            **additional_columns))

        # create plot
        tools = ['pan', 'box_zoom', 'wheel_zoom', 'save', 'reset']
        if tooltips:
            tools.append('hover')
        self.p = figure(
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
        self.p.toolbar.logo = None  # remove bokeh logo

        self.scatter = self.p.scatter(
            source=source,
            x='x',
            y='y',
            color=color,
            legend=False,
            line_width=0,
            alpha=0.8,
            size=5)
        self.scatter.visible = False

        # status message on top of graph
        self.status_message = Label(x=20, y=20, x_units='screen',
                                    y_units='screen', text='⏳',
                                    render_mode='css', text_font_size='40px')
        self.status_message.visible = False
        self.p.add_layout(self.status_message)

        # render all widgets
        self.out = Output(layout={'flex': '1', 'max_width': '90vh'})
        widget_container = self.create_all_widgets()
        self.hbox = HBox([self.out, widget_container], layout={'width': '100%'})

        def update_plot(command, iteration, payload):
            self.widgets._iteration.value = f'{iteration}'
            push_notebook(self.handle)

            if command == 'status':
                self.show_status(payload)
            elif command == 'speed':
                self.widgets._speed.value = f'{payload:.3f} seconds per iteration'
                push_notebook(self.handle)
            elif command == 'embedding':
                self.hide_status()
                if iteration == 1 or iteration % plot_every_iters == 0:
                    embedding = payload
                    self.show_plot(embedding[:, 0], embedding[:, 1])
            elif command == 'error':
                self.show_status(payload)
            elif command == 'start':
                self.clear_plot()

        self.process = EmbeddingTask(X, transformation_method=self.transformation_method.run_transformation)
        self.process.add_handler(update_plot)

        def status_changed(e):
            new_value = e.new
            play_pause_button = self.widgets._play_pause
            status_icon_mapping = dict(
                paused='play',
                running='pause',
                idle='play'
            )
            play_pause_button.icon = status_icon_mapping[new_value]

        self.process.observe(status_changed, names='status')

    def create_all_widgets(self):
        parameter_widgets = self.transformation_method.get_widgets()

        stats_widgets = create_widgets([
            dict(name='_stats', type='heading', text='Stats'),
            dict(name='_speed', type='text', value='Speed:'),
            dict(name='_iteration', type='text', value='Iteration:'),
        ])

        player_widgets = create_widgets([
            dict(name='_play_pause', type='button', icon='play',
                 layout=Layout(width='40px', height='40px')),
            dict(name='_stop', type='button', icon='stop',
                 layout=Layout(width='40px', height='40px')),
        ])

        all = [parameter_widgets, stats_widgets, player_widgets]

        self.widgets = objdict([p for widgets in all for p in widgets.items()])

        #
        # additional behaviour
        #

        # play / pause mechanics
        def play_pause_click(_):
            self.play_pause()

        def stop_button_click(_):
            self.stop()

        self.widgets._play_pause.on_click(play_pause_click)
        self.widgets._stop.on_click(stop_button_click)

        # subscribe to changes of all controls
        def on_change(_):
            if self.process.status == 'paused':
                self.stop()
            elif self.process.status == 'running':
                self.stop()
                self.start()
        for widget in parameter_widgets.values():
            widget.observe(on_change, names='value')

        # hide some widgets if advanced_mode is disabled
        if not self.advanced_mode:
            for widget in parameter_widgets.values():
                if widget.advanced:
                    widget.layout.display = 'none'

        # compose resulting UI
        player_container = HBox(list(player_widgets.values()))
        all_widgets = list(parameter_widgets.values()) + list(stats_widgets.values()) + [player_container]
        container = VBox(all_widgets, layout={'flex': '0 0 350px'})

        return container

    def start(self):
        self.process.start(**self.transformation_method.get_current_params(self.widgets))

    def resume(self):
        self.process.resume()

    def pause(self):
        self.process.pause()

    def play_pause(self):
        if self.process.status == 'paused':
            self.resume()
        elif self.process.status == 'idle':
            self.start()
        else:
            self.pause()

    def stop(self):
        self.process.stop()

    def show_status(self, msg):
        self.status_message.text = f'⏳ {msg}'
        self.status_message.visible = True
        push_notebook(self.handle)

    def hide_status(self):
        self.status_message.visible = False
        push_notebook(self.handle)

    def show_plot(self, x, y):
        self.scatter.visible = True
        self.scatter.data_source.data['x'] = x
        self.scatter.data_source.data['y'] = y
        push_notebook(self.handle)

    def clear_plot(self):
        self.scatter.visible = False
        push_notebook(self.handle)

    def _ipython_display_(self, **kwargs):
        display(self.hbox)
        with self.out:
            self.handle = show(self.p, notebook_handle=True)

        # TODO: consider autostarting in constructor
        if self.autostart:
            self.start()
