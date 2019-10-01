from ipywidgets import HTML, IntSlider, FloatSlider, ToggleButtons, Dropdown, Button
from .util import objdict

toggle_buttons_style = {'button_width': '100px'}
label_style = {'description_width': '100px'}


def create_heading(text, **kwargs):
    return HTML(value=f"<h3>{text}</h3>", **kwargs)


def create_text(**kwargs):
    return HTML(
        style=label_style)


def create_int_slider(**kwargs):
    return IntSlider(
        style=label_style,
        continuous_update=False,
        **kwargs)


def create_toggle_buttons(**kwargs):
    return ToggleButtons(
        style=dict(**toggle_buttons_style, **label_style),
        **kwargs)


def create_float_slider(**kwargs):
    return FloatSlider(
        style=label_style,
        continuous_update=False,
        **kwargs)


def create_dropdown(**kwargs):
    return Dropdown(
        style=label_style,
        **kwargs
    )


def create_button(**kwargs):
    return Button(**kwargs)


def create_widget(type, **kwargs):
    mapping = dict(
        heading=create_heading,
        text=create_text,
        int_slider=create_int_slider,
        float_slider=create_float_slider,
        toggle_buttons=create_toggle_buttons,
        dropdown=create_dropdown,
        button=create_button,
    )
    widget = mapping[type](**kwargs)
    return widget


def create_widgets(widget_params_list):
    widgets = [(params.pop('name'), create_widget(**params))
               for params in widget_params_list]
    return objdict(widgets)