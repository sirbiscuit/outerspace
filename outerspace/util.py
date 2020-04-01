from PIL import Image
from io import BytesIO
import base64
from collections import OrderedDict


class objdict(OrderedDict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

            
def pil_to_html_img(pil_img, style=''):
    bytesio = BytesIO()
    pil_img.save(bytesio, format='PNG')
    data = base64.b64encode(bytesio.getvalue()).decode()
    html = f'<img src="data:image/png;base64,{data}" style="{style}"/>'
    return html
            

def array2d_to_html_img(nparr, image_mode='L', resize=None, style=''):
    img = Image.fromarray(nparr, image_mode)
    if resize is not None:
        img = img.resize(resize)
    return pil_to_html_img(img, style=style)