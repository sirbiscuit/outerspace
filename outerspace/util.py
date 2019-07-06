from PIL import Image
from io import BytesIO
import base64


def array2d_to_html_img(nparr, image_mode='L', resize=None, style=''):
    img = Image.fromarray(nparr, image_mode)
    if resize is not None:
        img = img.resize(resize)
    bytesio = BytesIO()
    img.save(bytesio, format='PNG')
    data = base64.b64encode(bytesio.getvalue()).decode()
    html = f'<img src="data:image/png;base64,{data}" style="{style}"/>'
    return html
