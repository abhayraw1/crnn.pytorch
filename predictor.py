import torch
from torch.autograd import Variable
from . import utils
from . import dataset
from PIL import Image
from pathlib import Path

from . import crnn

model_path = Path(__file__).parent/'data/crnn.pth'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
model = crnn.CRNN(32, 1, 37, 256)

if torch.cuda.is_available():
    model = model.cuda()

print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)
transformer = dataset.resizeNormalize((100, 32))

def predict(img_path=None, arr=None):
    assert img_path is not None or arr is not None
    if arr is not None:
        image = Image.fromarray(arr)
    else:
        image = Image.open(img_path)
    image = image.convert('L')
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    # print('%-20s => %-20s' % (raw_pred, sim_pred))
    return sim_pred

