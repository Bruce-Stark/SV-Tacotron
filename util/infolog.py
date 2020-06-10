import json
import atexit
from datetime import datetime
from threading import Thread
from urllib.request import Request, urlopen
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


_format = '%Y-%m-%d %H:%M:%S.%f'
_file = None
_run_name = None
_slack_url = None


def init(filename, run_name, slack_url=None):
  global _file, _run_name, _slack_url
  _close_logfile()
  _file = open(filename, 'a')
  _file.write('\n-----------------------------------------------------------------\n')
  _file.write('Starting new training run\n')
  _file.write('-----------------------------------------------------------------\n')
  _run_name = run_name
  _slack_url = slack_url


def log(msg, slack=False):
  print(msg)
  if _file is not None:
    _file.write('[%s]  %s\n' % (datetime.now().strftime(_format)[:-3], msg))
  if slack and _slack_url is not None:
    Thread(target=_send_slack, args=(msg,)).start()


def _close_logfile():
  global _file
  if _file is not None:
    _file.close()
    _file = None


def _send_slack(msg):
  req = Request(_slack_url)
  req.add_header('Content-Type', 'application/json')
  urlopen(req, json.dumps({
    'username': 'tacotron',
    'icon_emoji': ':taco:',
    'text': '*%s*: %s' % (_run_name, msg)
  }).encode())


def plot_alignment(alignment, path, info=None):
  fig, ax = plt.subplots()
  im = ax.imshow(
    alignment,
    aspect='auto',
    origin='lower',
    interpolation='none')
  fig.colorbar(im, ax=ax)
  xlabel = 'Decoder timestep'
  if info is not None:
    xlabel += '\n\n' + info
  plt.xlabel(xlabel)
  plt.ylabel('Encoder timestep')
  plt.tight_layout()
  plt.savefig(path, format='png')


atexit.register(_close_logfile)