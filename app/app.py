import base64
import io
import os
from pathlib import Path

import torch
from espnet2.bin.tts_inference import Text2Speech
from flask import Flask, request
from parallel_wavegan.utils import load_model
from scipy.io.wavfile import write
from loguru import logger

from app.utils import resolve_num2words

fs = 22050

# model_name = os.environ['TTS_MODEL_NAME']
model_dir = Path(os.environ['MODEL_PATH'])

device = 'cuda'


def ensure_path_exists(path: Path):
    if not path.exists():
        filename = path.name
        if (another_path := path.parents[1] / filename).exists():
            return another_path
        raise FileNotFoundError(path)
    return path


## specify the path to vocoder's checkpoint
vocoder_checkpoint = ensure_path_exists(Path(os.environ['VOCODER_PATH']) / "checkpoint-400000steps.pkl")

vocoder = load_model(vocoder_checkpoint).to(device).eval()
vocoder.remove_weight_norm()

## specify path to the main model(transformer/tacotron2/fastspeech) and its config file
config_file = ensure_path_exists(model_dir / "config.yaml")
model_path = ensure_path_exists(model_dir / "train.loss.ave_5best.pth")


def get_text2speech():
    curdir = os.curdir
    os.chdir('/espnet')
    text2speech = Text2Speech(
        config_file,
        model_path,
        device=device,
        # Only for Tacotron 2
        threshold=0.5,
        minlenratio=0.0,
        maxlenratio=10.0,
        use_att_constraint=True,
        backward_window=1,
        forward_window=3,
        # Only for FastSpeech & FastSpeech2
        speed_control_alpha=1.0,
    )
    text2speech.spc2wav = None  # Disable griffin-lim
    os.chdir(curdir)
    return text2speech


text2speech = get_text2speech()


def run_tts(text: str):
    logger.debug("text: {}", text)
    with torch.no_grad():
        c_mel = text2speech(text.lower())['feat_gen']
        wav = vocoder.inference(c_mel)

    fd = io.BytesIO()

    ## here all of your synthesized audios will be saved
    # folder_to_save, wav_name = "synthesized_wavs", "example.wav"

    # Path(folder_to_save).mkdir(parents=True, exist_ok=True)

    write(fd, fs, wav.view(-1).cpu().numpy())
    return base64.b64encode(fd.read())


app = Flask(__name__)


@app.route('/data', methods=['GET', 'POST'])
def mydata():
    text = request.args.get('sentence').lower()
    text = resolve_num2words(text)
    return run_tts(text)


def create_app():
    return app
