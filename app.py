import io
import os
import tempfile
import time, argparse, sys
import torch
from espnet2.bin.tts_inference import Text2Speech
from parallel_wavegan.utils import download_pretrained_model
from parallel_wavegan.utils import load_model, read_hdf5
from scipy.io.wavfile import write
from pathlib import Path
from flask import Flask, request
import base64

fs = 22050

# model_name = os.environ['TTS_MODEL_NAME']
model_dir = Path(os.environ['MODEL_PATH'])
device = 'cpu'


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


def run_tts(text: str):
    with torch.no_grad():
        c_mel = text2speech(text.lower())['feat_gen']
        wav = vocoder.inference(c_mel)

    # fd, name = tempfile.mkstemp('.wav')
    fd = io.BytesIO()

    ## here all of your synthesized audios will be saved
    # folder_to_save, wav_name = "synthesized_wavs", "example.wav"

    # Path(folder_to_save).mkdir(parents=True, exist_ok=True)

    write(fd, fs, wav.view(-1).cpu().numpy())
    # with open(, "rb") as image_file:
    return base64.b64encode(fd.read())


web_service = Flask(__name__)


@web_service.route('/data', methods=['GET', 'POST'])
def mydata():
    text = request.args.get('sentence').lower()
    return run_tts(text)


def main():
    web_service.run(debug=True, host='0.0.0.0', port=80)

    # sample_text = "Мысалы: интерактивті ақылды көмекшілер, навигациялық жүйелер, ескерту жүйелері және ерекше қажеттіліктері бар адамдарға арналған қолданбалар."
    # res = run_tts(sample_text)
    # print(res)

    pass


if __name__ == '__main__':
    main()
