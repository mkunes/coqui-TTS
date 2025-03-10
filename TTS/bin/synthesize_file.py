#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from distutils.command.config import config
import sys
from argparse import RawTextHelpFormatter

# pylint: disable=redefined-outer-name, unused-argument
from pathlib import Path, PurePath

sys.path.insert(0, "/storage/plzen4-ntis/home/jmatouse/GIT_repos/Coqui-TTS.mod-0.6.1")

from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    # pylint: disable=bad-option-value
    parser = argparse.ArgumentParser(
        description="""Synthesize speech on command line.\n\n"""
        """You can either use your trained model or choose a model from the provided list.\n\n"""
        """If you don't specify any models, then it uses LJSpeech based English model.\n\n"""
        """
    # Example Runs:

    ## Single Speaker Models

    - list provided models

    ```
    $ ./TTS/bin/synthesize.py --list_models
    ```

    - run tts with default models.

    ```
    $ ./TTS/bin synthesize.py --text "Text for TTS"
    ```

    - run a tts model with its default vocoder model.

    ```
    $ ./TTS/bin synthesize.py --text "Text for TTS" --model_name "<language>/<dataset>/<model_name>
    ```

    - run with specific tts and vocoder models from the list

    ```
    $ ./TTS/bin/synthesize.py --text "Text for TTS" --model_name "<language>/<dataset>/<model_name>" --vocoder_name "<language>/<dataset>/<model_name>" --output_path
    ```

    - run your own TTS model (Using Griffin-Lim Vocoder)

    ```
    $ ./TTS/bin/synthesize.py --text "Text for TTS" --model_path path/to/model.pth.tar --config_path path/to/config.json --out_path output/path/speech.wav
    ```

    - run your own TTS and Vocoder models
    ```
    $ ./TTS/bin/synthesize.py --text "Text for TTS" --model_path path/to/config.json --config_path path/to/model.pth.tar --out_path output/path/speech.wav
        --vocoder_path path/to/vocoder.pth.tar --vocoder_config_path path/to/vocoder_config.json
    ```

    ## MULTI-SPEAKER MODELS

    - list the available speakers and choose as <speaker_id> among them.

    ```
    $ ./TTS/bin/synthesize.py --model_name "<language>/<dataset>/<model_name>"  --list_speaker_idxs
    ```

    - run the multi-speaker TTS model with the target speaker ID.

    ```
    $ ./TTS/bin/synthesize.py --text "Text for TTS." --out_path output/path/speech.wav --model_name "<language>/<dataset>/<model_name>"  --speaker_idx <speaker_id>
    ```

    - run your own multi-speaker TTS model.

    ```
    $ ./TTS/bin/synthesize.py --text "Text for TTS" --out_path output/path/speech.wav --model_path path/to/config.json --config_path path/to/model.pth.tar --speakers_file_path path/to/speaker.json --speaker_idx <speaker_id>
    ```
    """,
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument(
        "--list_models",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="list available pre-trained tts and vocoder models.",
    )
    parser.add_argument("--text_file", type=str, default=None, help="Text file to generate speech from.")

    # Args for running pre-trained TTS models.
    parser.add_argument(
        "--model_name",
        type=str,
        default="tts_models/en/ljspeech/tacotron2-DDC",
        help="Name of one of the pre-trained tts models in format <language>/<dataset>/<model_name>",
    )
    parser.add_argument(
        "--vocoder_name",
        type=str,
        default=None,
        help="Name of one of the pre-trained  vocoder models in format <language>/<dataset>/<model_name>",
    )

    # Args for running custom models
    parser.add_argument("--config_path", default=None, type=str, help="Path to model config file.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model file.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output wav file path directory.",
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default="utt",
        help="Output wav filename.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="",
        help="Output wav file path.",
    )
    parser.add_argument(
        "--concat_audio",
        action='store_true',
        help="Concatenate audio to a single output file",
        default=False
    )
    parser.add_argument(
        "-1", "--use_infile_label",
        action='store_true',
        help="Use in-file label (1st word) as output file name",
        default=False
    )
    parser.add_argument(
        "--rm_last_word",
        action='store_true',
        help="Remove last word (typically corresponding to a pause)",
        default=False
    )
    parser.add_argument("--use_cuda", action='store_true', help="Run model on CUDA.", default=False)
    parser.add_argument(
        "--vocoder_path",
        type=str,
        help="Path to vocoder model file. If it is not defined, model uses GL as vocoder. Please make sure that you installed vocoder library before (WaveRNN).",
        default=None,
    )
    parser.add_argument("--vocoder_config_path", type=str, help="Path to vocoder model config file.", default=None)
    parser.add_argument(
        "--encoder_path",
        type=str,
        help="Path to speaker encoder model file.",
        default=None,
    )
    parser.add_argument("--encoder_config_path", type=str, help="Path to speaker encoder config file.", default=None)

    # args for multi-speaker synthesis
    parser.add_argument("--speakers_file_path", type=str, help="JSON file for multi-speaker model.", default=None)
    parser.add_argument(
        "--speaker_idx",
        type=str,
        help="Target speaker ID for a multi-speaker TTS model.",
        default=None,
    )
    parser.add_argument(
        "--speaker_wav",
        nargs="+",
        help="wav file(s) to condition a multi-speaker TTS model with a Speaker Encoder. You can give multiple file paths. The d_vectors is computed as their average.",
        default=None,
    )
    parser.add_argument("--gst_style", help="Wav path file for GST style reference.", default=None)
    parser.add_argument(
        "--list_speaker_idxs",
        help="List available speaker ids for the defined multi-speaker model.",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
    )
    # aux args
    parser.add_argument(
        "--save_spectogram",
        action='store_true',
        help="If true save raw spectogram for further (vocoder) processing in out_path.",
        default=False,
    )

    args = parser.parse_args()

    # print the description if either text or list_models is not set
    if args.text_file is None and not args.list_models and not args.list_speaker_idxs:
        parser.parse_args(["-h"])

    # load model manager
    path = Path(__file__).parent / "../.models.json"
    manager = ModelManager(path)

    model_path = None
    config_path = None
    speakers_file_path = None
    vocoder_path = None
    vocoder_config_path = None
    encoder_path = None
    encoder_config_path = None

    # CASE1: list pre-trained TTS models
    if args.list_models:
        manager.list_models()
        sys.exit()

    # CASE2: load pre-trained model paths
    if args.model_name is not None and not args.model_path:
        model_path, config_path, model_item = manager.download_model(args.model_name)
        args.vocoder_name = model_item["default_vocoder"] if args.vocoder_name is None else args.vocoder_name

    if args.vocoder_name is not None and not args.vocoder_path:
        vocoder_path, vocoder_config_path, _ = manager.download_model(args.vocoder_name)

    # CASE3: set custom model paths
    # JMa: if config is not given => use config from the corresponding model/vocoder/encoder path
    if args.model_path is not None:
        model_path = args.model_path
        config_path = args.config_path if args.config_path else PurePath(Path(model_path).parent, "config.json")
        speakers_file_path = args.speakers_file_path

    if args.vocoder_path is not None:
        vocoder_path = args.vocoder_path
        vocoder_config_path = args.vocoder_config_path if args.vocoder_config_path else PurePath(Path(vocoder_path).parent, "config.json")

    if args.encoder_path is not None:
        encoder_path = args.encoder_path
        encoder_config_path = args.encoder_config_path if args.encoder_config_path else PurePath(Path(encoder_path).parent, "config.json")

    # load models
    synthesizer = Synthesizer(
        model_path,
        config_path,
        speakers_file_path,
        vocoder_path,
        vocoder_config_path,
        encoder_path,
        encoder_config_path,
        args.use_cuda,
    )

    # query speaker ids of a multi-speaker model.
    if args.list_speaker_idxs:
        print(
            " > Available speaker ids: (Set --speaker_idx flag to one of these values to use the multi-speaker model."
        )
        print(synthesizer.tts_model.speaker_manager.speaker_ids)
        return

    # check the arguments against a multi-speaker model.
    if synthesizer.tts_speakers_file and (not args.speaker_idx and not args.speaker_wav):
        print(
            " [!] Looks like you use a multi-speaker model. Define `--speaker_idx` to "
            "select the target speaker. You can list the available speakers for this model by `--list_speaker_idxs`."
        )
        return
    
    # Read lines (=sentences) from the input text file
    with open(args.text_file, 'rt') as fr:
        lines = fr.read().splitlines()
    
    # Resulting wav
    tot_wav = []

    # RUN THE SYNTHESIS line-by-line
    for ix, line in enumerate(lines):
        # Extract words
        words = line.split()

        # Use first word as utterance name?
        if args.use_infile_label:
            uname = words[0]
            sent_beg = 1
        else:
            uname = "{}{:03d}".format(args.out_name, ix)
            sent_beg = 0
        
        # Remove last word?
        sent_end = -1 if args.rm_last_word else len(words)

        # Prepare text to synthesize
        text = " ".join(words[sent_beg:sent_end])

        # kick it
        wav = synthesizer.tts(text, args.speaker_idx, args.speaker_wav, args.gst_style)

        # Concatenate resulting wav
        if args.concat_audio:
            print(" > Text #{:03d}: {}".format(ix, text))
            tot_wav.append(wav)
        else:
            # Save the wav for each line 
            # print(" > Saving output to {}".format(out_path))
            # Prepare output path
            out_path = PurePath(args.out_dir, "{}.wav".format(uname))
            print(" > Text #{:03d}: {} --> {}".format(ix, text, out_path))
            synthesizer.save_wav(wav, out_path)
    
    if args.concat_audio:
        # Concatenate resulting wav
        print(" > Saving audio to {}".format(args.out_path))
        single_wav = []
        for wav in tot_wav:
            single_wav.extend(list(wav))
        synthesizer.save_wav(single_wav, args.out_path)

if __name__ == "__main__":
    main()
