from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import torch
import torch.nn as nn
#import librosa
from scipy.special import softmax
import numpy as np
import torchaudio

# TODO: is torchaudio necessary for resampling? is there something similar in e.g. TTS.utils.audio?
# TODO: what about numpy and scipy - can they be replaced?


# add to config:
#    use_mos_loss: bool = True
#    mos_loss_weight: float = 1     # TODO: figure out the best value
#    mos_loss_params: dict = field(
#        default_factory=lambda: {
#            "model_path": "<path>", # directory of the MOS prediction model
#            "max_segment_duration": 30, # in seconds
#            "min_segment_duration": 10, # in seconds
#            "orig_sr": 22050, # or whatever is the sample rate of the synthesised speech
#            "target_sr": 16000,  # sample rate for wav2vec2 input
#        }
#    )



class MOSLoss(nn.Module):
    """
    Loss based on MOS prediction
    """
    def __init__(self, model_path, max_segment_duration=30, min_segment_duration=10, orig_sr=22050, target_sr=16000):
        super().__init__()

        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)

        self.orig_sr = orig_sr             # sample rate of synthesised wavs (coqui-tts default is 22050 Hz)
        self.target_sr = target_sr         # sample rate required by MOS predictor (wav2vec2 standard is 16kHz)

        # each wav will get split into shorter segments, with duration between min_segment_duration and max_segment_duration (in seconds)
        self.max_samples = max_segment_duration * self.target_sr # max segment length (number of audio samples)
        self.min_samples = min_segment_duration * self.target_sr # min segment length (number of audio samples)


        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1, 
            sampling_rate=self.target_sr, 
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True
        )

        self.resample = torchaudio.transforms.Resample(orig_freq=self.orig_sr, new_freq=16000)



    def forward(self, y_hat, target_MOS=5):
        # TODO: get target_MOS from somewhere, preferably without recalculating it on every call

        #speech_array, sampling_rate = librosa.load(filename)
        #speech_array = librosa.resample(speech_array, orig_sr=self.orig_sr, target_sr=16000)

        speech_batches = self.resample(y_hat)
        if speech_batches.ndim == 3: # B x 1 x T
            speech_batches = speech_batches.squeeze(1) # B x T

        # concatenate batches for speed (TODO: can I do this??)
        speech_batches = torch.reshape(speech_batches, (1,-1)) # B x T -> 1 x BT

        speech_batches = speech_batches.cpu().detach().numpy()

        loss = 0
        nlosses = 0
        for speech_array in speech_batches:
            segments = self.split_audio(speech_array)
            MOS_all = []

            for segment in segments:
                MOS_all.append(self.predict_MOS(segment))

            predicted_MOS = sum(MOS_all) / len(MOS_all)

            loss += max(0,target_MOS - predicted_MOS)
            nlosses += 1

        loss = torch.tensor(loss / nlosses,requires_grad=True)

        #print("loss:")
        #print(loss)

        return loss

    def split_audio(self, speech_array):

        max_samples = self.max_samples
        min_samples = self.min_samples

        segments = []
        startIdx = 0
        numSamples = len(speech_array)
        while startIdx < numSamples:
            if numSamples - startIdx < max_samples:
                if numSamples - startIdx < min_samples:
                    # if there's not enough remaining data for a segment, start the last segment a bit earlier -> there will be overlap with the segment before
                    startIdx = max(0, numSamples - min_samples)
                segments.append(speech_array[startIdx:numSamples])
                break
            else:
                segments.append(speech_array[startIdx:(startIdx + max_samples)])
                startIdx += max_samples
        return segments



    def predict_MOS(self, speech_array):

        #print("segment inside predict_MOS:")
        #print(type(speech_array))
        #print(speech_array.shape)


        model = self.model

        inputs = self.feature_extractor(
            speech_array, 
            sampling_rate=self.target_sr, 
            return_tensors="pt", 
            return_attention_mask=True
        )

        #print("inputs (inside predict_MOS):")
        #print(type(inputs))
        #print(inputs)
        #print(inputs.input_values.shape)

        # TODO: this probably shouldn't be HERE...
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input_vals = inputs.input_values.to(device)
        att_mask = inputs.attention_mask.to(device)

        with torch.no_grad():
            output = model(input_vals, attention_mask=att_mask)
        logits = output.logits[0].cpu().detach().numpy()
        probs = softmax(logits)
        mos = np.sum(np.arange(1,6) * probs)

        #print("mos:")
        #print(mos)

        return mos
