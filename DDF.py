import hydra
import hydra.utils as utils
import json
from pathlib import Path
import torch
import numpy as np
import librosa
from tqdm import tqdm
import pyloudnorm
from preprocess import preemphasis
from model import Encoder, Decoder


@hydra.main(config_path="Training/VQ-VAE/Configuration_files/DDF.yaml")
def DDF(cfg):

    filter_list_path = Path(utils.to_absolute_path(cfg.filter_list))
    with open(filter_list_path) as file:
        filter_list = json.load(file)
    in_dir = Path(utils.to_absolute_path(cfg.in_dir))
    out_dir = Path(utils.to_absolute_path(cfg.out_dir))
    out_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(**cfg.model.encoder)
    decoder = Decoder(**cfg.model.decoder)
    encoder.to(device)
    decoder.to(device)
    print("Load checkpoint from: {}:".format(cfg.checkpoint))
    checkpoint_path = utils.to_absolute_path(cfg.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    encoder.eval()
    decoder.eval()
    meter = pyloudnorm.Meter(cfg.preprocessing.sr)

    #---------------------------------------
    if cfg.privacy_preference == "Low":
       for wav_path, speaker_id, out_filename in tqdm(filter_list):
        wav_path = in_dir / wav_path
        # librosa.load (it will return audio time series, and its sampling rate)
        wav, _ = librosa.load(wav_path.with_suffix(".wav"), sr=cfg.preprocessing.sr)
        ref_loudness = meter.integrated_loudness(wav)
        wav = wav / np.abs(wav).max() * 0.999
        path = out_dir / out_filename
        
        # to return raw recording in mel-spectrogram without any filtering  
        if cfg.output_type == "Embedding": 
         mel = librosa.feature.melspectrogram(
            preemphasis(wav, cfg.preprocessing.preemph),
            sr=cfg.preprocessing.sr,
            n_fft=cfg.preprocessing.n_fft,
            n_mels=cfg.preprocessing.n_mels,
            hop_length=cfg.preprocessing.hop_length,
            win_length=cfg.preprocessing.win_length,
            fmin=cfg.preprocessing.fmin,
            power=1)
         logmel = librosa.amplitude_to_db(mel, top_db=cfg.preprocessing.top_db)
         logmel = logmel / cfg.preprocessing.top_db + 1
         mel = torch.FloatTensor(logmel).squeeze().to(device).numpy()
         np.savetxt(path.with_suffix(".mel.txt"), mel)
            
        # to return raw recording in waveform without any filtering 
        if cfg.output_type == "Recording": 
         librosa.output.write_wav(path.with_suffix(".wav"), wav.astype(np.float32), sr=cfg.preprocessing.sr)
         
    #---------------------------------------
    if cfg.privacy_preference == "Moderate":
      dataset_path = Path(utils.to_absolute_path("Training/Datasets")) / cfg.dataset.path
      with open(dataset_path / "speakers.json") as file:
           speakers = sorted(json.load(file))          
          
      for wav_path, speaker_id, out_filename in tqdm(filter_list):
        wav_path = in_dir / wav_path
        wav, _ = librosa.load(
            wav_path.with_suffix(".wav"),
            sr=cfg.preprocessing.sr)
        ref_loudness = meter.integrated_loudness(wav)
        wav = wav / np.abs(wav).max() * 0.999
        mel = librosa.feature.melspectrogram(
            preemphasis(wav, cfg.preprocessing.preemph),
            sr=cfg.preprocessing.sr,
            n_fft=cfg.preprocessing.n_fft,
            n_mels=cfg.preprocessing.n_mels,
            hop_length=cfg.preprocessing.hop_length,
            win_length=cfg.preprocessing.win_length,
            fmin=cfg.preprocessing.fmin,
            power=1)
        logmel = librosa.amplitude_to_db(mel, top_db=cfg.preprocessing.top_db)
        logmel = logmel / cfg.preprocessing.top_db + 1
        mel = torch.FloatTensor(logmel).unsqueeze(0).to(device)
        speaker = torch.LongTensor([speakers.index(speaker_id)]).to(device)
        path = out_dir / out_filename

        if cfg.output_type == "Recording":
         with torch.no_grad():
            vq, _ = encoder.encode(mel)
            output = decoder.generate(vq, speaker)
         output_loudness = meter.integrated_loudness(output)
         output = pyloudnorm.normalize.loudness(output, output_loudness, ref_loudness)
         librosa.output.write_wav(path.with_suffix(".wav"), output.astype(np.float32), sr=cfg.preprocessing.sr)
         
        if cfg.output_type == "Embedding":
         with torch.no_grad():
            vq, _ = encoder.encode(mel)
            speaker = decoder.speaker(speaker)
         vq = vq.squeeze().to(device).numpy()
         speaker = speaker.squeeze().to(device).numpy()

         np.savetxt(path.with_suffix(".vq.txt"), vq)
         np.savetxt(path.with_suffix(".speaker.txt"), speaker)
         
    #---------------------------------------
    if cfg.privacy_preference == "High":
      dataset_path = Path(utils.to_absolute_path("Training/Datasets")) / cfg.dataset.path
      with open(dataset_path / "speakers.json") as file:
           speakers = sorted(json.load(file))          
                
      for wav_path, speaker_id, out_filename in tqdm(filter_list):
        wav_path = in_dir / wav_path
        wav, _ = librosa.load(
            wav_path.with_suffix(".wav"),sr=cfg.preprocessing.sr)
        ref_loudness = meter.integrated_loudness(wav)
        wav = wav / np.abs(wav).max() * 0.999
        mel = librosa.feature.melspectrogram(
            preemphasis(wav, cfg.preprocessing.preemph),
            sr=cfg.preprocessing.sr,
            n_fft=cfg.preprocessing.n_fft,
            n_mels=cfg.preprocessing.n_mels,
            hop_length=cfg.preprocessing.hop_length,
            win_length=cfg.preprocessing.win_length,
            fmin=cfg.preprocessing.fmin,
            power=1)
        logmel = librosa.amplitude_to_db(mel, top_db=cfg.preprocessing.top_db)
        logmel = logmel / cfg.preprocessing.top_db + 1
        mel = torch.FloatTensor(logmel).unsqueeze(0).to(device)
        speaker = torch.LongTensor([speakers.index(speaker_id)]).to(device)
        path = out_dir / out_filename

        if cfg.output_type == "Recording":
         with torch.no_grad():
            vq, _ = encoder.encode(mel)
            output = decoder.generate(vq, speaker)
         output_loudness = meter.integrated_loudness(output)
         output = pyloudnorm.normalize.loudness(output, output_loudness, ref_loudness)
         librosa.output.write_wav(path.with_suffix(".wav"), output.astype(np.float32), sr=cfg.preprocessing.sr)
        
        if cfg.output_type == "Embedding":
         with torch.no_grad():
            vq, _ = encoder.encode(mel)
         vq = vq.squeeze().cpu().numpy()
         np.savetxt(path.with_suffix(".vq.txt"), vq)

if __name__ == "__main__":
    DDF()
