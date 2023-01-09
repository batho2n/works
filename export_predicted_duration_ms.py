import os
import json
import argparse
import math
import time

import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

import commons
import monotonic_align
import utils
from models import (
    SynthesizerTrn,
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch, linear_to_mel, audio_to_mel, spectrogram_torch
from text import cleaned_text_to_sequence
from text.symbols import symbols

torch.backends.cudnn.benchmark = True
global_step = 0
start = 0

def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    hps = get_hparams()

    os.environ["PYTHONWARNINGS"] = 'ignore:semaphore_tracker:UserWarning'

    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(0)

    dataset_loader = TextAudioSpeakerLoader(hps.meta_file, hps.data)
    collate_fn = TextAudioSpeakerCollate()
    dataloader = DataLoader(dataset_loader, num_workers=4, shuffle=False, pin_memory=True, collate_fn=collate_fn)

    net_g = AlignmentExporter(
        len(symbols),
        # hps.data.filter_length // 2 + 1,
        hps.data.n_mel_channels,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).cuda(0)

    epoch_str, global_step, _ = utils.load_checkpoints(hps.model_path, None, None, net_g, model_d=None, model_d_dp=None,
                                                                   optimizer_g=None, optimizer_d=None,
                                                                   optimizer_d_dp=None, strict=True, speaker_initialization=0)

    export(hps, net_g, dataloader)


def export(hps, nets, loaders):
    net_g = nets
    dataloader = loaders

    net_g.eval()
    start = time.time()

    with open(hps.output_file, 'w', encoding='utf-8') as fo, torch.no_grad():
        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers, audiopaths, texts) in enumerate(dataloader):
            x, x_lengths = x.cuda(0, non_blocking=True), x_lengths.cuda(0, non_blocking=True)
            spec, spec_lengths = spec.cuda(0, non_blocking=True), spec_lengths.cuda(0, non_blocking=True)
            y, y_lengths = y.cuda(0, non_blocking=True), y_lengths.cuda(0, non_blocking=True)
            speakers = speakers.cuda(0, non_blocking=True)

            with autocast(enabled=hps.train.fp16_run):
                mel = linear_to_mel(hps, spec)

                durations = net_g(x, x_lengths, mel, spec_lengths, speakers)

                for a, t, d in zip(audiopaths, texts, durations):
                    fo.write('{}\t{}\t{}\n'.format(a[0], t[0], d[0].cpu().numpy().tolist()))
                    # print(d.size(), a, t)

    print('Elapsed time : {} sec\n'.format(int(time.time() - start)))


def get_hparams(init=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-mp', '--model_path', type=str, required=True, help='Model file path')
    parser.add_argument('-mf', '--meta_file', type=str, required=True, help='Metadata file')
    parser.add_argument('-of', '--output_file', type=str, required=True, help='Output file')
    args = parser.parse_args()

    config_path = os.path.join(os.path.split(args.model_path)[0], 'config.json')
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = utils.HParams(**config)
    hparams.model_path = args.model_path
    hparams.meta_file = args.meta_file
    hparams.output_file = args.output_file
    return hparams


class AlignmentExporter(SynthesizerTrn):
    """
    Module for exporting alignments.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, x_lengths, y, y_lengths, sid=None):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        logw = self.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask
        w_ceil = torch.round(w)

        return w_ceil


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_sid_text, hparams):
        self.prefix_as_dir = getattr(hparams, "prefix_as_dir", False)
        self.audiopaths_sid_text = utils.load_filepaths_and_text(audiopaths_sid_text, '^', self.prefix_as_dir, multi=True)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length  = hparams.filter_length
        self.hop_length     = hparams.hop_length
        self.win_length     = hparams.win_length
        self.sampling_rate  = hparams.sampling_rate

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", hparams.max_input_length)
        self.wav_path = getattr(hparams, "wav_path")
        self.spec_path = getattr(hparams, "spec_path")

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        lengths = []
        for filename, sid, text in self.audiopaths_sid_text:
            audiopath = os.path.join(self.wav_path, filename)
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_sid_text_new.append([filename, sid, text])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        print('Number of training instances =', len(audiopaths_sid_text_new))
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid, text = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2]
        ids = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        sid = self.get_sid(sid)
        return (ids, spec, wav, sid, audiopath, text)

    def get_audio(self, filename):
        wav_path = os.path.join(self.wav_path, filename)
        audio, sampling_rate = utils.load_wav_to_torch(wav_path)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(audio_norm, self.filter_length,
                                self.sampling_rate, self.hop_length, self.win_length,
                                center=False)
        spec = torch.squeeze(spec, 0)
        return spec, audio_norm

    def get_text(self, text):
        text_norm = cleaned_text_to_sequence(text)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        # _, ids_sorted_decreasing = torch.sort(
        #     torch.LongTensor([x[1].size(1) for x in batch]),
        #     dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))
        audiopaths = []
        texts = []

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        # for i in range(len(ids_sorted_decreasing)):
        #     row = batch[ids_sorted_decreasing[i]]
        for i in range(len(batch)):
            row = batch[i]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]

            audiopaths.append([row[4]])
            texts.append([row[5]])

        # if self.return_ids:
        #     return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, ids_sorted_decreasing
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, audiopaths, texts


if __name__ == "__main__":
    main()
