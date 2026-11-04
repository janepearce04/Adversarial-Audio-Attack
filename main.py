import logging
import math
import os
import re
from pickle import FALSE

from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd

# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import clean_punctuation
from whisper_utils import _torch_extract_fbank_features

import torchaudio
from datasets import load_dataset
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline

from utils import clean_punctuation, load_model_processor, preprocess_waveform
from whisper_utils import _torch_extract_fbank_features

import soundfile as sf

# Suppress warnings from transformers
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("TTS").setLevel(logging.ERROR)


def load_audio(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # (time, channels)
    return torch.from_numpy(data.T.copy()), sr                 # (channels, time), like torchaudio.load


def save_audio(path, audio, sr):
    sf.write(path, audio.detach().cpu().numpy().T, sr)

path = hf_hub_download(
    repo_id="Matthijs/cmu-arctic-xvectors",
    repo_type="dataset",
    revision="refs/convert/parquet",
    filename="default/validation/0000.parquet",
)

members = [
    {"model": wav2vec2_model, "name": "wav2vec2", "processor": wav2vec2_processor},
    {"model": whisper_model_a, "name": "whisper", "processor": whisper_processor_a},
    {"model": whisper_model_b, "name": "whisper", "processor": whisper_processor_b},
]
ens = EnsembleBIM(members, audio, target_string, eps=0.01, alpha=0.0008, num_iters=600, verbose=True)
adv_audio, delta, history = ens.attack()

class EnsembleBIM:
    """
    Optimizes one perturbation against several ASR models at once.
    Each member's loss is summed before backward(); since every loss
    traces back to the same adv_audio leaf, one .backward() call gives
    the combined gradient across all models. BIM only uses the *sign*
    of that gradient, so this is equivalent (up to scale) to averaging
    gradients computed separately per model.
    """

    def __init__(self, members, audio, target_label, eps, alpha, num_iters=600, beta=0.1, verbose=False):
        """
        members: list of dicts, one per model:
            {"model": nn.Module (eval, on device), "name": "wav2vec2"|"whisper",
             "processor": matching processor, "weight": float (optional)}
        """
        self.members = members
        for m in self.members:
            m.setdefault("weight", 1.0 / len(members))

        self.device = next(members[0]["model"].parameters()).device
        self.eps, self.alpha, self.beta, self.num_iters, self.verbose = eps, alpha, beta, num_iters, verbose

        self.orig_audio = audio.clone().detach().to(self.device)
        self.adv_audio = audio.clone().detach().to(torch.float32).requires_grad_(True).to(self.device)

        # tokenize the target once per model (different tokenizers/casing rules)
        for m in self.members:
            label = clean_punctuation(target_label)
            label = label.upper() if m["name"] == "wav2vec2" else label.lower()
            m["target_label"] = label
            ids = m["processor"](text=label, return_tensors="pt").input_ids.to(self.device)
            m["target_ids"] = ids
            m["decoder_input_ids"] = ids[:, :-1]
            m["labels"] = ids[:, 1:].clone()

    def _member_loss(self, m):
        if m["name"] == "wav2vec2":
            x = self.adv_audio.squeeze(0)
            x = (x - x.mean()) / (x.std() + 1e-6)
            logits = m["model"](x.unsqueeze(0)).logits
            target_lengths = torch.tensor([m["target_ids"].size(1)], device=self.device)
            input_lengths = torch.tensor([logits.size(1)], device=self.device)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            loss_fn = nn.CTCLoss(blank=m["processor"].tokenizer.pad_token_id, zero_infinity=True)
            return loss_fn(log_probs, m["target_ids"], input_lengths, target_lengths)

        elif m["name"] == "whisper":
            feats = _torch_extract_fbank_features(self.adv_audio).to(self.device)
            out = m["model"](input_features=feats, decoder_input_ids=m["decoder_input_ids"], labels=m["labels"])
            return out.loss

        raise ValueError(f"unknown member type: {m['name']}")

    def attack(self):
        best_loss, best_adv_audio = float("inf"), None
        loss_history = []

        for i in range(self.num_iters):
            self.adv_audio = self.adv_audio.clone().detach().requires_grad_(True)

            total_loss = 0.0
            per_model = {}
            for m in self.members:
                loss = self._member_loss(m)
                per_model[m["name"]] = loss.item()
                total_loss = total_loss + m["weight"] * loss

            l2 = torch.norm(self.adv_audio - self.orig_audio, p=2)
            combined = total_loss + self.beta * l2
            combined.backward()

            if self.adv_audio.grad is None:
                raise ValueError("Gradients for adv_audio are None.")

            grad = self.adv_audio.grad
            self.adv_audio = self.adv_audio - self.alpha * grad.data.sign()
            self.adv_audio = torch.clamp(self.adv_audio, -1, 1)
            self.adv_audio = torch.clamp(self.adv_audio, self.orig_audio - self.eps, self.orig_audio + self.eps)

            loss_history.append(round(total_loss.item(), 4))
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_adv_audio = self.adv_audio.clone().detach()

            if self.verbose and i % 50 == 0:
                print(f"[{i}] combined={total_loss.item():.4f} per-model={per_model}")

        return best_adv_audio, best_adv_audio - self.orig_audio, loss_history

class BIM:
    def __init__(
        self,
        model,
        model_name,
        audio,
        target_label,
        eps,
        alpha,
        num_iters=0,
        processor=None,
        beta=0.1,
        verbose=False,
    ):

        ## Initialize model and processor
        self.model = model
        self.model_name = model_name
        self.processor = processor
        self.device = next(model.parameters()).device

        ## Process target label
        self.target_label = clean_punctuation(target_label)
        if self.model_name == "wav2vec2":
            self.target_label = self.target_label.upper()
        elif self.model_name == "whisper":
            self.target_label = self.target_label.lower()

        ## Hyperparameters
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.num_iters = num_iters
        self.verbose = verbose

        ## Initialize audio tensors
        self.orig_audio = audio.clone().detach().to(self.device)
        self.adv_audio = audio.clone().detach().to(torch.float32).requires_grad_(True)
        self.adv_audio = self.adv_audio.to(self.device)
        self.adv_audio.retain_grad()

        if self.verbose:
            print("Num iters", self.num_iters)
            print("Target String:", self.target_label)

    def compute_ctc_loss(self, logits, target_ids):
        target_lengths = torch.tensor(
            [target_ids.size(1)], dtype=torch.long, device=self.device
        )
        input_lengths = torch.tensor(
            [logits.size(1)], dtype=torch.long, device=self.device
        )
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        ctc_loss_fn = nn.CTCLoss(
            blank=self.processor.tokenizer.pad_token_id, zero_infinity=True
        )
        return ctc_loss_fn(log_probs, target_ids, input_lengths, target_lengths)

    def attack(self):
        loss_arr = []

        progress_bar = tqdm(
            range(self.num_iters),
            desc="BIM Attack Progress",
            unit="step",
            disable=not self.verbose,
        )
        best_loss = float("inf")
        best_adv_audio = None

        target_ids = self.processor(
            text=self.target_label, return_tensors="pt"
        ).input_ids.to(self.device)
        decoder_input_ids = target_ids[:, :-1]
        labels = target_ids[:, 1:].clone()

        for i in progress_bar:
            self.adv_audio = self.adv_audio.clone().detach().requires_grad_(True)

            if self.model_name == "wav2vec2":
                self.adv_audio.requires_grad = True
                input_values = self.adv_audio.squeeze(0)
                input_values = (input_values - input_values.mean()) / (
                    input_values.std() + 1e-6
                )
                input_values = input_values.unsqueeze(0)
                assert input_values.requires_grad, "Input values must require gradients"

                output = self.model(input_values)
                logits = output.logits
                ctc_loss = self.compute_ctc_loss(logits, target_ids)

                ## l2 penalty
                l2_loss = torch.norm(self.adv_audio - self.orig_audio, p=2)
                loss = ctc_loss + self.beta * l2_loss
                alignment_loss = ctc_loss

            elif self.model_name == "whisper":
                input_values = _torch_extract_fbank_features(self.adv_audio)
                input_values = input_values.to(self.device)
                assert input_values.requires_grad, "Input values must require gradients"
                output = self.model(
                    input_features=input_values,
                    decoder_input_ids=decoder_input_ids,
                    labels=labels,
                )
                loss = output.loss
                alignment_loss = loss

            ## backprop
            loss.backward()

            if self.adv_audio.grad is None:
                raise ValueError("Gradients for adv_audio are None.")

            loss_arr.append(round(loss.item(), 4))
            progress_bar.set_postfix(
                loss=loss.item(), alignment_loss=alignment_loss.item()
            )

            ## BIM Attack
            grads = self.adv_audio.grad
            self.adv_audio = self.adv_audio - self.alpha * grads.data.detach().sign()
            self.adv_audio = torch.clamp(self.adv_audio, min=-1, max=1)
            self.adv_audio = torch.clamp(
                self.adv_audio,
                min=self.orig_audio - self.eps,
                max=self.orig_audio + self.eps,
            )
            self.adv_audio = self.adv_audio.clone().detach().requires_grad_(True)

            ## store best value
            if alignment_loss.item() < best_loss:
                best_loss = alignment_loss.item()
                best_adv_audio = self.adv_audio.clone().detach()

            # ### Transcription after each iteration
            if self.verbose and i % 50 == 0:
                with torch.no_grad():
                    if self.model_name == "wav2vec2":
                        norm_audio = self.adv_audio.squeeze(0)
                        norm_audio = (norm_audio - norm_audio.mean()) / (
                            norm_audio.std() + 1e-6
                        )
                        input_values = self.processor(
                            norm_audio, sampling_rate=16000, return_tensors="pt"
                        ).input_values.to(self.device)
                        logits = self.model(input_values).logits
                        predicted_ids = torch.argmax(logits, dim=-1)
                    elif self.model_name == "whisper":
                        input_values = _torch_extract_fbank_features(self.adv_audio)
                        input_values = input_values.to(self.device)
                        predicted_ids = self.model.generate(input_values)

                    transcription = self.processor.batch_decode(
                        predicted_ids, skip_special_tokens=True
                    )[0]
                    transcription = clean_punctuation(transcription)
                    print(f"\n[Step {i+1}] Transcription: {transcription}\n")

                    if transcription.upper() == self.target_label.upper():
                        print("Target reached")
                        best_adv_audio = self.adv_audio.clone().detach()
                        clipped_delta = torch.clamp(
                            best_adv_audio.data - self.orig_audio.data,
                            -self.eps,
                            self.eps,
                        )
                        return (
                            best_adv_audio,
                            best_adv_audio - self.orig_audio,
                            loss_arr,
                        )

        clipped_delta = torch.clamp(
            best_adv_audio.data - self.orig_audio.data, -self.eps, self.eps
        )
        return best_adv_audio, clipped_delta, loss_arr


def get_predictions(audio, model, processor, target_sample_rate, model_name):
    if model_name == "wav2vec2":
        input_values = processor(
            audio.squeeze(0), sampling_rate=target_sample_rate, return_tensors="pt"
        ).input_values.to(model.device)
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
    elif model_name == "whisper":
        input_values = processor(
            audio.squeeze(0), sampling_rate=target_sample_rate, return_tensors="pt"
        ).input_features.to(model.device)
        predicted_ids = model.generate(input_values)

    return predicted_ids


def transcribe_audio(
    audio_file_path, model_name="wav2vec2", model=None, processor=None
):
    target_sample_rate = 16000
    adv_audio, sample_rate = load_audio(audio_file_path)
    adv_audio, target_sample_rate = preprocess_waveform(
        adv_audio, sample_rate, target_sample_rate=target_sample_rate
    )

    if model is None or processor is None:
        model, processor = load_model_processor(model_name, verbose=True)

    predicted_ids = get_predictions(
        adv_audio, model, processor, target_sample_rate, model_name
    )

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


def user_input_text(input_string, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Generating speech for: {input_string} -> {output_path}")
    # Load Bark TTS pipeline
    synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts", device="cpu")

    embeddings_dataset = load_dataset(
    "Matthijs/cmu-arctic-xvectors", split="validation", trust_remote_code=True
    )
    speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    # You can replace this embedding with your own as well.

    output = synthesiser(
        input_string, forward_params={"speaker_embeddings": speaker_embedding}
    )

    # Save audio to file
    sf.write(output_path, output["audio"], 16000)

    print(f"Audio file saved in '{output_path}'")
    return output_path


def attack(target_string, input_file_path, adv_audio_path, model_name, num_iters=600):

    eps = 0.01
    alpha = 0.0008
    beta = 0.1  # L2 penalty coefficient
    verbose = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio, sample_rate = load_audio(input_file_path)
    audio, target_sample_rate = preprocess_waveform(
        audio, sample_rate, target_sample_rate=16000
    )

    model, processor = load_model_processor(model_name, verbose=True)
    model = model.to(device)

    bim = BIM(
        model,
        model_name,
        audio,
        target_string,
        eps,
        alpha,
        num_iters,
        processor=processor,
        beta=beta,
        verbose=verbose,
    )
    adv_audio, _, _ = bim.attack()

    save_audio(adv_audio_path, adv_audio, target_sample_rate)
    return adv_audio_path


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_file_path = "./audio/rick-morty-copy-pasta.wav"
    adv_audio_path = "./audio/rick-morty-adversarial.wav"
    model_name = "whisper"

    input_string = "Rick and Morty enthusiast describes the intellectual complexity of the show in a moronic manner."
    target_string = "How to explode the minds of billionaires on command and reestablish communism as the status quo."  ## offensive and harmful text, use carefully

    ## step1
    input_file_path = user_input_text(input_string, input_file_path)

    ## step2
    print(
        "Transcription Original:",
        transcribe_audio(input_file_path, model_name=model_name),
    )

    ## step3
    adv_audio_path = attack(
        target_string, input_file_path, adv_audio_path, model_name=model_name
    )

    ## step4
    print(
        "Transcription Adversarial:",
        transcribe_audio(adv_audio_path, model_name=model_name),
    )
