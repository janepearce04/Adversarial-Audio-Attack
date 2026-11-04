# batch_runner.py
import csv
from dataclasses import dataclass
from pathlib import Path

from main import BIM, load_audio, save_audio, preprocess_waveform
from utils import load_model_processor


@dataclass
class Job:
    input_path: str
    target_string: str
    output_path: str


def load_jobs(csv_path: str) -> list[Job]:
    with open(csv_path, newline="") as f:
        return [Job(**row) for row in csv.DictReader(f)]


class BatchRunner:
    def __init__(self, model_name: str, eps=0.01, alpha=0.0008, num_iters=600, beta=0.1, device=None):
        import torch
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model, self.processor = load_model_processor(model_name, verbose=False)
        self.model = self.model.to(self.device).eval()
        self.eps, self.alpha, self.num_iters, self.beta = eps, alpha, num_iters, beta

    def run(self, jobs: list[Job]):
        results = []
        for job in jobs:
            audio, sr = load_audio(job.input_path)
            audio, sr = preprocess_waveform(audio, sr, target_sample_rate=16000)

            bim = BIM(
                self.model, self.model_name, audio, job.target_string,
                self.eps, self.alpha, self.num_iters,
                processor=self.processor, beta=self.beta, verbose=False,
            )
            adv_audio, delta, loss_arr = bim.attack()

            Path(job.output_path).parent.mkdir(parents=True, exist_ok=True)
            save_audio(job.output_path, adv_audio, sr)
            results.append((job.input_path, job.output_path, loss_arr[-1] if loss_arr else None))
            print(f"done: {job.input_path} -> {job.output_path} (final loss {results[-1][2]})")
        return results


if __name__ == "__main__":
    runner = BatchRunner(model_name="whisper", num_iters=600)
    jobs = load_jobs("jobs.csv")  # columns: input_path,target_string,output_path
    runner.run(jobs)