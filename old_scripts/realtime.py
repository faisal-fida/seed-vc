import os
import sys
from dotenv import load_dotenv
import time

load_dotenv()

os.environ["OMP_NUM_THREADS"] = "4"
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)
import warnings
import yaml

warnings.simplefilter("ignore")

from modules.commons import *
import librosa
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from hf_utils import load_custom_model_from_hf

import torch
from modules.commons import str2bool

# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
device = None
flag_vc = False
prompt_condition, mel2, style2 = None, None, None
reference_wav_name = ""
prompt_len = 3  # in seconds
ce_dit_difference = 2.0  # 2 seconds
fp16 = False


@torch.no_grad()
def custom_infer(
    model_set,
    reference_wav,
    new_reference_wav_name,
    input_wav_res,
    block_frame_16k,
    skip_head,
    skip_tail,
    return_length,
    diffusion_steps,
    inference_cfg_rate,
    max_prompt_length,
    cd_difference=2.0,
):
    """
    The same custom_infer logic from the original code.
    """
    global prompt_condition, mel2, style2
    global reference_wav_name
    global prompt_len
    global ce_dit_difference

    (
        model,
        semantic_fn,
        vocoder_fn,
        campplus_model,
        to_mel,
        mel_fn_args,
    ) = model_set

    sr = mel_fn_args["sampling_rate"]
    hop_length = mel_fn_args["hop_size"]

    if ce_dit_difference != cd_difference:
        ce_dit_difference = cd_difference
        print(f"Setting ce_dit_difference to {cd_difference} seconds.")

    # If reference audio changed or new prompt length
    if (
        prompt_condition is None
        or reference_wav_name != new_reference_wav_name
        or prompt_len != max_prompt_length
    ):
        prompt_len = max_prompt_length
        print(f"Setting max prompt length to {max_prompt_length} seconds.")

        reference_wav = reference_wav[: int(sr * prompt_len)]
        reference_wav_tensor = torch.from_numpy(reference_wav).to(device)

        ori_waves_16k = torchaudio.functional.resample(reference_wav_tensor, sr, 16000)
        S_ori = semantic_fn(ori_waves_16k.unsqueeze(0))

        feat2 = kaldi.fbank(
            ori_waves_16k.unsqueeze(0),
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000,
        )
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = campplus_model(feat2.unsqueeze(0))

        mel2 = to_mel(reference_wav_tensor.unsqueeze(0))
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)
        prompt_condition = model.length_regulator(
            S_ori, ylens=target2_lengths, n_quantizers=3, f0=None
        )[0]

        reference_wav_name = new_reference_wav_name

    # Run semantic function on the new chunk
    converted_waves_16k = input_wav_res
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    S_alt = semantic_fn(converted_waves_16k.unsqueeze(0))
    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Time taken for semantic_fn: {elapsed_time_ms}ms")

    ce_dit_frame_difference = int(ce_dit_difference * 50)
    S_alt = S_alt[:, ce_dit_frame_difference:]
    target_lengths = torch.LongTensor(
        [(skip_head + return_length + skip_tail - ce_dit_frame_difference) / 50 * sr // hop_length]
    ).to(S_alt.device)
    print(f"target_lengths: {target_lengths}")
    cond = model.length_regulator(S_alt, ylens=target_lengths, n_quantizers=3, f0=None)[0]
    cat_condition = torch.cat([prompt_condition, cond], dim=1)

    with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):
        vc_target = model.cfm.inference(
            cat_condition,
            torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
            mel2,
            style2,
            None,
            n_timesteps=diffusion_steps,
            inference_cfg_rate=inference_cfg_rate,
        )
        vc_target = vc_target[:, :, mel2.size(-1) :]
        print(f"vc_target.shape: {vc_target.shape}")
        vc_wave = vocoder_fn(vc_target).squeeze()

    output_len = return_length * sr // 50
    tail_len = skip_tail * sr // 50
    output = vc_wave[-output_len - tail_len : -tail_len]
    return output


def load_models(args):
    """
    The same load_models logic from the original code.
    """
    global fp16
    fp16 = args.fp16
    print(f"Using fp16: {fp16}")

    if args.checkpoint_path is None or args.checkpoint_path == "":
        dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
            "Plachta/Seed-VC",
            "DiT_uvit_tat_xlsr_ema.pth",
            "config_dit_mel_seed_uvit_xlsr_tiny.yml",
        )
    else:
        dit_checkpoint_path = args.checkpoint_path
        dit_config_path = args.config_path

    config = yaml.safe_load(open(dit_config_path, "r"))
    model_params = recursive_munch(config["model_params"])
    model_params.dit_type = "DiT"
    model = build_model(model_params, stage="DiT")
    hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
    sr = config["preprocess_params"]["sr"]

    # Load checkpoints
    model, _, _, _ = load_checkpoint(
        model,
        None,
        dit_checkpoint_path,
        load_only_params=True,
        ignore_modules=[],
        is_distributed=False,
    )
    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    # Load additional modules
    from modules.campplus.DTDNN import CAMPPlus

    campplus_ckpt_path = load_custom_model_from_hf(
        "funasr/campplus", "campplus_cn_common.bin", config_filename=None
    )
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model.eval()
    campplus_model.to(device)

    vocoder_type = model_params.vocoder.type
    if vocoder_type == "bigvgan":
        from modules.bigvgan import bigvgan

        bigvgan_name = model_params.vocoder.name
        bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
        bigvgan_model.remove_weight_norm()
        bigvgan_model = bigvgan_model.eval().to(device)
        vocoder_fn = bigvgan_model
    elif vocoder_type == "hifigan":
        from modules.hifigan.generator import HiFTGenerator
        from modules.hifigan.f0_predictor import ConvRNNF0Predictor

        hift_config = yaml.safe_load(open("configs/hifigan.yml", "r"))
        hift_gen = HiFTGenerator(
            **hift_config["hift"], f0_predictor=ConvRNNF0Predictor(**hift_config["f0_predictor"])
        )
        hift_path = load_custom_model_from_hf("FunAudioLLM/CosyVoice-300M", "hift.pt", None)
        hift_gen.load_state_dict(torch.load(hift_path, map_location="cpu"))
        hift_gen.eval()
        hift_gen.to(device)
        vocoder_fn = hift_gen
    elif vocoder_type == "vocos":
        vocos_config = yaml.safe_load(open(model_params.vocoder.vocos.config, "r"))
        vocos_path = model_params.vocoder.vocos.path
        vocos_model_params = recursive_munch(vocos_config["model_params"])
        vocos = build_model(vocos_model_params, stage="mel_vocos")
        vocos_checkpoint_path = vocos_path
        vocos, _, _, _ = load_checkpoint(
            vocos,
            None,
            vocos_checkpoint_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        _ = [vocos[key].eval().to(device) for key in vocos]
        _ = [vocos[key].to(device) for key in vocos]
        total_params = sum(
            sum(p.numel() for p in vocos[key].parameters() if p.requires_grad)
            for key in vocos.keys()
        )
        print(f"Vocoder model total parameters: {total_params / 1_000_000:.2f}M")
        vocoder_fn = vocos.decoder
    else:
        raise ValueError(f"Unknown vocoder type: {vocoder_type}")

    speech_tokenizer_type = model_params.speech_tokenizer.type
    if speech_tokenizer_type == "whisper":
        from transformers import AutoFeatureExtractor, WhisperModel

        whisper_name = model_params.speech_tokenizer.name
        whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(
            device
        )
        del whisper_model.decoder
        whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

        def semantic_fn(waves_16k):
            ori_inputs = whisper_feature_extractor(
                [waves_16k.squeeze(0).cpu().numpy()],
                return_tensors="pt",
                return_attention_mask=True,
            )
            ori_input_features = whisper_model._mask_input_features(
                ori_inputs.input_features, attention_mask=ori_inputs.attention_mask
            ).to(device)
            with torch.no_grad():
                ori_outputs = whisper_model.encoder(
                    ori_input_features.to(whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            S_ori = ori_outputs.last_hidden_state.to(torch.float32)
            S_ori = S_ori[:, : waves_16k.size(-1) // 320 + 1]
            return S_ori

    elif speech_tokenizer_type == "cnhubert":
        from transformers import Wav2Vec2FeatureExtractor, HubertModel

        hubert_model_name = config["model_params"]["speech_tokenizer"]["name"]
        hubert_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model_name)
        hubert_model = HubertModel.from_pretrained(hubert_model_name)
        hubert_model = hubert_model.to(device)
        hubert_model = hubert_model.eval()
        hubert_model = hubert_model.half()

        def semantic_fn(waves_16k):
            ori_waves_16k_input_list = [
                waves_16k[bib].cpu().numpy() for bib in range(len(waves_16k))
            ]
            ori_inputs = hubert_feature_extractor(
                ori_waves_16k_input_list,
                return_tensors="pt",
                return_attention_mask=True,
                padding=True,
                sampling_rate=16000,
            ).to(device)
            with torch.no_grad():
                ori_outputs = hubert_model(
                    ori_inputs.input_values.half(),
                )
            S_ori = ori_outputs.last_hidden_state.float()
            return S_ori

    elif speech_tokenizer_type == "xlsr":
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

        model_name = config["model_params"]["speech_tokenizer"]["name"]
        output_layer = config["model_params"]["speech_tokenizer"]["output_layer"]
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
        # Truncate layers
        wav2vec_model.encoder.layers = wav2vec_model.encoder.layers[:output_layer]
        wav2vec_model = wav2vec_model.to(device)
        wav2vec_model = wav2vec_model.eval()
        wav2vec_model = wav2vec_model.half()

        def semantic_fn(waves_16k):
            ori_waves_16k_input_list = [
                waves_16k[bib].cpu().numpy() for bib in range(len(waves_16k))
            ]
            ori_inputs = wav2vec_feature_extractor(
                ori_waves_16k_input_list,
                return_tensors="pt",
                return_attention_mask=True,
                padding=True,
                sampling_rate=16000,
            ).to(device)
            with torch.no_grad():
                ori_outputs = wav2vec_model(
                    ori_inputs.input_values.half(),
                )
            S_ori = ori_outputs.last_hidden_state.float()
            return S_ori

    else:
        raise ValueError(f"Unknown speech tokenizer type: {speech_tokenizer_type}")

    # Generate mel spectrograms
    mel_fn_args = {
        "n_fft": config["preprocess_params"]["spect_params"]["n_fft"],
        "win_size": config["preprocess_params"]["spect_params"]["win_length"],
        "hop_size": config["preprocess_params"]["spect_params"]["hop_length"],
        "num_mels": config["preprocess_params"]["spect_params"]["n_mels"],
        "sampling_rate": sr,
        "fmin": config["preprocess_params"]["spect_params"].get("fmin", 0),
        "fmax": None
        if config["preprocess_params"]["spect_params"].get("fmax", "None") == "None"
        else 8000,
        "center": False,
    }
    from modules.audio import mel_spectrogram

    to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

    return (
        model,
        semantic_fn,
        vocoder_fn,
        campplus_model,
        to_mel,
        mel_fn_args,
    )


def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)


# ----------------------------------------------------------------------------
# AudioProcessor: Maintains the same chunk-based logic from the original code
# but used inside a WebSocket server
# ----------------------------------------------------------------------------
import numpy as np
import torch.nn.functional as F
import torchaudio.transforms as tat


class AudioProcessor:
    """
    This class encapsulates the real-time chunk processing logic
    (VAD, crossfade, SOLA, inference) that was previously in the GUI callback.
    """

    def __init__(
        self,
        model_set,
        vad_model,
        reference_wav_path,
        block_time=0.25,
        crossfade_time=0.05,
        extra_time_ce=2.5,
        extra_time=0.5,
        extra_time_right=2.0,
        diffusion_steps=10,
        inference_cfg_rate=0.7,
        max_prompt_length=3.0,
    ):
        # Store references
        self.model_set = model_set
        self.vad_model = vad_model
        self.block_time = block_time
        self.crossfade_time = crossfade_time
        self.extra_time_ce = extra_time_ce
        self.extra_time = extra_time
        self.extra_time_right = extra_time_right
        self.diffusion_steps = diffusion_steps
        self.inference_cfg_rate = inference_cfg_rate
        self.max_prompt_length = max_prompt_length

        # Load reference waveform
        # Use model samplerate from mel_fn_args
        self.model_sr = self.model_set[-1]["sampling_rate"]
        self.reference_wav, _ = librosa.load(reference_wav_path, sr=self.model_sr)

        # Some default chunk settings (these can be tuned or made configurable)
        self.samplerate = 16000  # We'll assume the inbound audio is 16k, or adapt as needed
        self.channels = 1

        # Derived parameters
        self.zc = self.samplerate // 50  # e.g. block size for 20ms
        self.block_frame = int(np.round(self.block_time * self.samplerate / self.zc)) * self.zc
        self.block_frame_16k = 320 * self.block_frame // self.zc
        self.crossfade_frame = (
            int(np.round(self.crossfade_time * self.samplerate / self.zc)) * self.zc
        )
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = int(np.round(self.extra_time_ce * self.samplerate / self.zc)) * self.zc
        self.extra_frame_right = (
            int(np.round(self.extra_time_right * self.samplerate / self.zc)) * self.zc
        )

        # Buffers
        self.input_wav = torch.zeros(
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame
            + self.extra_frame_right,
            device=device,
            dtype=torch.float32,
        )
        self.input_wav_res = torch.zeros(
            320 * self.input_wav.shape[0] // self.zc, device=device, dtype=torch.float32
        )

        self.rms_buffer = np.zeros(4 * self.zc, dtype="float32")
        self.sola_buffer = torch.zeros(self.sola_buffer_frame, device=device, dtype=torch.float32)

        self.skip_head = self.extra_frame // self.zc
        self.skip_tail = self.extra_frame_right // self.zc
        self.return_length = (
            self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        ) // self.zc

        self.fade_in_window = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(
                    0.0, 1.0, steps=self.sola_buffer_frame, device=device, dtype=torch.float32
                )
            )
            ** 2
        )
        self.fade_out_window = 1 - self.fade_in_window

        self.resampler_16k = tat.Resample(
            orig_freq=self.samplerate, new_freq=16000, dtype=torch.float32
        ).to(device)

        # If model SR != self.samplerate, we resample back up to model SR for output
        if self.model_sr != self.samplerate:
            self.resampler_model_sr = tat.Resample(
                orig_freq=self.model_sr, new_freq=self.samplerate, dtype=torch.float32
            ).to(device)
        else:
            self.resampler_model_sr = None

        # VAD cache
        self.vad_cache = {}
        self.vad_chunk_size = int(1000 * self.block_time)
        self.vad_speech_detected = False
        self.set_speech_detected_false_at_end_flag = False

    def process_audio_chunk(self, indata: np.ndarray) -> np.ndarray:
        start_time = time.perf_counter()

        # Convert to mono if needed
        if indata.ndim > 1 and indata.shape[0] > 1:
            indata = librosa.to_mono(indata)
        else:
            indata = indata.flatten()

        # 1) VAD
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()

        # For VAD, assume inbound is self.samplerate, resample to 16k if different
        if self.samplerate != 16000:
            indata_16k = librosa.resample(indata, orig_sr=self.samplerate, target_sr=16000)
        else:
            indata_16k = indata

        res = self.vad_model.generate(
            input=indata_16k, cache=self.vad_cache, is_final=False, chunk_size=self.vad_chunk_size
        )
        res_value = res[0]["value"]
        if len(res_value) % 2 == 1 and not self.vad_speech_detected:
            self.vad_speech_detected = True
        elif len(res_value) % 2 == 1 and self.vad_speech_detected:
            self.set_speech_detected_false_at_end_flag = True

        end_event.record()
        torch.cuda.synchronize()  # Wait for the events to be recorded!
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"Time taken for VAD: {elapsed_time_ms}ms")

        # 2) Shift buffer
        self.input_wav[: -self.block_frame] = self.input_wav[self.block_frame :].clone()
        sf.write("output.wav", self.input_wav.cpu().numpy(), 22050)
        self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(device)

        # Prepare 16k buffer for inference
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[
            self.block_frame_16k :
        ].clone()
        # Resample only the newly added chunk plus a bit of overlap
        block_and_overlap = self.input_wav[-indata.shape[0] - 2 * self.zc :]
        self.input_wav_res[-320 * (indata.shape[0] // self.zc + 1) :] = self.resampler_16k(
            block_and_overlap
        )[320:]

        # 3) Inference
        if self.extra_time_ce - self.extra_time < 0:
            raise ValueError("Content encoder extra context must be >= DiT extra context!")

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()

        infer_wav = custom_infer(
            self.model_set,
            self.reference_wav,
            reference_wav_name,  # The original code uses the path as a "name"
            self.input_wav_res,
            self.block_frame_16k,
            self.skip_head,
            self.skip_tail,
            self.return_length,
            int(self.diffusion_steps),
            self.inference_cfg_rate,
            self.max_prompt_length,
            self.extra_time_ce - self.extra_time,
        )

        if self.resampler_model_sr is not None:
            infer_wav = self.resampler_model_sr(infer_wav)

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"Time taken for VC: {elapsed_time_ms}ms")

        # If no speech, we zero the output
        if not self.vad_speech_detected:
            infer_wav = torch.zeros_like(self.input_wav[self.extra_frame :])

        # 4) SOLA
        conv_input = infer_wav[None, None, : self.sola_buffer_frame + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(conv_input**2, torch.ones(1, 1, self.sola_buffer_frame, device=device)) + 1e-8
        )

        if sys.platform == "darwin":
            _, sola_offset = torch.max(cor_nom[0, 0] / cor_den[0, 0])
            sola_offset = sola_offset.item()
        else:
            sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0]).item()

        print(f"sola_offset = {int(sola_offset)}")

        infer_wav = infer_wav[sola_offset:]
        infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
        infer_wav[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out_window

        self.sola_buffer[:] = infer_wav[
            self.block_frame : self.block_frame + self.sola_buffer_frame
        ]
        output_chunk = infer_wav[: self.block_frame]

        # Prepare final output as float32 NumPy
        outdata = output_chunk.cpu().numpy()

        # 5) If we had set the speech-detected false at the end
        if self.set_speech_detected_false_at_end_flag:
            self.vad_speech_detected = False
            self.set_speech_detected_false_at_end_flag = False

        total_time = time.perf_counter() - start_time
        print(f"Infer time: {total_time:.2f} seconds")

        # Return the processed chunk
        return outdata


from funasr import AutoModel
import argparse
import soundfile as sf

parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint-path", type=str, default=None, help="Path to the model checkpoint"
)
parser.add_argument("--config-path", type=str, default=None, help="Path to the vocoder checkpoint")
parser.add_argument(
    "--fp16", type=str2bool, nargs="?", const=True, help="Whether to use fp16", default=True
)
parser.add_argument("--gpu", type=int, help="Which GPU id to use", default=0)
args = parser.parse_args()

cuda_target = f"cuda:{args.gpu}" if args.gpu else "cuda"
device = torch.device(cuda_target if torch.cuda.is_available() else "cpu")

model_set = load_models(args)

vad_model = AutoModel(model="fsmn-vad", model_revision="v2.0.4")

reference_audio_path = os.path.join(os.getcwd(), "examples/reference/s1p1.wav")


processor = AudioProcessor(
    model_set=model_set,
    vad_model=vad_model,
    reference_wav_path=reference_audio_path,
    block_time=0.25,  # 250ms
    crossfade_time=0.05,  # 50ms
    extra_time_ce=2.5,
    extra_time=0.5,
    extra_time_right=2.0,
    diffusion_steps=10,
    inference_cfg_rate=0.7,
    max_prompt_length=3.0,
)


audio = sf.SoundFile("examples/source/source_s1.wav").read(dtype="float32")

output = processor.process_audio_chunk(audio)
