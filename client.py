import os
import sys
import shutil
import json
import asyncio
from multiprocessing import cpu_count
import numpy as np
import FreeSimpleGUI as sg
import sounddevice as sd

import websockets

os.environ["OMP_NUM_THREADS"] = "4"
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

import warnings  # noqa: E402

warnings.simplefilter("ignore")


# Load model and configuration
flag_vc = False
prompt_len = 3  # in seconds


def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)


current_dir = os.getcwd()
n_cpu = cpu_count()


class GUIConfig:
    def __init__(self) -> None:
        self.reference_audio_path: str = ""
        self.diffusion_steps: int = 10
        self.sr_type: str = "sr_model"
        self.block_time: float = 0.25  # s
        self.threhold: int = -60
        self.crossfade_time: float = 0.05
        self.extra_time_ce: float = 2.5
        self.extra_time: float = 0.5
        self.extra_time_right: float = 2.0
        self.inference_cfg_rate: float = 0.7
        self.sg_hostapi: str = ""
        self.sg_input_device: str = ""
        self.sg_output_device: str = ""
        self.samplerate: int = 44100
        self.channels: int = 2  # stereo


class GUI:
    def __init__(self) -> None:
        self.gui_config = GUIConfig()
        self.function = "vc"
        self.delay_time = 0
        self.hostapis = None
        self.input_devices = None
        self.output_devices = None
        self.input_devices_indices = None
        self.output_devices_indices = None
        self.stream = None
        self.websocket = None
        self.update_devices()
        self.launcher()

    def load(self):
        try:
            os.makedirs("configs/inuse", exist_ok=True)
            if not os.path.exists("configs/inuse/config.json"):
                shutil.copy("configs/config.json", "configs/inuse/config.json")
            with open("configs/inuse/config.json", "r") as j:
                data = json.load(j)
                data["sr_model"] = data["sr_type"] == "sr_model"
                data["sr_device"] = data["sr_type"] == "sr_device"
                if data["sg_hostapi"] in self.hostapis:
                    self.update_devices(hostapi_name=data["sg_hostapi"])
                    if (
                        data["sg_input_device"] not in self.input_devices
                        or data["sg_output_device"] not in self.output_devices
                    ):
                        self.update_devices()
                        data["sg_hostapi"] = self.hostapis[0]
                        data["sg_input_device"] = self.input_devices[
                            self.input_devices_indices.index(sd.default.device[0])
                        ]
                        data["sg_output_device"] = self.output_devices[
                            self.output_devices_indices.index(sd.default.device[1])
                        ]
                else:
                    data["sg_hostapi"] = self.hostapis[0]
                    data["sg_input_device"] = self.input_devices[
                        self.input_devices_indices.index(sd.default.device[0])
                    ]
                    data["sg_output_device"] = self.output_devices[
                        self.output_devices_indices.index(sd.default.device[1])
                    ]
        except:
            with open("configs/inuse/config.json", "w") as j:
                data = {
                    "sg_hostapi": self.hostapis[0],
                    "sg_wasapi_exclusive": False,
                    "sg_input_device": self.input_devices[
                        self.input_devices_indices.index(sd.default.device[0])
                    ],
                    "sg_output_device": self.output_devices[
                        self.output_devices_indices.index(sd.default.device[1])
                    ],
                    "sr_type": "sr_model",
                    "block_time": 0.3,
                    "crossfade_length": 0.04,
                    "extra_time_ce": 2.5,
                    "extra_time": 0.5,
                    "extra_time_right": 0.02,
                    "diffusion_steps": 10,
                    "inference_cfg_rate": 0.7,
                    "max_prompt_length": 3.0,
                }
                data["sr_model"] = data["sr_type"] == "sr_model"
                data["sr_device"] = data["sr_type"] == "sr_device"
        return data

    def launcher(self):
        data = self.load()
        sg.theme("LightBlue3")
        layout = [
            [
                sg.Frame(
                    title="Load reference audio",
                    layout=[
                        [
                            sg.Input(
                                default_text=data.get("reference_audio_path", ""),
                                key="reference_audio_path",
                            ),
                            sg.FileBrowse(
                                "choose an audio file",
                                initial_folder=os.path.join(os.getcwd(), "examples/reference"),
                                file_types=(
                                    (". wav"),
                                    (". mp3"),
                                    (". flac"),
                                    (". m4a"),
                                    (". ogg"),
                                    (". opus"),
                                ),
                            ),
                        ],
                    ],
                )
            ],
            [
                sg.Frame(
                    layout=[
                        [
                            sg.Text("Device type"),
                            sg.Combo(
                                self.hostapis,
                                key="sg_hostapi",
                                default_value=data.get("sg_hostapi", ""),
                                enable_events=True,
                                size=(20, 1),
                            ),
                            sg.Checkbox(
                                "WASAPI Exclusive Device",
                                key="sg_wasapi_exclusive",
                                default=data.get("sg_wasapi_exclusive", False),
                                enable_events=True,
                            ),
                        ],
                        [
                            sg.Text("Input Device"),
                            sg.Combo(
                                self.input_devices,
                                key="sg_input_device",
                                default_value=data.get("sg_input_device", ""),
                                enable_events=True,
                                size=(45, 1),
                            ),
                        ],
                        [
                            sg.Text("Output Device"),
                            sg.Combo(
                                self.output_devices,
                                key="sg_output_device",
                                default_value=data.get("sg_output_device", ""),
                                enable_events=True,
                                size=(45, 1),
                            ),
                        ],
                        [
                            sg.Button("Reload devices", key="reload_devices"),
                            sg.Radio(
                                "Use model sampling rate",
                                "sr_type",
                                key="sr_model",
                                default=data.get("sr_model", True),
                                enable_events=True,
                            ),
                            sg.Radio(
                                "Use device sampling rate",
                                "sr_type",
                                key="sr_device",
                                default=data.get("sr_device", False),
                                enable_events=True,
                            ),
                            sg.Text("Sampling rate:"),
                            sg.Text("", key="sr_stream"),
                        ],
                    ],
                    title="Sound Device",
                )
            ],
            [
                sg.Frame(
                    layout=[
                        [
                            sg.Text("Diffusion steps"),
                            sg.Slider(
                                range=(1, 30),
                                key="diffusion_steps",
                                resolution=1,
                                orientation="h",
                                default_value=data.get("diffusion_steps", 10),
                                enable_events=True,
                            ),
                        ],
                        [
                            sg.Text("Inference cfg rate"),
                            sg.Slider(
                                range=(0.0, 1.0),
                                key="inference_cfg_rate",
                                resolution=0.1,
                                orientation="h",
                                default_value=data.get("inference_cfg_rate", 0.7),
                                enable_events=True,
                            ),
                        ],
                        [
                            sg.Text("Max prompt length (s)"),
                            sg.Slider(
                                range=(1.0, 20.0),
                                key="max_prompt_length",
                                resolution=0.5,
                                orientation="h",
                                default_value=data.get("max_prompt_length", 3.0),
                                enable_events=True,
                            ),
                        ],
                    ],
                    title="Regular settings",
                ),
                sg.Frame(
                    layout=[
                        [
                            sg.Text("Block time"),
                            sg.Slider(
                                range=(0.04, 3.0),
                                key="block_time",
                                resolution=0.02,
                                orientation="h",
                                default_value=data.get("block_time", 1.0),
                                enable_events=True,
                            ),
                        ],
                        [
                            sg.Text("Crossfade length"),
                            sg.Slider(
                                range=(0.02, 0.5),
                                key="crossfade_length",
                                resolution=0.02,
                                orientation="h",
                                default_value=data.get("crossfade_length", 0.1),
                                enable_events=True,
                            ),
                        ],
                        [
                            sg.Text("Extra content encoder context time (left)"),
                            sg.Slider(
                                range=(0.5, 10.0),
                                key="extra_time_ce",
                                resolution=0.1,
                                orientation="h",
                                default_value=data.get("extra_time_ce", 5.0),
                                enable_events=True,
                            ),
                        ],
                        [
                            sg.Text("Extra DiT context time (left)"),
                            sg.Slider(
                                range=(0.5, 10.0),
                                key="extra_time",
                                resolution=0.1,
                                orientation="h",
                                default_value=data.get("extra_time", 5.0),
                                enable_events=True,
                            ),
                        ],
                        [
                            sg.Text("Extra context time (right)"),
                            sg.Slider(
                                range=(0.02, 10.0),
                                key="extra_time_right",
                                resolution=0.02,
                                orientation="h",
                                default_value=data.get("extra_time_right", 2.0),
                                enable_events=True,
                            ),
                        ],
                    ],
                    title="Performance settings",
                ),
            ],
            [
                sg.Button("Start Voice Conversion", key="start_vc"),
                sg.Button("Stop Voice Conversion", key="stop_vc"),
                sg.Radio(
                    "Input listening",
                    "function",
                    key="im",
                    default=False,
                    enable_events=True,
                ),
                sg.Radio(
                    "Voice Conversion",
                    "function",
                    key="vc",
                    default=True,
                    enable_events=True,
                ),
                sg.Text("Algorithm delay (ms):"),
                sg.Text("0", key="delay_time"),
                sg.Text("Inference time (ms):"),
                sg.Text("0", key="infer_time"),
            ],
        ]
        self.window = sg.Window("Seed-VC - GUI", layout=layout, finalize=True)
        self.event_handler()

    def event_handler(self):
        global flag_vc
        while True:
            event, values = self.window.read()
            if event == sg.WINDOW_CLOSED:
                self.stop_stream()
                exit()
            if event == "reload_devices" or event == "sg_hostapi":
                self.gui_config.sg_hostapi = values["sg_hostapi"]
                self.update_devices(hostapi_name=values["sg_hostapi"])
                if self.gui_config.sg_hostapi not in self.hostapis:
                    self.gui_config.sg_hostapi = self.hostapis[0]
                self.window["sg_hostapi"].Update(values=self.hostapis)
                self.window["sg_hostapi"].Update(value=self.gui_config.sg_hostapi)
                if (
                    self.gui_config.sg_input_device not in self.input_devices
                    and len(self.input_devices) > 0
                ):
                    self.gui_config.sg_input_device = self.input_devices[0]
                self.window["sg_input_device"].Update(values=self.input_devices)
                self.window["sg_input_device"].Update(value=self.gui_config.sg_input_device)
                if self.gui_config.sg_output_device not in self.output_devices:
                    self.gui_config.sg_output_device = self.output_devices[0]
                self.window["sg_output_device"].Update(values=self.output_devices)
                self.window["sg_output_device"].Update(value=self.gui_config.sg_output_device)
            if event == "start_vc" and not flag_vc:
                if self.initialize_variables(values):
                    # self.start_vc() # start voice conversion on server
                    print("Starting voice conversion")
                    print(f"Blocktime: {self.gui_config.block_time}")
                    print(f"Block frame: {self.block_frame}")
                    print(f"ZC: {self.zc}")

                    self.start_stream()
                    settings = {
                        "reference_audio_path": values["reference_audio_path"],
                        # "index_path": values["index_path"],
                        "sg_hostapi": values["sg_hostapi"],
                        "sg_wasapi_exclusive": values["sg_wasapi_exclusive"],
                        "sg_input_device": values["sg_input_device"],
                        "sg_output_device": values["sg_output_device"],
                        "sr_type": ["sr_model", "sr_device"][
                            [
                                values["sr_model"],
                                values["sr_device"],
                            ].index(True)
                        ],
                        # "threhold": values["threhold"],
                        "diffusion_steps": values["diffusion_steps"],
                        "inference_cfg_rate": values["inference_cfg_rate"],
                        "max_prompt_length": values["max_prompt_length"],
                        "block_time": values["block_time"],
                        "crossfade_length": values["crossfade_length"],
                        "extra_time_ce": values["extra_time_ce"],
                        "extra_time": values["extra_time"],
                        "extra_time_right": values["extra_time_right"],
                    }
                    with open("configs/inuse/config.json", "w") as j:
                        json.dump(settings, j)
                    if self.stream is not None:
                        self.delay_time = (
                            self.stream.latency[-1]
                            + values["block_time"]
                            + values["crossfade_length"]
                            + values["extra_time_right"]
                            + 0.01
                        )
                    self.window["sr_stream"].update(self.gui_config.samplerate)
                    self.window["delay_time"].update(int(np.round(self.delay_time * 1000)))
            elif event == "diffusion_steps":
                self.gui_config.diffusion_steps = values["diffusion_steps"]
            elif event == "inference_cfg_rate":
                self.gui_config.inference_cfg_rate = values["inference_cfg_rate"]
            elif event in ["vc", "im"]:
                self.function = event
            elif event == "stop_vc" or event != "start_vc":
                self.stop_stream()

    def initialize_variables(self, values):
        self.set_devices(values["sg_input_device"], values["sg_output_device"])
        self.gui_config.sg_hostapi = values["sg_hostapi"]
        self.gui_config.sg_wasapi_exclusive = values["sg_wasapi_exclusive"]
        self.gui_config.sg_input_device = values["sg_input_device"]
        self.gui_config.sg_output_device = values["sg_output_device"]
        self.gui_config.sr_type = ["sr_model", "sr_device"][
            [
                values["sr_model"],
                values["sr_device"],
            ].index(True)
        ]
        self.gui_config.diffusion_steps = int(10)
        self.gui_config.inference_cfg_rate = float("0.7")
        self.gui_config.max_prompt_length = float("3")
        self.gui_config.block_time = float("0.3")  # 0.54
        self.gui_config.crossfade_time = float("0.02")
        self.gui_config.extra_time_ce = float("2.5")
        self.gui_config.extra_time = float("0.5")
        self.gui_config.extra_time_right = float("0.02")

        self.zc = self.gui_config.samplerate // 50  # 44100 // 100 = 441
        print(f"Blocktime: {self.gui_config.block_time}")
        self.block_frame = (
            int(np.round(self.gui_config.block_time * self.gui_config.samplerate / self.zc))
            * self.zc
        )
        return True

    async def connect_websocket(self):
        try:
            self.websocket = await websockets.connect("ws://20.99.233.208:6006")
            return True
        except Exception as e:
            sg.popup(f"Failed to connect to server: {e}")
            return False

    def start_stream(self):
        global flag_vc
        if not flag_vc:
            flag_vc = True
            if "WASAPI" in self.gui_config.sg_hostapi and self.gui_config.sg_wasapi_exclusive:
                extra_settings = sd.WasapiSettings(exclusive=True)
            else:
                extra_settings = None

            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            if not self.loop.run_until_complete(self.connect_websocket()):
                return

            print("Starting stream")
            print(
                f"Blocksize: {self.block_frame}, Samplerate: {self.gui_config.samplerate}, Channels: {self.gui_config.channels}"
            )
            self.stream = sd.Stream(
                callback=self.audio_callback,
                blocksize=self.block_frame,
                samplerate=self.gui_config.samplerate,
                channels=self.gui_config.channels,
                dtype="float32",
                extra_settings=extra_settings,
            )
            self.stream.start()

    def stop_stream(self):
        global flag_vc
        if flag_vc:
            flag_vc = False
            if self.stream is not None:
                self.stream.abort()
                self.stream.close()
                self.stream = None
            if self.websocket is not None:
                self.loop.run_until_complete(self.websocket.close())
                self.websocket = None
            if self.loop is not None:
                self.loop.close()
                self.loop = None

    def audio_callback(self, indata: np.ndarray, outdata: np.ndarray, frames, times, status):
        """
        Audio block callback function that sends audio to server and receives processed audio
        """
        if status:
            print(status)

        try:
            print(indata.shape, outdata.shape)
            self.loop.run_until_complete(self.websocket.send(indata.tobytes()))

            processed_audio = self.loop.run_until_complete(self.websocket.recv())

            processed_array = np.frombuffer(processed_audio, dtype=np.float16)
            # processed_array = processed_array.reshape(-1, self.gui_config.channels)
            outdata[:] = processed_array
            return outdata

        except Exception as e:
            print(f"Error in audio callback: {e}")
            outdata.fill(0)
            return outdata

    def update_devices(self, hostapi_name=None):
        """Get input and output devices."""
        global flag_vc
        flag_vc = False
        sd._terminate()
        sd._initialize()
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        for hostapi in hostapis:
            for device_idx in hostapi["devices"]:
                devices[device_idx]["hostapi_name"] = hostapi["name"]
        self.hostapis = [hostapi["name"] for hostapi in hostapis]
        if hostapi_name not in self.hostapis:
            hostapi_name = self.hostapis[0]
        self.input_devices = [
            d["name"]
            for d in devices
            if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]
        self.output_devices = [
            d["name"]
            for d in devices
            if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]
        self.input_devices_indices = [
            d["index"] if "index" in d else d["name"]
            for d in devices
            if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]
        self.output_devices_indices = [
            d["index"] if "index" in d else d["name"]
            for d in devices
            if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]

    def set_devices(self, input_device, output_device):
        """set input and output devices."""
        sd.default.device[0] = self.input_devices_indices[self.input_devices.index(input_device)]
        sd.default.device[1] = self.output_devices_indices[self.output_devices.index(output_device)]
        printt("Input device: %s:%s", str(sd.default.device[0]), input_device)
        printt("Output device: %s:%s", str(sd.default.device[1]), output_device)

    def get_device_samplerate(self):
        return int(sd.query_devices(device=sd.default.device[0])["default_samplerate"])

    def get_device_channels(self):
        max_input_channels = sd.query_devices(device=sd.default.device[0])["max_input_channels"]
        max_output_channels = sd.query_devices(device=sd.default.device[1])["max_output_channels"]
        return min(max_input_channels, max_output_channels, 2)


if __name__ == "__main__":
    GUI()
