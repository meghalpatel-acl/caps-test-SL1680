import subprocess
import os
import numpy as np
import time
import threading
#from logger_config import logger

CARD_INDEX = 1


class AudioManager:
    def __init__(self, device=None, sample_rate=16000):
        self._sample_rate = sample_rate
        self.arecord_process = None
        self.audio_lock = threading.Lock()
        self._audio_control_names = []
        self._device = device or self._get_usb_audio_device()

    @property
    def device(self):
        """Get the current audio device."""
        return self._device

    @device.setter
    def device(self, new_device):
        """Set a new audio device."""
        self._device = new_device

    @property
    def sample_rate(self):
        """Get the current sample rate."""
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, new_sample_rate):
        """Set a new sample rate."""
        self._sample_rate = new_sample_rate

    def play(self, filename):
        """Play the audio file using the specified audio device."""
        global CARD_INDEX
        if not self._device:
            raise RuntimeError("Audio device not set.")
        print(f"Playing audio file: {filename} on device: {self._device}")
        self.audio_lock.acquire()
        try:
            # Mute mic
            subprocess.run(
                ["amixer", "-c", CARD_INDEX, "sset", self._audio_control_names[1], "nocap"], check=True
            )
            subprocess.run(["aplay", "-q", "-D", self._device, filename], check=True)
            time.sleep(0.2)
            # UnMute mic
            subprocess.run(
                ["amixer", "-c", CARD_INDEX, "sset", self._audio_control_names[1], "cap"], check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error playing audio: {e}")
            #self.audio_lock.release()
            raise RuntimeError("Issue while playing audio.")
        finally:
            self.audio_lock.release()
    
    def start_record(self, chunk_size=512):
        """Start the arecord subprocess."""
        if self.arecord_process:
            self.stop_arecord()
        command = f"arecord -D {self._device} -f S16_LE -r {self._sample_rate} -c 2"
        self.arecord_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=chunk_size,
            shell=True,
        )

    def stop_record(self):
        """Stop the arecord subprocess."""
        if self.arecord_process:
            self.arecord_process.terminate()
            self.arecord_process.wait()
            self.arecord_process = None

    def read(self, chunk_size=512):
        # print("Reading audio")
        """Read audio data from the arecord subprocess."""
        if not self.arecord_process:
            raise RuntimeError("arecord process not running.")

        while True:
            data = self.arecord_process.stdout.read(chunk_size * 4)
            #print("Audio data:",len(data))
            # logger.info("Audio data: %s", len(data),extra={"event": "AUDIO_DEBUG"})
            if not data:
                #print("No audio data")
                logger.info("No Audio data", len(data),extra={"event": "AUDIO_DEBUG"})
                break
            yield np.frombuffer(data, dtype=np.int16)[::2].astype(np.float32) / 32768.0
            # logger.info("Af yield",extra={"event": "AUDIO_DEBUG"})

    def wait_for_audio(self):
        """Wait until a USB audio device is available."""
        print("Waiting for audio device...")
        while True:
            process = os.popen("aplay -l | grep USB\\ Audio && sleep 0.5")
            output = process.read()
            process.close()
            if "USB Audio" in output:
                print(output)
                break

    def _get_amixer_simple_control (self, card_index):
        try:
            # 1. Run only the base amixer command
            amixer_result = subprocess.run(
                ['amixer', 'scontrols', '-c', card_index],
                capture_output=True,
                text=True,
                check=True
            )

            amixer_output = amixer_result.stdout

            # 2. Process the output string directly in Python
            for line in amixer_output.splitlines():
                # Find the text between single quotes
                if 'Simple mixer control' in line:
                    # Splits the line based on the single quote character (')
                    parts = line.split("'")
                    if len(parts) >= 2:
                        # The control name is the second element after splitting
                        control_name = parts[1]
                        self._audio_control_names.append(control_name)


            print("Extracted Control Names:", self._audio_control_names)

        except subprocess.CalledProcessError as e:
            print(f"Error executing amixer: {e}")
            print(f"Stderr: {e.stderr}")

    def _get_usb_audio_device(self):
        """Finds the audio device ID for a USB Audio device using `aplay -l`."""
        global CARD_INDEX
        self.wait_for_audio()
        try:
            result = subprocess.run(
                ["aplay", "-l"], capture_output=True, text=True, check=True
            )
            lines = result.stdout.splitlines()
            for line in lines:
                if "USB Audio" in line:
                    # Extract card and device numbers
                    card_line = line.split()
                    card_index = card_line[1][:-1]  # Removes trailing colon
                    CARD_INDEX = card_index
                    self._get_amixer_simple_control (card_index)
                    device_name = f"plughw:{card_index},0"
                    print(f"Found audio device: {device_name}")
                    #set speaker unmute
                    subprocess.run(
                        ["amixer", "-c", CARD_INDEX, "sset", self._audio_control_names[0], "unmute"], check=True
                    )
                    # Set speaker's volume
                    subprocess.run(
                        ["amixer", "-c", CARD_INDEX, "sset", self._audio_control_names[0], "75%"], check=True
                    )
                    return device_name
        except subprocess.CalledProcessError as e:
            print(f"Error running `aplay -l`: {e}")
            return None

        return "default"
