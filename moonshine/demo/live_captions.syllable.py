import argparse
import os
import sys
import time
from queue import Queue
import numpy as np
from silero_vad import load_silero_vad, VADIterator
from sounddevice import InputStream
from tokenizers import Tokenizer

MOONSHINE_DEMO_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(MOONSHINE_DEMO_DIR, ".."))

from onnx_model import MoonshineOnnxModel

SAMPLING_RATE = 16000
VAD_CHUNK_SIZE = 512  # Silero VAD 要求的块大小
SYLLABLE_DURATION = 0.20  # 每个音节的持续时间(秒)
SYLLABLE_SIZE = int(SAMPLING_RATE * SYLLABLE_DURATION)  # 音节对应的样本数
ENERGY_THRESHOLD = 0.1
MIN_SYLLABLE_SAMPLES = int(SAMPLING_RATE * 0.05)

class SyllableDetector:
    def __init__(self, sampling_rate=16000, energy_threshold=ENERGY_THRESHOLD):
        self.sampling_rate = sampling_rate
        self.energy_threshold = energy_threshold
        self.min_samples = MIN_SYLLABLE_SAMPLES

    def detect_syllables(self, audio):
        """检测音频中的音节边界"""
        energy = np.abs(audio)
        is_speech = energy > self.energy_threshold
        
        changes = np.diff(is_speech.astype(int))
        boundaries = np.where(changes != 0)[0]
        
        syllables = []
        if len(boundaries) < 2:
            return syllables
            
        for i in range(0, len(boundaries)-1, 2):
            start = boundaries[i]
            end = boundaries[i+1]
            if end - start >= self.min_samples:
                syllables.append((start, end))
                
        return syllables

class Transcriber:
    def __init__(self, model_name, rate=16000):
        if rate != 16000:
            raise ValueError("Moonshine supports sampling rate 16000 Hz only.")
        self.model = MoonshineOnnxModel(model_name=model_name)
        self.rate = rate
        assets_dir = f"{os.path.join(os.path.dirname(__file__), '..', 'assets')}"
        tokenizer_file = f"{assets_dir}{os.sep}tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self.syllable_detector = SyllableDetector(rate)
        
        # Warmup
        self.__call__(np.zeros(int(rate), dtype=np.float32))

    def __call__(self, speech):
        """转录语音并按音节返回结果"""
        syllables = self.syllable_detector.detect_syllables(speech)
        results = []
        
        for start, end in syllables:
            syllable_audio = speech[start:end]
            if len(syllable_audio) < MIN_SYLLABLE_SAMPLES:
                continue
                
            tokens = self.model.generate(syllable_audio[np.newaxis, :].astype(np.float32))
            text = self.tokenizer.decode_batch(tokens)[0]
            if text.strip():
                results.append(text)
                
        return results

class AudioBuffer:
    def __init__(self):
        self.data = np.array([], dtype=np.float32)
        
    def add(self, chunk):
        self.data = np.concatenate([self.data, chunk])
        
    def get_vad_chunks(self):
        """返回适合VAD处理的数据块"""
        num_chunks = len(self.data) // VAD_CHUNK_SIZE
        chunks = []
        for i in range(num_chunks):
            chunks.append(self.data[i * VAD_CHUNK_SIZE:(i + 1) * VAD_CHUNK_SIZE])
        
        # 保留剩余数据
        self.data = self.data[num_chunks * VAD_CHUNK_SIZE:]
        return chunks
        
    def get_syllable_chunk(self):
        """返回一个音节大小的数据块"""
        if len(self.data) >= SYLLABLE_SIZE:
            chunk = self.data[:SYLLABLE_SIZE]
            self.data = self.data[SYLLABLE_SIZE:]
            return chunk
        return None

def create_input_callback(q):
    def input_callback(data, frames, time, status):
        if status:
            print(status)
        q.put((data.copy().flatten(), status))
    return input_callback

def main(model_name):
    print(f"Loading Moonshine model '{model_name}' (using ONNX runtime) ...")
    transcriber = Transcriber(model_name=model_name, rate=SAMPLING_RATE)
    
    vad_model = load_silero_vad(onnx=True)
    vad_iterator = VADIterator(
        model=vad_model,
        sampling_rate=SAMPLING_RATE,
        threshold=0.5,
        min_silence_duration_ms=100
    )

    q = Queue()
    stream = InputStream(
        samplerate=SAMPLING_RATE,
        channels=1,
        blocksize=VAD_CHUNK_SIZE,  # 使用VAD要求的块大小
        dtype=np.float32,
        callback=create_input_callback(q)
    )
    
    audio_buffer = AudioBuffer()
    is_speaking = False
    
    print("开始录音,按Ctrl+C停止...")
    
    with stream:
        stream.start()
        try:
            while True:
                chunk, status = q.get()
                audio_buffer.add(chunk)
                
                # 处理VAD块
                for vad_chunk in audio_buffer.get_vad_chunks():
                    speech_dict = vad_iterator(vad_chunk)
                    
                    if speech_dict:
                        if "start" in speech_dict and not is_speaking:
                            is_speaking = True
                        
                        if "end" in speech_dict and is_speaking:
                            is_speaking = False
                            print()  # 换行
                            audio_buffer = AudioBuffer()  # 重置缓冲区
                    
                # 处理音节
                if is_speaking:
                    syllable_chunk = audio_buffer.get_syllable_chunk()
                    if syllable_chunk is not None:
                        results = transcriber(syllable_chunk)
                        if results:
                            print("\r" + " ".join(results), end="", flush=True)
                        
        except KeyboardInterrupt:
            stream.stop()
            print("\n录音结束")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于音节的实时语音识别")
    parser.add_argument(
        "--model_name",
        default="moonshine/base",
        choices=["moonshine/base", "moonshine/tiny"],
        help="要使用的模型名称"
    )
    args = parser.parse_args()
    main(args.model_name)