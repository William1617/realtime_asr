import pyaudio
import soundfile as sf
import numpy as np
def int16to_float(high,low):
    if(high>128):
        result=(high-256)*256+low
    else:
        result=high*256+low
    return result/32768

class Recoder_model():
    def __init__(self,chunk_size=1600):
        self.chunk_size=chunk_size
        self.p=pyaudio.PyAudio()
        self.stream=self.p.open(16000,1,pyaudio.paInt16,input=True,
                                frames_per_buffer=self.chunk_size*2)
    def start_record(self):
        if(self.stream.is_stopped()):
            self.stream.start_stream()
        
    def get_audio_chunk(self):
        audio_chunk=[]
        if(self.stream.is_active()):
            audio_frame=self.stream.read(self.chunk_size)
            for idx in range(self.chunk_size):
                sample=int16to_float(audio_frame[idx*2+1],audio_frame[idx*2],)
                audio_chunk.append(sample)

            
        return audio_chunk
    def stop_record(self):
        if(self.stream.is_active()):
            self.stream.stop_stream()
    def close_record(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

if __name__ == '__main__':
    recoder=Recoder_model(chunk_size=640)
    recoder.start_record()
    audio=[]
    for idx in range(100):
        data=recoder.get_audio_chunk()
        for i in range(640):
            audio.append(data[i])
    sf.write('./testout.wav',audio,16000)
