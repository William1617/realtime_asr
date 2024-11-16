
import onnxruntime
import kaldi_native_fbank as knf

from threading import Lock
import queue

import numpy as np
from vad_model import vad_model

class ASR_model():
    def __init__(self,threshold=0.5,max_frame=5,block_len=640):
        with open('./tokens.txt','r',encoding='UTF-8') as char_map_str:
            self.index_map = {}
            for line in char_map_str:
                ch, index = line.split()
                self.index_map[int(index)] = ch
      
     
        self.lock=Lock()
        self.encoder=onnxruntime.InferenceSession('./models/encoder.onnx')
        self.encoderl_input_name= [inp.name for inp in self.encoder.get_inputs()]

        self.encoder_input={}
        inp = self.encoder.get_inputs()
        for idx in range(36):
            if(idx>0.1 and idx<5.1):
                self.encoder_input[self.encoderl_input_name[idx]] =np.zeros(inp[idx].shape).astype('int64')
            else:
                self.encoder_input[self.encoderl_input_name[idx]] =np.zeros(inp[idx].shape).astype('float32')
        
        self.decoder=onnxruntime.InferenceSession('./models/decoder.onnx')
        self.decoderl_input_name= [inp.name for inp in self.decoder.get_inputs()]
        self.decoder_input = {
            inp.name: np.zeros(
                [dim if isinstance(dim, int) else 1 for dim in inp.shape],
                dtype=int)
            for inp in self.decoder.get_inputs()}
        
        self.joiner=onnxruntime.InferenceSession('./models/joiner.onnx')
        self.joiner_input_name= [inp.name for inp in self.joiner.get_inputs()]
        self.joiner_input = {
            inp.name: np.zeros(
                [dim if isinstance(dim, int) else 1 for dim in inp.shape],
                dtype=np.float32)
            for inp in self.joiner.get_inputs()}
    
        opts = knf.FbankOptions()
        opts.frame_opts.dither = 0
        opts.mel_opts.num_bins = 80
        opts.frame_opts.snip_edges = False
        opts.mel_opts.debug_mel = False
        self.fbank = knf.OnlineFbank(opts)

        self.hyps=[]
        self.hyps.append(0)
        self.hyps.append(0)
        self.num_silence=0
        self.audio_queue=queue.Queue()
        self.reset_flag=False
        self.vadmodel=vad_model(threshold=threshold, max_frame=max_frame,block_len=block_len)
        self.pre_audio=np.zeros(block_len)
        self.pre_vadflag=False
        self.end_flag=False

        self.frame_start=0
        self.pre_size=2
        self.result=''
    
    def greedy_search(self,encoder_out):
        decoder_in=np.zeros((1,2)).astype('int64')
        decoder_in[0,0]=self.hyps[-2]
        decoder_in[0,1]=self.hyps[-1]
        self.decoder_input[self.decoderl_input_name[0]]=decoder_in
        decodr_out = self.decoder.run(None,self.decoder_input)[0]
        for t in range(8):
            current_encoder_out = encoder_out[:, t , :]
            self.joiner_input[self.joiner_input_name[0]]=current_encoder_out
            self.joiner_input[self.joiner_input_name[1]]=decodr_out
            logits=self.joiner.run(None,self.joiner_input)[0]
            y = np.argmax(logits)
            
            if y != 0:
                self.hyps.append(y)
                decoder_in[0,0]=self.hyps[-2]
                decoder_in[0,1]=self.hyps[-1]
                self.decoder_input[self.decoderl_input_name[0]]=decoder_in
                decodr_out=self.decoder.run(None,self.decoder_input)[0]
                    
    def get_result(self):
        self.lock.acquire()
        if(len(self.hyps)>self.pre_size):
            for idx in range(self.pre_size,len(self.hyps)):
                if(self.hyps[idx]>0):
                    self.result +=self.index_map.get(self.hyps[idx])
                else:
                    self.result +=','
    
        self.pre_size=len(self.hyps)
        
        self.lock.release()
        if(len(self.hyps)>100):
            self.hyps=[]
            self.hyps.append(0)
            self.hyps.append(0)
        
    
    
    def Resetout(self):
        self.lock.acquire()
        self.hyps=[]
        self.hyps.append(0)
        self.hyps.append(0)
        self.reset_flag=False
        self.result =''
        self.lock.release()    
    
    def accept_audio(self,audio):
        self.lock.acquire()
        #prevent overflow
        if(self.audio_queue.qsize()<30):
            self.audio_queue.put(audio)
        out_result=self.result
        self.lock.release()
        return out_result
    def extract_audio(self):
        self.lock.acquire()
        if(self.audio_queue.qsize()>0):
            audio_frame=self.audio_queue.get()
            vad_result=self.vadmodel.cal_vad(audio_frame)
            if(vad_result):
                if(self.pre_vadflag==False):
                    self.fbank.accept_waveform(16000,self.pre_audio)
                self.fbank.accept_waveform(16000,audio_frame)
            else:
                if(self.pre_vadflag):
                    self.reset_flag=True
            self.pre_audio=audio_frame
            self.pre_vadflag=vad_result      
        self.lock.release()

    def decode(self):
        self.extract_audio()
        
        is_ready=(self.fbank.num_frames_ready-self.frame_start)>39
       
        while(is_ready):
           
            encoder_in=np.zeros((1,39,80)).astype('float32')
            for j in range(39):
                
                in_fatures = self.fbank.get_frame(self.frame_start+j)
                encoder_in[:,j,:]=in_fatures.astype('float32')

            self.encoder_input[self.encoderl_input_name[0]]=encoder_in
            encoder_out=self.encoder.run(None,self.encoder_input)
            for id2 in range(35):
                self.encoder_input[self.encoderl_input_name[id2+1]] = encoder_out[id2+1]
            self.greedy_search(encoder_out[0])
            self.frame_start +=32
          
            is_ready=(self.fbank.num_frames_ready-self.frame_start)>39
        self.get_result()
        if(self.reset_flag):
            self.Resetout()
    
    def set_end(self):
        self.lock.acquire()
        self.end_flag=True
        self.lock.release()
    
    def is_end(self):
        self.lock.acquire()
        end_flag=self.end_flag
        self.lock.release()
        return end_flag

    

            
            

   


