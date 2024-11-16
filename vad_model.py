import onnxruntime
import numpy as np
import soundfile as sf

class vad_model():
    def __init__(self,threshold=0.5,max_frame=5,block_len=1600):
        self.model=onnxruntime.InferenceSession('./models/silero_vad.onnx')
        self.threshold=threshold
        self.max_frame=max_frame
        self.triggerd=False
        sr=np.zeros(1)
        sr[0]=16000

        self.modelinput={}
        self. modelinput['input']=np.zeros((1,block_len))
        self. modelinput['sr'] =sr.astype('int64')
        self.modelinput['h']=np.zeros((2,1,64)).astype('float32')
        self.modelinput['c']=np.zeros((2,1,64)).astype('float32')

        self.frame_num=max_frame
        self.block_len=block_len
    
    def cal_vad(self,audio):
        in_block=np.reshape(audio,(1,self.block_len)).astype('float32')
        self.modelinput['input']=in_block
        modelout=self.model.run(None,self.modelinput)
        self.modelinput['h'] =modelout[1]
        self.modelinput['c'] =modelout[2]
        out=modelout[0]
        return(self.get_result(out))
    
    def get_result(self,model_out):
        result=True
        
        if(model_out>self.threshold):
            self.triggerd=True
            result=True
            self.frame_num=self.max_frame
        else:
            if(self.triggerd==False):
                result=False
            else:
                if(model_out>self.threshold-0.15):
                    result=True
                    self.frame_num=self.max_frame
                else:
                    self.frame_num -=1
                    if(self.frame_num<0):
                        self.frame_num=-1
                        self.triggerd=False
                        result=False
        
        return result

if __name__=='__main__':
    blocklen=1600
    vadmodel=vad_model(block_len=blocklen)
    in_name='./4.wav'
    audio,sf1=sf.read(in_name)
    frame_num=len(audio)//blocklen
    out_audio=np.zeros_like(audio)
    for idx in range(frame_num):
        in_frame=audio[idx*blocklen:idx*blocklen+blocklen]
        vad_result=vadmodel.cal_vad(in_frame)
        out_audio[idx*blocklen:idx*blocklen+blocklen]=in_frame*vad_result
    sf.write('./test1out.wav',out_audio,16000)

    
