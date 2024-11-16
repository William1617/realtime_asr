import threading
from asr_model import ASR_model
import soundfile as sf
from recorder import Recoder_model

import time
asrmodel=ASR_model(block_len=1600)
def decode():
    for idx in range(1000):
        
        asrmodel.decode()
        time.sleep(0.01)
        if(asrmodel.is_end()==True):
            break
    
if __name__ == '__main__':
    
    p1 = threading.Thread(target=decode, args=())
    p1.start()
    recoder=Recoder_model()
    recoder.start_record()
    
    pre_len=2
    for idx in range(100):
        data=recoder.get_audio_chunk()
        out_reusly=asrmodel.accept_audio(data)
        if(len(out_reusly)!=pre_len):
            print(out_reusly)
        pre_len=len(out_reusly)
    recoder.stop_record()
    asrmodel.set_end()
    p1.join()
    
