test = False
dropout = 0.05

lr = 0.0006
batch_size = 32
r = 5

restore_epoch = 0
epochs = 1000
display_interval = 8
warmup_steps = 100

model_base_path = './drive/My Drive/model/'
model_path = ''
embed_path = './drive/My Drive/model/embeddings.pt'
encoder_path = './drive/My Drive/model/encoder.pth'


dataset = {
   'train': {
     'audio':'./drive/My Drive/data/{}/wav/*.wav',
    },
   'test': {
     'audio':'./drive/My Drive/data/{}/wav/*.wav',
    },
   'val': {
     'audio':'./drive/My Drive/data/{}/wav/*.wav',
    }
}


speakers = {
   'train': ['ABA', 'ASI', 'BWC', 'EBVS', 'HJK', 'LXC', 'MBMPS',  'NJS', 'RRBI', 'SKA', 'SVBI', 'TXHC', 'YDCK', 'YKWK', 'ZHAA', 'TNI', 'JMK', 'KSP', 'AWB', 'CLB', 'SLT', 'BDL', 'RMS'],
   'test': ['YBAA', 'ERMS'],
   'val': ['NCC', 'HKK']
}
