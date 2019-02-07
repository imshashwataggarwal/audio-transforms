test = False            # Set true for test
dropout = 0.05

lr = 0.001
batch_size = 32

restore_epoch = 0       # Restore Epoch to resume training
epochs = 1000     
display_interval = 8    # Logging Interval

model_base_path = ''    # Base Dir Path, where model will be saved. If on Colab, then something like: ./drive/My Drive/model/'
model_path = ''         # Path to Load saved model

num_classes = 10        # Adjust Num of classes according to number of instruments

# Music Instrument to int mapping, Adjust according to names of Instruments
m2i = {
   'name_of_inst_1' : 0,
   'name_of_inst_2' : 1,
   'name_of_inst_n' : 10,
}
