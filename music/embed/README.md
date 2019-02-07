# Instructions to Run


### 1. Install Requirements:  

    Locally, 

    pip install -r requirements.txt

    On Colab,

    !pip install torch torchvision
    !pip install pydub
    !pip install librosa

### 2. Create `labels.txt` file with each line: 

    wav_file_path<space>instrument_name<newline>

### 3. Modify hp.py:
    - Adjust Model Paths
    
    - Create m2i dictionary, with key as the instrument name and value as the corrasponding integer mapping
    
    - Adjust num_classes according to number of musical instruments

### 4. For Training:

    - For training on local machine run:

        python main.py

    - For training on colab:

        - Copy main.py code on colab
        
        - Add following lines after importing sys library:

            > sys.path.append('PATH TO THIS FOLDER')
            > sys.path.append('PATH TO THIS FOLDER/scripts')

            e.g.

            > sys.path.append('./drive/My Drive/embed/')
            > sys.path.append('./drive/My Drive/embed/scripts')

### 5. Mounting Google Drive with Colab:

    To mount your drive with colab, enter following:

    > from google.colab import drive
    > drive.mount('/content/drive')

    This will prompt for authorization.

    Open the link in a new tab-> you will get a code - copy that back into the prompt and enter. 
    
    You now have access to google drive check.

    The base path for drive is: `'./drive/My Drive/'`

    To run any command from terminal, type: !cmd  in the cell in colab, e.g.:

    !pip install torch

