I’ve laid out:

Environment setup & requirements.txt

data_utils.py for unzipping, CSV loading, splitting, and exporting

streamlit_app.py for interactive preview, split/export buttons, and kicking off training

train.py with a PyTorch ResNet-18 training loop

Everything lives under your aiml_image_processing/ folder.


Next steps:

Drop your ZIP + CSVs into data/

pip install -r requirements.txt

Launch the UI via streamlit run streamlit_app.py

Click Split & Export, then Start Training