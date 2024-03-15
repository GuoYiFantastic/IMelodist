import os

# install dependency
os.system("pip install midi2audio")
os.system("pip install music21")
os.system("pip install streamlit")
os.system("pip install transformers")

# run demo
os.system("streamlit run chat/IMelodist_demo.py --server.address=0.0.0.0 --server.port 7860")
